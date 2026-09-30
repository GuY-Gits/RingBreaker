"""End-to-end API contract tests against the real engine, models and data.

Covers the PRD API table (F9-F14, F19) plus the dashboard endpoints. A booted
engine replays the held-out stream until real alerts fire; nothing is mocked.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ringbreaker.api.main import app, get_engine, set_engine
from ringbreaker.engine import Engine


@pytest.fixture(scope="module")
def client():
    set_engine(Engine())
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def alert_id(client):
    """Step the stream until the first BLOCK alert fires."""
    for _ in range(40):
        client.post("/api/stream/step", params={"n": 50})
        blocked = [a for a in client.get("/alerts").json() if a["action"] == "BLOCK"]
        if blocked:
            return blocked[0]["alert_id"]
    pytest.fail("stream produced no BLOCK alert")


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["accounts"] >= 2000


def test_score_contract_and_validation(client):
    clock = get_engine().clock.isoformat()
    ok = client.post("/score", json={"sender": "U00001", "receiver": "U00002", "amount": 250.0,
                                      "timestamp": clock, "device": "DEV_T1"})
    assert ok.status_code == 200
    data = ok.json()
    for key in ("risk_score", "sender_anomaly", "receiver_mule_propensity", "relationship_plausibility"):
        assert 0.0 <= data[key] <= 1.0
    assert data["action"] in ("ALLOW", "WARN_SENDER", "HOLD_RECEIVER", "BLOCK", "REVIEW")
    assert set(data["signals"]) == {"pair_risk", "anomaly", "coordination", "network_risk"}

    bad = [
        {"sender": "U00001", "receiver": "U00001", "amount": 5, "timestamp": clock},
        {"sender": "U00001", "receiver": "U00002", "amount": -5, "timestamp": clock},
        {"sender": "U 1", "receiver": "U00002", "amount": 5, "timestamp": clock},
        {"sender": "U00001", "receiver": "U00002", "amount": 5, "timestamp": "not-a-date"},
    ]
    for body in bad:
        assert client.post("/score", json=body).status_code == 422, body
    # NaN/inf are rejected at the schema layer (JSON cannot carry them; strings are refused)
    assert client.post("/score", json={"sender": "A1", "receiver": "B1", "amount": "inf",
                                       "timestamp": clock}).status_code == 422


def test_out_of_order_payment_rejected(client):
    """Temporal correctness: a payment older than the clock would see the future."""
    resp = client.post("/score", json={"sender": "U00001", "receiver": "U00002", "amount": 10,
                                       "timestamp": "2020-01-01T00:00:00"})
    assert resp.status_code == 409


def test_register_account(client):
    clock = get_engine().clock.isoformat()
    resp = client.post("/accounts", json={"account_id": "NEW_ACC_1", "signup_at": clock,
                                          "device": "DEV_NEW", "phone": "+910000000001"})
    assert resp.status_code == 200
    profile = client.get("/accounts/NEW_ACC_1").json()
    assert {i["type"] for i in profile["identities"]} >= {"device", "phone"}


def test_alert_queue_and_case_file(client, alert_id):
    queue = client.get("/alerts").json()
    risks = [a["overall_risk"] for a in queue]
    assert risks == sorted(risks, reverse=True)
    case = client.get(f"/alerts/{alert_id}").json()
    for key in ("payment", "sub_scores", "signals", "reasons", "patterns", "subgraph", "timeline",
                "top_factors", "counterfactual", "summary", "parties", "relationship", "suspects"):
        assert key in case, key
    assert case["subgraph"]["nodes"] and case["subgraph"]["edges"]
    assert any(e.get("is_trigger") for e in case["subgraph"]["edges"])
    assert any(ev.get("is_trigger") for ev in case["timeline"])
    assert case["top_factors"] and "label" in case["top_factors"][0]
    assert case["counterfactual"]["counterfactual_line"]
    assert case["payment"]["sender"] in case["summary"]
    assert client.get("/alerts/NOPE").status_code == 404


def test_confirm_verdict_propagates_risk(client, alert_id):
    case = client.get(f"/alerts/{alert_id}").json()
    resp = client.post(f"/alerts/{alert_id}/verdict", json={"verdict": "confirm"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "confirmed"
    assert set(body["seeds"]) == set(case["suspects"])
    assert any(not c["seed"] and c["risk_after"] > c["risk_before"] for c in body["risk_changes"])
    for seed in body["seeds"]:
        assert client.get(f"/accounts/{seed}").json()["status"] == "confirmed_fraud"
    # second verdict on the same alert is a conflict
    assert client.post(f"/alerts/{alert_id}/verdict", json={"verdict": "clear"}).status_code == 409
    assert client.get(f"/alerts/{alert_id}").json()["status"] == "confirmed"


def test_graph_snapshot_modes(client, alert_id):
    live = client.get("/graph/snapshot").json()
    assert live["nodes"] and all(0.0 <= n["risk"] <= 1.0 for n in live["nodes"])
    focus = client.get("/graph/snapshot", params={"focus": "U00001"}).json()
    assert "U00001" in focus["highlight"]
    by_alert = client.get("/graph/snapshot", params={"alert": alert_id}).json()
    assert by_alert["edges"]
    assert client.get("/graph/snapshot", params={"focus": "MISSING"}).status_code == 404


def test_dashboard_endpoints(client):
    ov = client.get("/api/overview").json()
    assert ov["scored_payments"] > 0 and "stream" in ov and ov["timeline"]
    assert client.get("/api/payments/recent", params={"limit": 5}).json()
    pats = client.get("/api/patterns").json()
    assert pats and {"id", "type", "members"} <= set(pats[0])
    accs = client.get("/api/accounts", params={"limit": 5}).json()
    assert len(accs) == 5
    learning = client.get("/api/learning").json()
    assert learning["verified"]["confirmed"] >= 1


def test_retrain_uses_verified_labels(client):
    resp = client.post("/admin/retrain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["verified_positive"] >= 1
    assert body["model_version"] == get_engine().models.version
    assert "after" in body["metrics"]
