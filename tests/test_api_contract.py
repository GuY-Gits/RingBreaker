"""Integration and Contract tests for RingBreaker API (PRD F9, F11, F12, F13, F14, F19)."""

import pytest
from fastapi.testclient import TestClient
from ringbreaker.api.main import app, _transaction_graph, _account_feature_cache, _alerts


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_score_payment_contract(client):
    req = {
        "sender": "U00100",
        "receiver": "U00200",
        "amount": 5500.0,
        "device": "DEV_TEST_1",
        "timestamp": "2026-03-20 12:00:00",
    }
    resp = client.post("/score", json=req)
    assert resp.status_code == 200
    data = resp.json()

    # Assert exact PRD F10 fields
    assert "risk_score" in data
    assert "sender_anomaly" in data
    assert "receiver_mule_propensity" in data
    assert "relationship_plausibility" in data
    assert "action" in data

    assert 0.0 <= data["risk_score"] <= 1.0
    assert 0.0 <= data["sender_anomaly"] <= 1.0
    assert 0.0 <= data["receiver_mule_propensity"] <= 1.0
    assert 0.0 <= data["relationship_plausibility"] <= 1.0
    assert data["action"] in (
        "ALLOW", "REVIEW", "BLOCK", "WARN_SENDER", "HOLD_RECEIVER",
        "allow", "review", "block", "warn_sender", "hold_receiver"
    )


def test_canonical_risk_and_action_boundaries():
    """Verify canonical boundary thresholds, adaptive sub-score routing, and format representation."""
    from ringbreaker.scoring.action import determine_action, format_canonical_risk

    # Required boundary tests without sub_scores (backward compatibility)
    assert determine_action(0.10) == "ALLOW"
    assert determine_action(0.29) == "ALLOW"
    assert determine_action(0.30) == "REVIEW"
    assert determine_action(0.69) == "REVIEW"
    assert determine_action(0.70) == "BLOCK"
    assert determine_action(0.95) == "BLOCK"

    # F11 Adaptive action routing in review tier (0.30 <= risk < 0.70)
    # 1. Receiver mule propensity dominates -> HOLD_RECEIVER
    assert determine_action(0.45, {"sender_anomaly": 0.20, "receiver_mule_propensity": 0.75}) == "HOLD_RECEIVER"

    # 2. Sender anomaly dominates -> WARN_SENDER
    assert determine_action(0.45, {"sender_anomaly": 0.80, "receiver_mule_propensity": 0.25}) == "WARN_SENDER"

    # 3. Floor invariance: risk < 0.30 is ALWAYS ALLOW, regardless of isolated sub-scores
    assert determine_action(0.15, {"sender_anomaly": 0.95, "receiver_mule_propensity": 0.90}) == "ALLOW"

    # 4. Ceiling invariance: risk >= 0.70 is ALWAYS BLOCK, regardless of low sub-scores
    assert determine_action(0.75, {"sender_anomaly": 0.05, "receiver_mule_propensity": 0.05}) == "BLOCK"

    # Decimal vs percentage verification
    allow_payload = format_canonical_risk(0.015)
    assert allow_payload["overall_risk"] == 0.015
    assert allow_payload["risk_percent"] == 1.5
    assert allow_payload["action"] == "ALLOW"

    block_payload = format_canonical_risk(0.8626)
    assert block_payload["overall_risk"] == 0.8626
    assert block_payload["risk_percent"] == 86.26
    assert block_payload["action"] == "BLOCK"

    hold_payload = format_canonical_risk(0.50, {"sender_anomaly": 0.2, "receiver_mule_propensity": 0.6})
    assert hold_payload["overall_risk"] == 0.50
    assert hold_payload["action"] == "HOLD_RECEIVER"


def test_alert_generation_and_case_file_schema(client):
    # Set high risk cache so transaction exceeds threshold
    _account_feature_cache["MULE_SRC"] = {
        "risk_boost": 0.8,
        "anomaly_score": 0.85,
        "coordination_score": 0.75,
    }
    req = {
        "sender": "MULE_SRC",
        "receiver": "MULE_DST",
        "amount": 25000.0,
        "device": "DEV_MULE_SHARED",
        "timestamp": "2026-03-21 02:30:00",
    }
    resp = client.post("/score", json=req)
    assert resp.status_code == 200
    score_data = resp.json()

    alert_id = score_data.get("alert_id")
    assert alert_id is not None, "High-risk transaction must produce an alert_id"
    assert score_data["action"] in ("REVIEW", "BLOCK", "WARN_SENDER", "HOLD_RECEIVER")

    # GET /alerts
    alerts_resp = client.get("/alerts")
    assert alerts_resp.status_code == 200
    alerts_list = alerts_resp.json()
    assert any(a["alert_id"] == alert_id for a in alerts_list)

    # GET /alerts/{id} (F12 Case File)
    case_resp = client.get(f"/alerts/{alert_id}")
    assert case_resp.status_code == 200
    case = case_resp.json()

    # Verify all F12, F16, F17 case file fields
    assert case["alert_id"] == alert_id
    assert "subgraph" in case
    assert "nodes" in case["subgraph"]
    assert "edges" in case["subgraph"]
    assert "timeline" in case
    assert "top_factors" in case
    assert "counterfactual" in case
    assert "counterfactual_line" in case["counterfactual"]
    assert "evasion_cost" in case["counterfactual"]
    assert "summary" in case
    assert len(case["summary"]) > 0


def test_confirm_verdict_and_risk_propagation(client):
    # Set high risk cache so transaction triggers alert
    _account_feature_cache["SEED_A"] = {
        "risk_boost": 0.8,
        "anomaly_score": 0.85,
        "coordination_score": 0.75,
    }
    req = {
        "sender": "SEED_A",
        "receiver": "SEED_B",
        "amount": 9999.0,
        "timestamp": "2026-03-22 10:00:00",
    }
    score_resp = client.post("/score", json=req)
    alert_id = score_resp.json()["alert_id"]
    assert alert_id is not None

    # Confirm fraud (F13 Personalized PageRank)
    verdict_resp = client.post(f"/alerts/{alert_id}/verdict", json={"verdict": "confirm"})
    assert verdict_resp.status_code == 200
    data = verdict_resp.json()

    assert data["alert_id"] == alert_id
    assert data["verdict"] == "confirm"
    assert "risk_changes" in data
    assert len(data["risk_changes"]) > 0
    # Seed nodes should have risk_after boosted
    seed_changes = [c for c in data["risk_changes"] if c["account_id"] in ("SEED_A", "SEED_B")]
    assert len(seed_changes) > 0
    assert seed_changes[0]["risk_after"] >= 0.50


def test_graph_snapshot_endpoint(client):
    resp = client.get("/graph/snapshot")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert "stats" in data
    assert "total_nodes" in data["stats"]
    assert "total_edges" in data["stats"]


def test_admin_retrain_endpoint(client):
    resp = client.post("/admin/retrain")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_version" in data
    assert "metrics" in data
    assert data["metrics"].get("status") == "success"


def test_preloaded_alerts_schema_contract(client):
    """Ensure any preloaded alert conforms to the full CaseFile schema."""
    resp = client.get("/alerts")
    assert resp.status_code == 200
    alerts = resp.json()
    assert len(alerts) > 0

    first_alert_id = alerts[0]["alert_id"]
    detail_resp = client.get(f"/alerts/{first_alert_id}")
    assert detail_resp.status_code == 200
    case = detail_resp.json()

    assert "sub_scores" in case
    assert "sender_anomaly" in case["sub_scores"]
    assert "receiver_mule_propensity" in case["sub_scores"]
    assert "relationship_plausibility" in case["sub_scores"]
    assert "members" in case
    assert isinstance(case["members"], list)
    assert "top_factors" in case
    assert isinstance(case["top_factors"], list)
    assert "timeline" in case
    assert isinstance(case["timeline"], list)
