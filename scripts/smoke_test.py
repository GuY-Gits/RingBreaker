#!/usr/bin/env python3
"""End-to-end smoke test against a running RingBreaker API (HTTP only).

Walks the analyst journey: dashboard loads -> stream produces a BLOCK alert ->
case file with evidence, graph, pattern and timeline -> confirm verdict ->
propagated risk visible -> retrain -> state reflects it.

    python scripts/smoke_test.py --url http://127.0.0.1:8001

Resets the demo engine first, so do not run it against a session you care about.
"""

from __future__ import annotations

import argparse
import sys
import time

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    base = ap.parse_args().url.rstrip("/")
    api = f"{base}/api"
    s = requests.Session()

    def check(cond: bool, msg: str) -> None:
        print(("  ok  " if cond else "  FAIL") + f"  {msg}")
        if not cond:
            raise SystemExit(1)

    def get(path, **kw):
        r = s.get(api + path, timeout=30, **kw)
        r.raise_for_status()
        return r.json()

    def post(path, **kw):
        r = s.post(api + path, timeout=60, **kw)
        r.raise_for_status()
        return r.json()

    print(f"RingBreaker smoke test -> {base}")
    check(s.get(f"{base}/app/", timeout=10).status_code == 200, "dashboard served at /app")
    post("/stream/reset")
    ov = get("/overview")
    check(ov["accounts"] > 0 and ov["history_payments"] > 0, f"graph warmed: {ov['accounts']} accounts, {ov['history_payments']} payments")

    post("/stream/start", json={"rate": 200, "pause_on_block": True})
    deadline = time.time() + 60
    status = get("/stream/status")
    while status["running"] and time.time() < deadline:
        time.sleep(0.3)
        status = get("/stream/status")
    check(bool(status["last_event"]) and "BLOCK" in status["last_event"], f"stream paused on a block: {status['last_event']}")
    alert_id = status["last_event"].split(": ")[1]

    alerts = get("/alerts")
    check(any(a["alert_id"] == alert_id for a in alerts), f"{len(alerts)} open alerts incl. {alert_id}")
    case = get(f"/alerts/{alert_id}")
    check(case["action"] == "BLOCK" and case["overall_risk"] >= 0.7, f"risk {case['overall_risk']:.2f} -> {case['action']}")
    check(len(case["reasons"]) > 0, f"reasons: {case['reasons'][0]['text']}")
    check(len(case["subgraph"]["nodes"]) > 2 and any(e.get("is_trigger") for e in case["subgraph"]["edges"]), "evidence subgraph with the flagged payment")
    # Not every blocked payment sits inside a named pattern (the pair model can
    # block on its own); the case file must still list what was found.
    check(isinstance(case["patterns"], list),
          f"patterns: {', '.join(p['type'] for p in case['patterns']) or 'none (model-driven block)'}")
    check(any(e.get("is_trigger") for e in case["timeline"]), f"timeline: {len(case['timeline'])} events")
    check(len(case["top_factors"]) > 0 and case["counterfactual"]["counterfactual_line"], "SHAP factors and counterfactual")

    v = post(f"/alerts/{alert_id}/verdict", json={"verdict": "confirm"})
    check(v["status"] == "confirmed", f"confirmed {len(v['seeds'])} accounts in {v['elapsed_ms']} ms")
    propagated = [c for c in v["risk_changes"] if not c["seed"]]
    check(len(propagated) > 0, f"risk propagated to {len(propagated)} neighbours")
    seed = v["seeds"][0]
    check(get(f"/accounts/{seed}")["status"] == "confirmed_fraud", f"{seed} now confirmed fraud")
    check(get(f"/alerts/{alert_id}")["status"] == "confirmed", "case reflects the verdict")
    graph = get("/graph/snapshot", params={"alert": alert_id})
    check(any(n.get("status") == "confirmed_fraud" for n in graph["nodes"]), "graph shows confirmed nodes")

    r = post("/admin/retrain")
    check(r["verified_positive"] >= 1, f"retrained -> model {r['model_version']}")
    check(get("/health")["model_version"] == r["model_version"], "new model is live")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
