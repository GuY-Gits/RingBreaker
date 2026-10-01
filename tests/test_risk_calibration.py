"""Decision calibration and propagation rules (engine/risk.py, engine/engine.py)."""

from datetime import datetime, timedelta

import numpy as np

from ringbreaker.engine import Engine
from ringbreaker.engine.risk import calibrate, fit_calibration, fuse


def test_calibration_hits_budgets_on_fit_data():
    rng = np.random.default_rng(0)
    raw = rng.beta(1, 8, size=20000)
    cal = fit_calibration(raw, alert_budget=0.01, block_budget=0.002)
    risk = np.array([calibrate(r, cal) for r in raw])
    assert abs((risk >= 0.30).mean() - 0.01) < 0.002
    assert abs((risk >= 0.70).mean() - 0.002) < 0.001


def test_calibration_with_labels_picks_best_f1_cutoff():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.01).astype(int)
    raw = np.clip(np.where(y == 1, rng.normal(0.7, 0.1, 20000), rng.normal(0.2, 0.1, 20000)), 0, 1)
    cal = fit_calibration(raw, y)

    def f1(cut):
        pred = raw >= cut
        tp = (pred & (y == 1)).sum()
        return 2 * tp / (pred.sum() + y.sum())

    assert cal["alert_rule"] == "max_f1"
    assert all(f1(cal["alert_raw"]) >= f1(c) - 1e-9 for c in np.linspace(0.05, 0.95, 181))
    assert abs(cal["validation"]["f1"] - f1(cal["alert_raw"])) < 1e-3
    assert cal["alert_raw"] < cal["block_raw"]
    # no positive labels -> falls back to the review budget
    assert fit_calibration(raw, np.zeros(20000, dtype=int))["alert_rule"] == "budget"


def test_calibration_is_monotone_and_bounded():
    cal = {"alert_raw": 0.4, "block_raw": 0.8}
    xs = np.linspace(0, 1, 501)
    ys = [calibrate(x, cal) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))
    assert ys[0] == 0.0 and abs(ys[-1] - 1.0) < 1e-9
    assert abs(calibrate(0.4, cal) - 0.30) < 1e-9 and abs(calibrate(0.8, cal) - 0.70) < 1e-9


def test_confirmed_network_risk_dominates_fusion():
    cal = {"alert_raw": 0.6, "block_raw": 0.8}
    assert fuse(0.01, 0.1, 0.0, 1.0, cal) == 1.0
    assert fuse(0.01, 0.1, 0.0, 0.0, cal) < 0.30


def _engine_with_confirmed(account: str) -> Engine:
    e = Engine(load_history=False)
    e.accounts.setdefault(account, {"id": account, "signup_at": None})
    e.confirmed_accounts.add(account)
    e.propagated[account] = 1.0
    return e


def test_taint_follows_laundering_not_friendship():
    t0 = datetime(2026, 3, 1, 12, 0)
    e = _engine_with_confirmed("MULE")
    # small everyday payment from a confirmed mule: no taint
    e.score({"sender": "MULE", "receiver": "FRIEND", "amount": 200, "timestamp": t0.isoformat()})
    assert e.propagated.get("FRIEND", 0.0) == 0.0
    # laundering-sized transfer taints the next hop
    e.score({"sender": "MULE", "receiver": "HOP1", "amount": 5000,
             "timestamp": (t0 + timedelta(minutes=5)).isoformat()})
    assert e.propagated["HOP1"] > 0.5
    # HOP1 forwards quickly -> HOP2 tainted; a slow forward much later does not cascade
    e.score({"sender": "HOP1", "receiver": "HOP2", "amount": 4800,
             "timestamp": (t0 + timedelta(minutes=30)).isoformat()})
    assert e.propagated["HOP2"] > 0.3
    e.score({"sender": "HOP1", "receiver": "LATER", "amount": 4000,
             "timestamp": (t0 + timedelta(days=3)).isoformat()})
    assert e.propagated.get("LATER", 0.0) == 0.0
