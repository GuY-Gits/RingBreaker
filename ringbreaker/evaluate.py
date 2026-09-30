"""PRD success metrics, measured by replaying the held-out stream through the
real engine (same path as the live API).

    python -m ringbreaker.evaluate

Two runs over identical data:
  * baseline  — no analyst action
  * one confirm — the analyst confirms the first alert of the held-out mule
    chain; measures the extra ring payments caught by propagation (F13)

Ground-truth labels are read here only to score the runs; the engine never
sees them. Writes ``models/evaluation.json`` (shown on the dashboard).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from ringbreaker import config
from ringbreaker.engine import Engine

CONFIRM_RING = "RING_MULE_HO"


def _labels() -> pd.DataFrame:
    df = pd.read_csv(config.PAYMENTS_CSV, usecols=["transaction_id", "is_fraud", "ring_id", "amount", "timestamp"])
    return df.set_index("transaction_id")


def _run(engine: Engine, labels: pd.DataFrame, confirm_ring: Optional[str]) -> Dict[str, Any]:
    confirmed = None
    lockstep_first: Optional[str] = None
    sleeper = set()
    rings = pd.read_csv(config.RINGS_CSV)
    for rid, members in zip(rings["ring_id"], rings["members"]):
        if rid == "RING_SLEEPER":
            sleeper = set(members.split(";"))
    for p in engine.patterns.values():
        if p["type"] == "lockstep_cluster" and len(sleeper & set(p["members"])) >= 3:
            lockstep_first = p["first_detected_at"]
    for row in engine.stream_rows:
        out = engine.score(row, source="evaluation")
        if lockstep_first is None:
            for p in engine.patterns.values():
                if p["type"] == "lockstep_cluster" and len(sleeper & set(p["members"])) >= 3:
                    lockstep_first = p["first_detected_at"]
        if (confirm_ring and confirmed is None and out["alert_id"]
                and labels.at[out["transaction_id"], "ring_id"] == confirm_ring):
            engine.verdict(out["alert_id"], "confirm")
            confirmed = out["alert_id"]

    rows = []
    for p in engine.payments:
        lab = labels.loc[p["transaction_id"]]
        rows.append({"flag": p["action"] != "ALLOW", "block": p["action"] == "BLOCK",
                     "fraud": int(lab["is_fraud"]), "ring": lab["ring_id"] if isinstance(lab["ring_id"], str) else None})
    df = pd.DataFrame(rows)
    tp = int((df.flag & (df.fraud == 1)).sum())
    fp = int((df.flag & (df.fraud == 0)).sum())
    fn = int((~df.flag & (df.fraud == 1)).sum())
    tn = int((~df.flag & (df.fraud == 0)).sum())
    per_ring = {}
    for ring, g in df[df.ring.notna()].groupby("ring"):
        per_ring[ring] = {"payments": int(len(g)), "flagged": int(g.flag.sum()),
                          "recall": round(float(g.flag.mean()), 4)}
    bust = labels[(labels.ring_id == "RING_SLEEPER") & (labels.amount > 1000)]["timestamp"].min()
    lead_hours = None
    if lockstep_first and isinstance(bust, str):
        lead_hours = round((datetime.fromisoformat(bust) - datetime.fromisoformat(lockstep_first)).total_seconds() / 3600, 1)
    lat = sorted(engine.latencies)
    return {
        "payments": len(df),
        "fraud_payments": int(df.fraud.sum()),
        "flagged": int(df.flag.sum()),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "tp": tp, "fp": fp, "fn": fn,
        "per_ring": per_ring,
        "sleeper_lockstep_first_detected": lockstep_first,
        "sleeper_bust_out_start": bust if isinstance(bust, str) else None,
        "sleeper_lead_time_hours": lead_hours,
        "latency_ms_p50": round(lat[len(lat) // 2], 2) if lat else None,
        "latency_ms_p95": round(lat[int(len(lat) * 0.95)], 2) if lat else None,
        "confirmed_alert": confirmed,
    }


def run() -> Dict[str, Any]:
    labels = _labels()
    baseline = _run(Engine(), labels, None)
    confirm = _run(Engine(), labels, CONFIRM_RING)
    extra = (confirm["per_ring"].get(CONFIRM_RING, {}).get("flagged", 0)
             - baseline["per_ring"].get(CONFIRM_RING, {}).get("flagged", 0))
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "one_confirmation": confirm,
        "extra_ring_payments_caught_after_confirm": extra,
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
