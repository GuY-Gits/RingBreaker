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
    cols = ["transaction_id", "is_fraud", "ring_id", "amount", "timestamp"]
    header = pd.read_csv(config.PAYMENTS_CSV, nrows=0).columns
    df = pd.read_csv(config.PAYMENTS_CSV, usecols=cols + [c for c in ("group_id",) if c in header])
    return df.set_index("transaction_id")


def _run(engine: Engine, labels: pd.DataFrame, confirm_ring: Optional[str]) -> Dict[str, Any]:
    confirmed = None
    risky_before: set = set()
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
            risky_before = {a for a in engine.accounts if engine.account_risk(a) >= 0.3}

    rings = pd.read_csv(config.RINGS_CSV)
    ring_meta_map = {}
    for _, r_row in rings.iterrows():
        rid = r_row.get("ring_id")
        if rid:
            family = r_row.get("family", r_row.get("ring_type", "unknown"))
            variant = r_row.get("variant", "standard")
            ring_meta_map[rid] = {"family": family, "variant": variant}

    rows = []
    for p in engine.payments:
        lab = labels.loc[p["transaction_id"]]
        r_id = lab["ring_id"] if isinstance(lab["ring_id"], str) else None
        meta = ring_meta_map.get(r_id, {})
        rows.append({
            "transaction_id": p["transaction_id"],
            "flag": p["action"] != "ALLOW",
            "block": p["action"] == "BLOCK",
            "fraud": int(lab["is_fraud"]),
            "ring": r_id,
            "family": meta.get("family"),
            "variant": meta.get("variant"),
            "group": lab["group_id"] if isinstance(lab.get("group_id"), str) else None,
        })
    df = pd.DataFrame(rows)
    tp = int((df.flag & (df.fraud == 1)).sum())
    fp = int((df.flag & (df.fraud == 0)).sum())
    fn = int((~df.flag & (df.fraud == 1)).sum())
    tn = int((~df.flag & (df.fraud == 0)).sum())
    per_ring = {}
    for ring, g in df[df.ring.notna()].groupby("ring"):
        per_ring[ring] = {"payments": int(len(g)), "flagged": int(g.flag.sum()),
                          "recall": round(float(g.flag.mean()), 4)}
    per_family = {}
    for fam, g in df[df.family.notna()].groupby("family"):
        per_family[fam] = {
            "payments": int(len(g)),
            "flagged": int(g.flag.sum()),
            "recall": round(float(g.flag.mean()), 4),
        }
    novel_df = df[df.variant == "held_out_novel"]
    novel_recall = round(float(novel_df.flag.mean()), 4) if len(novel_df) > 0 else None
    # False-positive rate on the benign look-alike groups (hard negatives) vs background.
    benign_df = df[(df.fraud == 0) & df.group.notna()]
    background_df = df[(df.fraud == 0) & df.group.isna()]
    benign_fpr = round(float(benign_df.flag.mean()), 4) if len(benign_df) else None
    background_fpr = round(float(background_df.flag.mean()), 4) if len(background_df) else None
    benign_types = {}
    if config.BENIGN_GROUPS_CSV.exists() and len(benign_df):
        gtype = dict(pd.read_csv(config.BENIGN_GROUPS_CSV)[["group_id", "type"]].values)
        for t, g in benign_df.assign(t=benign_df.group.map(gtype)).groupby("t"):
            benign_types[t] = {"payments": int(len(g)), "flagged": int(g.flag.sum())}

    blocked = df[df.block]
    block_tp = int((blocked.fraud == 1).sum())

    bust = labels[(labels.ring_id == "RING_SLEEPER") & (labels.amount > 1000)]["timestamp"].min()
    lead_hours = None
    if lockstep_first and isinstance(bust, str):
        lead_hours = round((datetime.fromisoformat(bust) - datetime.fromisoformat(lockstep_first)).total_seconds() / 3600, 1)
    # Propagation acts on accounts, and a confirmed mule's everyday payments are
    # labelled genuine, so payment-level metrics alone mis-score it. Report
    # (a) metrics on "new cases" (payments not involving an analyst-confirmed
    # account, which the account action already covers) and (b) account reach.
    ring_accounts = {m for ms in rings["members"] for m in str(ms).split(";")}
    conf = set(engine.confirmed_accounts)
    involves_conf = df.transaction_id.map(
        lambda t: engine.payment_index[t]["sender"] in conf or engine.payment_index[t]["receiver"] in conf)
    new = df[~involves_conf]
    n_tp = int((new.flag & (new.fraud == 1)).sum())
    n_fp = int((new.flag & (new.fraud == 0)).sum())
    new_cases = {"payments": int(len(new)), "tp": n_tp, "fp": n_fp,
                 "precision": round(n_tp / (n_tp + n_fp), 4) if n_tp + n_fp else None,
                 "recall": round(n_tp / int((new.fraud == 1).sum()), 4) if int((new.fraud == 1).sum()) else None}
    reach = None
    if confirmed:
        newly = {a for a in engine.accounts if engine.account_risk(a) >= 0.3} - risky_before - conf
        reach = {"confirmed_accounts": len(conf), "confirmed_ring_accounts": len(conf & ring_accounts),
                 "newly_risky_ring_accounts": len(newly & ring_accounts),
                 "newly_risky_other_accounts": len(newly - ring_accounts)}
    lat = sorted(engine.latencies)
    return {
        "payments": len(df),
        "fraud_payments": int(df.fraud.sum()),
        "flagged": int(df.flag.sum()),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "benign_group_false_positive_rate": benign_fpr,
        "background_false_positive_rate": background_fpr,
        "benign_group_flags": benign_types,
        "tp": tp, "fp": fp, "fn": fn,
        "per_ring": per_ring,
        "per_family": per_family,
        "novel_family_recall": novel_recall,
        "blocked": int(len(blocked)),
        "block_precision": round(block_tp / len(blocked), 4) if len(blocked) else None,
        "block_recall": round(block_tp / (tp + fn), 4) if tp + fn else None,
        "sleeper_lockstep_first_detected": lockstep_first,
        "sleeper_bust_out_start": bust if isinstance(bust, str) else None,
        "sleeper_lead_time_hours": lead_hours,
        "latency_ms_p50": round(lat[len(lat) // 2], 2) if lat else None,
        "latency_ms_p95": round(lat[int(len(lat) * 0.95)], 2) if lat else None,
        "confirmed_alert": confirmed,
        "new_cases": new_cases,
        "confirmation_reach": reach,
    }


def run() -> Dict[str, Any]:
    labels = _labels()
    # backfill=False: metrics cover the held-out stream only, never historical scores.
    baseline = _run(Engine(backfill=False), labels, None)
    confirm = _run(Engine(backfill=False), labels, CONFIRM_RING)
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
