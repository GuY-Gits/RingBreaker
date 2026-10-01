"""Historical backfill: score the pre-stream period once and cache the result.

    python -m ringbreaker.backfill

The training and validation periods (first 85% of the timeline) are replayed
through the engine's normal scoring path, in order, so every score uses only
earlier payments. The result is cached in ``models/history_backfill.json.gz``
and loaded at start-up so the dashboard opens with alerts, risky accounts and
decision history already in place; the live stream then plays only the
held-back 15%.

Caveat: the pair model was trained on the first 70% of the timeline, so
historical scores are in-sample (better than the model would do on new data).
They are for context only and are excluded from all reported metrics.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from typing import Any, Dict, Optional

from ringbreaker import config

CACHE_PATH = config.MODELS_DIR / "history_backfill.json.gz"
SLOW_PATH_EVERY = 500  # patterns only matter at alert time during the backfill


def _cache_key(models: Any) -> str:
    h = hashlib.sha1()
    h.update(config.PAYMENTS_CSV.read_bytes())
    h.update(open(models.pair_path, "rb").read())
    h.update(json.dumps(models.calibration, sort_keys=True).encode())
    return h.hexdigest()


def load_cache(models: Any) -> Optional[Dict[str, Any]]:
    """Return the cache if it matches the current data, model and calibration."""
    if not CACHE_PATH.exists() or not config.PAYMENTS_CSV.exists():
        return None
    with gzip.open(CACHE_PATH, "rt", encoding="utf-8") as fh:
        cache = json.load(fh)
    return cache if cache.get("key") == _cache_key(models) else None


def run() -> Dict[str, Any]:
    from ringbreaker.engine import Engine

    t0 = time.time()
    previous = config.SLOW_PATH_EVERY
    config.SLOW_PATH_EVERY = SLOW_PATH_EVERY
    try:
        engine = Engine(replay="history", backfill=False)
        for row in engine.stream_rows:
            engine.score(row, source="history")
    finally:
        config.SLOW_PATH_EVERY = previous

    payments: Dict[str, list] = {}
    alerts: Dict[str, Dict[str, Any]] = {}
    patterns: Dict[str, Dict[str, Any]] = {}
    for p in engine.payments:
        sig, sub = p["signals"], p["sub_scores"]
        payments[p["transaction_id"]] = [
            p["overall_risk"], sig["pair_risk"], sig["anomaly"], sig["coordination"],
            sub["sender_anomaly"], sub["receiver_mule_propensity"], sub["relationship_plausibility"], p["action"],
        ]
        if p["alert_id"]:
            ids = engine.alerts[p["alert_id"]]["pattern_ids"]
            for pid in ids:
                patterns[pid] = engine.patterns[pid]
            alerts[p["transaction_id"]] = {"reasons": p["reasons"], "features": p["_features"],
                                           "shap": p["_shap"], "context": p["_context"], "patterns": ids}
    cache = {"key": _cache_key(engine.models), "model_version": engine.models.version,
             "payments": payments, "alerts": alerts, "patterns": patterns}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_PATH, "wt", encoding="utf-8") as fh:
        json.dump(cache, fh, default=str)
    return {"payments": len(payments), "alerts": len(alerts), "seconds": round(time.time() - t0, 1),
            "bytes": CACHE_PATH.stat().st_size}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
