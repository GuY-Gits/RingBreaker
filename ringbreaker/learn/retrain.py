"""F19: retrain the pair model with analyst-verified labels.

Training data = the historical (pre-stream) feature rows with their recorded
outcomes + every stream payment an analyst confirmed (label 1) or cleared
(label 0), using the exact feature vector computed when it was scored.

Evaluation = held-out stream payments that were NOT verified, labelled with the
simulator's ground truth. Ground truth is read here for reporting only; it is
never visible to the scoring path.

    python -m ringbreaker.learn.retrain          # retrain on historical labels only
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from ringbreaker import config
from ringbreaker.features.online import PAIR_FEATURE_NAMES
from ringbreaker.split import timeline_split

VERIFIED_WEIGHT = 5.0  # an analyst-verified outcome counts more than a bulk label
THRESHOLD = 0.5


def _historical() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(config.PAIR_FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    split = timeline_split(df["timestamp"])
    hist = df.iloc[: split.stream_start_row]
    return hist[PAIR_FEATURE_NAMES].astype(float), hist["is_fraud"].astype(int)


def _ground_truth() -> Dict[str, int]:
    df = pd.read_csv(config.PAYMENTS_CSV, usecols=["transaction_id", "is_fraud"])
    return dict(zip(df["transaction_id"].astype(str), df["is_fraud"].astype(int)))


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    pred = p >= THRESHOLD
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if y.sum() else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def retrain_with_verified(
    verified: Sequence[Tuple[Dict[str, float], int]],
    version: str,
    stream_rows: Optional[List[Tuple[str, Dict[str, float]]]] = None,
    current: Optional[xgb.Booster] = None,
    verified_ids: Optional[set] = None,
) -> Dict[str, Any]:
    X_hist, y_hist = _historical()
    weights = np.ones(len(X_hist))
    if verified:
        X_ver = pd.DataFrame([f for f, _ in verified])[PAIR_FEATURE_NAMES].astype(float)
        y_ver = pd.Series([int(l) for _, l in verified])
        X = pd.concat([X_hist, X_ver], ignore_index=True)
        y = pd.concat([y_hist, y_ver], ignore_index=True)
        weights = np.concatenate([weights, np.full(len(X_ver), VERIFIED_WEIGHT)])
    else:
        X, y = X_hist, y_hist
    pos = max(int(y.sum()), 1)
    clf = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.8, scale_pos_weight=(len(y) - pos) / pos,
        eval_metric="aucpr", random_state=42,
    )
    clf.fit(X, y, sample_weight=weights, verbose=False)
    booster = clf.get_booster()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.RETRAINED_MODEL_PATH))

    evaluation: Dict[str, Any] = {}
    if stream_rows:
        truth = _ground_truth()
        excluded = verified_ids or set()
        rows = [(tid, f) for tid, f in stream_rows if tid in truth and tid not in excluded]
        if rows:
            y_eval = np.array([truth[tid] for tid, _ in rows])
            dm = xgb.DMatrix(
                np.array([[f[n] for n in PAIR_FEATURE_NAMES] for _, f in rows], dtype=np.float32),
                feature_names=PAIR_FEATURE_NAMES,
            )
            evaluation = {
                "rows": len(rows),
                "fraud": int(y_eval.sum()),
                "after": _metrics(y_eval, booster.predict(dm)),
            }
            if current is not None:
                evaluation["before"] = _metrics(y_eval, current.predict(dm))
    return {
        "booster": booster,
        "model_path": str(config.RETRAINED_MODEL_PATH),
        "train_rows": int(len(X)),
        "verified_positive": int(sum(1 for _, l in verified if l == 1)),
        "verified_negative": int(sum(1 for _, l in verified if l == 0)),
        "metrics": evaluation,
    }


def main() -> None:
    out = retrain_with_verified([], version="cli")
    out.pop("booster")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
