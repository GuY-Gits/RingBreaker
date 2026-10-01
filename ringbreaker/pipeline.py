"""One-command build: simulate data, build features, train both models.

    python -m ringbreaker.pipeline            # regenerate data + retrain
    python -m ringbreaker.pipeline --no-generate   # retrain on existing CSVs

Steps (PRD build order 1, 2, 7, 9):
  1. Simulator (F1) writes users.csv, payments.csv, rings.csv.
  2. Shared online feature store builds pair + behaviour features causally.
  3. XGBoost pair model (model 1) trained on the first 70% of the timeline,
     early-stopped on the next 15%; the last 15% (the live stream) is never seen.
  4. Extended Isolation Forest (model 2) trained unsupervised on the first 70%,
     calibrated against its own training-score distribution.
  5. Metadata with held-out pair-model metrics is written for the dashboard.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from ringbreaker import config
from ringbreaker.anomaly.eif_model import CalibratedEIF, ExtendedIsolationForest
from ringbreaker.engine.risk import fit_calibration, raw_fused
from ringbreaker.features.online import BEHAVIOUR_FEATURE_NAMES, PAIR_FEATURE_NAMES
from ringbreaker.features.pair_features import build_feature_frame
from ringbreaker.simulator import generate as sim
from ringbreaker.split import timeline_split

SEED = 42


def generate_data(seed: int = SEED) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = sim.SimulatorConfig(random_seed=seed, output_dir=str(config.DATA_DIR))
    users, payments, rings_df = sim.generate_dataset(cfg)
    users.to_csv(config.USERS_CSV, index=False)
    payments.to_csv(config.PAYMENTS_CSV, index=False)
    rings_df.to_csv(config.RINGS_CSV, index=False)
    payments.attrs["benign_groups"].to_csv(config.BENIGN_GROUPS_CSV, index=False)


def split_bounds(frame: pd.DataFrame) -> tuple[int, int]:
    split = timeline_split(frame["timestamp"])
    return split.train_rows, split.stream_start_row


def train_pair_model(frame: pd.DataFrame) -> Dict[str, Any]:
    train_end, val_end = split_bounds(frame)
    X = frame[PAIR_FEATURE_NAMES].astype(float)
    label_col = "label_observed" if "label_observed" in frame.columns else "is_fraud"
    y_signal = frame[label_col].astype(int)
    X_tr, y_tr = X.iloc[:train_end], y_signal.iloc[:train_end]
    X_va, y_va = X.iloc[train_end:val_end], y_signal.iloc[train_end:val_end]
    X_te, y_te = X.iloc[val_end:], frame["is_fraud"].iloc[val_end:].astype(int)

    pos = max(int(y_tr.sum()), 1)
    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=(len(y_tr) - pos) / pos,
        eval_metric="aucpr",
        random_state=SEED,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(config.PAIR_MODEL_PATH))

    probs = clf.predict_proba(X_te)[:, 1]
    return {
        "train_rows": int(train_end),
        "validation_rows": int(val_end - train_end),
        "heldout_rows": int(len(frame) - val_end),
        "train_fraud": int(y_tr.sum()),
        "heldout_fraud": int(y_te.sum()),
        "heldout_roc_auc": _safe_metric(roc_auc_score, y_te, probs),
        "heldout_pr_auc": _safe_metric(average_precision_score, y_te, probs),
    }


def train_eif(frame: pd.DataFrame) -> Dict[str, Any]:
    train_end, _ = split_bounds(frame)
    X_tr = frame[BEHAVIOUR_FEATURE_NAMES].iloc[:train_end].to_numpy(dtype=np.float64)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6)
    np.random.seed(SEED)
    model = ExtendedIsolationForest(n_trees=128, sample_size=256, extension_level=X_tr.shape[1] - 1)
    model.fit(X_tr)
    calibrated = CalibratedEIF(model, model.raw_scores(X_tr))
    with open(config.EIF_MODEL_PATH, "wb") as fh:
        pickle.dump(calibrated, fh)
    return {"train_rows": int(train_end), "features": BEHAVIOUR_FEATURE_NAMES, "trees": 128}


def calibrate_on_validation(frame: pd.DataFrame) -> Dict[str, Any]:
    """Fit alert/block cut-offs on the validation slice (never the held-out stream)."""
    train_end, val_end = split_bounds(frame)
    val = frame.iloc[train_end:val_end]
    booster = xgb.Booster()
    booster.load_model(str(config.PAIR_MODEL_PATH))
    pair = booster.predict(xgb.DMatrix(val[PAIR_FEATURE_NAMES].astype(float).values, feature_names=PAIR_FEATURE_NAMES))
    # Lockstep coordination is computed online only; it is 0 for almost all payments.
    raw = [raw_fused(p, a, 0.0) for p, a in zip(pair, val["eif_anomaly"])]
    # Recorded (noisy) outcomes, as an institution would have them; never ground truth.
    label_col = "label_observed" if "label_observed" in val.columns else "is_fraud"
    return fit_calibration(raw, val[label_col].astype(int).to_numpy())


def _safe_metric(fn, y, s) -> float | None:
    try:
        return round(float(fn(y, s)), 4)
    except ValueError:
        return None


def run(generate: bool = True, seed: int = SEED, evaluate: bool = True) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    if generate or not config.PAYMENTS_CSV.exists():
        print("[1/4] Simulating users, payments and planted rings ...")
        generate_data(seed)
    payments = pd.read_csv(config.PAYMENTS_CSV)
    payments["timestamp"] = pd.to_datetime(payments["timestamp"])
    payments = payments.sort_values("timestamp", kind="stable").reset_index(drop=True)
    users = pd.read_csv(config.USERS_CSV)

    print("[2/4] Building causal features with the shared online feature store ...")
    frame = build_feature_frame(payments, users, include_behaviour=True)
    save_cols = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud"]
    if "label_observed" in frame.columns:
        save_cols.append("label_observed")
    save_cols.extend(["ring_id", *PAIR_FEATURE_NAMES])
    frame[save_cols].to_csv(config.PAIR_FEATURES_CSV, index=False)

    print("[3/4] Training XGBoost pair model (70% train / 15% validation) ...")
    pair_metrics = train_pair_model(frame)
    print("[4/4] Training Extended Isolation Forest behavioural model ...")
    eif_meta = train_eif(frame)

    # Anomaly score per payment (a model output, not a label) — reused by
    # calibration here and when retraining.
    with open(config.EIF_MODEL_PATH, "rb") as fh:
        eif = pickle.load(fh)
    frame["eif_anomaly"] = eif.score(np.nan_to_num(frame[BEHAVIOUR_FEATURE_NAMES].to_numpy(dtype=np.float64)))
    calibration = calibrate_on_validation(frame)
    save_cols.append("eif_anomaly")
    frame[save_cols].to_csv(config.PAIR_FEATURES_CSV, index=False)

    metadata = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "pair_model": {"features": PAIR_FEATURE_NAMES, **pair_metrics},
        "eif": eif_meta,
        "risk_calibration": calibration,
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(pair_metrics, indent=2))
    from ringbreaker.backfill import run as run_backfill

    print("[+] Scoring the pre-stream period for the dashboard's historical view ...")
    print(json.dumps(run_backfill()))
    if evaluate:
        from ringbreaker.evaluate import run as run_evaluation

        print("[5/5] Replaying the held-out stream through the engine for PRD metrics ...")
        ev = run_evaluation()
        print(json.dumps({k: ev["baseline"][k] for k in ("precision", "recall", "false_positive_rate")}, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RingBreaker data and models")
    parser.add_argument("--no-generate", action="store_true", help="reuse existing CSVs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-evaluate", action="store_true", help="skip the held-out engine replay")
    args = parser.parse_args()
    run(generate=not args.no_generate, seed=args.seed, evaluate=not args.no_evaluate)


if __name__ == "__main__":
    main()
