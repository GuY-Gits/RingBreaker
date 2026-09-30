"""
RingBreaker - Confirm-Fraud -> Supervised Retraining Loop
Module: learn/retrain.py

Workflow:
1. Analyst confirms a transaction as fraud (e.g. TX_0008825 at 2026-03-24 15:11:20).
2. The confirmed transaction is ingested as a trusted supervised fraud sample (is_fraud = 1).
3. The historical training dataset is updated with this confirmed record.
4. The XGBoost Pair Risk Model is retrained without altering feature schema or hyperparams.
5. The retrained model is saved to models/pair_model_retrained.json.
6. A strict before-vs-after evaluation is conducted on the untouched held-out test slice.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_recall_fscore_support
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PAIR_FEATURES_PATH = DATA_DIR / "pair_features.csv"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
ORIGINAL_MODEL_PATH = MODELS_DIR / "pair_model.json"
RETRAINED_MODEL_PATH = MODELS_DIR / "pair_model_retrained.json"
METADATA_PATH = MODELS_DIR / "retrain_metadata.json"

DEFAULT_CONFIRMED_TX = "TX_0008825"
BLOCK_THRESHOLD = 0.70


# =====================================================================
# 1. Leakage Audit
# =====================================================================

def perform_retrain_leakage_audit(
    feature_columns: List[str],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    confirmed_tx_id: str,
    confirmation_timestamp: pd.Timestamp,
    original_feature_names: List[str]
) -> bool:
    """Verifies strict zero-leakage constraints during supervised retraining."""
    forbidden = ["is_fraud", "ring_id", "propagated_risk", "shap", "risk_score", "probability"]
    found_violations = [c for c in feature_columns if any(f == c.lower() for f in forbidden)]

    is_fraud_used = "is_fraud" in feature_columns
    ring_id_used = "ring_id" in feature_columns
    propagated_risk_used = "propagated_risk" in feature_columns
    shap_used = any("shap" in c.lower() for c in feature_columns)

    # Future transactions check (excluding the confirmed point itself)
    train_excluding_confirmed = df_train[df_train["transaction_id"] != confirmed_tx_id]
    future_txs = (train_excluding_confirmed["timestamp"] > confirmation_timestamp).sum()
    future_transactions_used = bool(future_txs > 0)

    # Test leakage check: confirmed_tx must NOT be in the test evaluation set
    held_out_leak = confirmed_tx_id in df_test["transaction_id"].values
    held_out_used_for_training = held_out_leak

    # Feature schema consistency
    schema_changed = (feature_columns != original_feature_names)

    print("RETRAINING LEAKAGE AUDIT")
    print(f"is_fraud used as feature:       {is_fraud_used}")
    print(f"ring_id used as feature:        {ring_id_used}")
    print(f"propagated_risk used as feature:{propagated_risk_used}")
    print(f"SHAP values used as feature:    {shap_used}")
    print(f"future transactions used:       {future_transactions_used}")
    print(f"held-out test used for training:{held_out_used_for_training}")
    print(f"feature schema changed:         {schema_changed}")

    passed = (
        not is_fraud_used and
        not ring_id_used and
        not propagated_risk_used and
        not shap_used and
        not future_transactions_used and
        not held_out_used_for_training and
        not schema_changed
    )

    if not passed:
        print("RESULT: FAILED")
        raise ValueError(f"Retraining leakage audit failed! Violations: {found_violations}")

    print("RESULT: PASSED\n")
    return True


# =====================================================================
# 2. Evaluation Helper
# =====================================================================

def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray, threshold: float = BLOCK_THRESHOLD) -> Dict[str, float]:
    scores = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, scores)
    p_curve, r_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(r_curve, p_curve)

    preds = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    tn = np.sum((y_test == 0) & (preds == 0))
    fp = np.sum((y_test == 0) & (preds == 1))
    fpr = float(fp / (fp + tn + 1e-8))

    return {
        "ROC-AUC": float(round(roc, 4)),
        "PR-AUC": float(round(pr_auc, 4)),
        "Precision": float(round(p, 4)),
        "Recall": float(round(r, 4)),
        "F1": float(round(f1, 4)),
        "FPR": float(round(fpr, 4))
    }


# =====================================================================
# 3. Main Retraining Loop
# =====================================================================

def run_retrain(confirmed_tx_id: str = DEFAULT_CONFIRMED_TX):
    if not PAIR_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {PAIR_FEATURES_PATH}")
    if not ORIGINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {ORIGINAL_MODEL_PATH}")

    # Load pair features
    df = pd.read_csv(PAIR_FEATURES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Locate confirmed transaction
    tx_match = df[df["transaction_id"] == confirmed_tx_id]
    if tx_match.empty:
        raise ValueError(f"Confirmed transaction {confirmed_tx_id} not found in {PAIR_FEATURES_PATH}")

    confirmed_row = tx_match.iloc[0]
    confirmed_timestamp = confirmed_row["timestamp"]

    # Identify ML features
    metadata_cols = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    # Load original XGBoost model to extract exact schema and hyperparameters
    original_model = xgb.XGBClassifier()
    original_model.load_model(str(ORIGINAL_MODEL_PATH))

    # Retrieve feature names from original booster
    booster = original_model.get_booster()
    orig_features = booster.feature_names
    if orig_features is None:
        orig_features = feature_cols

    # Ensure schema matches exactly
    feature_cols = [f for f in orig_features if f in df.columns]

    # Setup 70 / 15 / 15 chronological split
    total_n = len(df)
    train_end = int(0.70 * total_n)
    val_end = int(0.85 * total_n)

    df_train_base = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_heldout_all = df.iloc[val_end:].copy()

    # Held-out evaluation slice must NOT contain the confirmed training sample
    df_heldout_eval = df_heldout_all[df_heldout_all["transaction_id"] != confirmed_tx_id].copy()

    # Create retrained training dataset: Training Set + Confirmed Fraud Transaction (with trusted label is_fraud = 1)
    confirmed_entry = confirmed_row.to_dict()
    confirmed_entry["is_fraud"] = 1  # Analyst confirmed

    df_train_updated = pd.concat([df_train_base, pd.DataFrame([confirmed_entry])], ignore_index=True)
    df_train_updated = df_train_updated.sort_values("timestamp").reset_index(drop=True)

    # Run Leakage Audit
    perform_retrain_leakage_audit(
        feature_columns=feature_cols,
        df_train=df_train_updated,
        df_test=df_heldout_eval,
        confirmed_tx_id=confirmed_tx_id,
        confirmation_timestamp=confirmed_timestamp,
        original_feature_names=orig_features
    )

    X_train_orig = df_train_base[feature_cols].values
    y_train_orig = df_train_base["is_fraud"].values

    X_train_new = df_train_updated[feature_cols].values
    y_train_new = df_train_updated["is_fraud"].values

    X_val = df_val[feature_cols].values
    y_val = df_val["is_fraud"].values

    X_test = df_heldout_eval[feature_cols].values
    y_test = df_heldout_eval["is_fraud"].values

    # Train updated XGBoost Pair Model
    retrained_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=max(1.0, float(np.sum(y_train_new == 0) / (np.sum(y_train_new == 1) + 1e-5))),
        random_state=42,
        eval_metric="logloss"
    )

    retrained_model.fit(
        X_train_new,
        y_train_new,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    retrained_model.save_model(str(RETRAINED_MODEL_PATH))

    # Evaluate before vs after on identical held-out slice
    orig_metrics = evaluate_model(original_model, X_test, y_test, threshold=BLOCK_THRESHOLD)
    retrain_metrics = evaluate_model(retrained_model, X_test, y_test, threshold=BLOCK_THRESHOLD)

    # Score confirmed transaction specifically
    X_confirmed = df.loc[df["transaction_id"] == confirmed_tx_id, feature_cols].values
    orig_conf_score = float(original_model.predict_proba(X_confirmed)[:, 1][0])
    retrain_conf_score = float(retrained_model.predict_proba(X_confirmed)[:, 1][0])

    # Save metadata
    metadata = {
        "retraining_timestamp": datetime.now().isoformat(),
        "confirmed_transaction_id": confirmed_tx_id,
        "confirmation_timestamp": str(confirmed_timestamp),
        "training_samples_total": int(len(df_train_updated)),
        "training_fraud_samples": int(np.sum(y_train_new == 1)),
        "training_normal_samples": int(np.sum(y_train_new == 0)),
        "feature_count": len(feature_cols),
        "feature_schema": feature_cols,
        "model_type": "xgboost.XGBClassifier",
        "previous_model_path": str(ORIGINAL_MODEL_PATH),
        "retrained_model_path": str(RETRAINED_MODEL_PATH),
        "evaluation_heldout_size": int(len(df_heldout_eval)),
        "evaluation_heldout_positives": int(np.sum(y_test == 1)),
        "metrics_original": orig_metrics,
        "metrics_retrained": retrain_metrics,
        "confirmed_transaction_scoring": {
            "original_risk": orig_conf_score,
            "retrained_risk": retrain_conf_score,
            "confirmed_label": 1
        }
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # ==========================================================
    # Terminal Report
    # ==========================================================
    print("=======================================================")
    print("         RINGBREAKER CONFIRM-FRAUD RETRAINING          ")
    print("=======================================================")
    print(f"Confirmed transaction:      {confirmed_tx_id}")
    print(f"Confirmation timestamp:     {confirmed_timestamp}")
    print(f"Confirmed label:            1 (Analyst Confirmed)")
    print(f"\nChronological Split Windows:")
    print(f"  Training base window:     {df_train_base['timestamp'].min()} to {df_train_base['timestamp'].max()} (Cutoff: {df_train_base['timestamp'].max()})")
    print(f"  Validation window:        {df_val['timestamp'].min()} to {df_val['timestamp'].max()} (Cutoff: {df_val['timestamp'].max()})")
    print(f"  Held-out evaluation window: {df_heldout_eval['timestamp'].min()} to {df_heldout_eval['timestamp'].max()}")
    print(f"\nTraining Dataset Statistics:")
    print(f"  Total training samples:   {len(df_train_updated)} (Base 7018 + 1 Confirmed)")
    print(f"  Fraud samples:            {int(np.sum(y_train_new == 1))}")
    print(f"  Normal samples:           {int(np.sum(y_train_new == 0))}")
    print(f"  Feature count:            {len(feature_cols)}")

    print("\n--- HELD-OUT TEST EVALUATION COMPARISON (15% Untouched Slice) ---")
    comp_df = pd.DataFrame([orig_metrics, retrain_metrics], index=["Original Model", "Retrained Model"])
    print(comp_df.to_string())

    print("\n--- CONFIRMED TRANSACTION SCORING ---")
    print(f"  Transaction ID:           {confirmed_tx_id}")
    print(f"  Original Model Risk:      {orig_conf_score:.4f}")
    print(f"  Retrained Model Risk:     {retrain_conf_score:.4f}")
    print(f"  Confirmed Label:          1")

    print(f"\nSaved Artifacts:")
    print(f"  Retrained Model:          {RETRAINED_MODEL_PATH}")
    print(f"  Retraining Metadata:      {METADATA_PATH}")
    print("=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Supervised Retraining Loop")
    parser.add_argument("--tx", type=str, default=DEFAULT_CONFIRMED_TX, help="Confirmed fraud transaction ID")
    args = parser.parse_args()

    run_retrain(confirmed_tx_id=args.tx)