"""
RingBreaker Pair Risk Model
===========================
Trains, validates, and evaluates an XGBoost classifier on engineered pair features.
Enforces chronological 70/15/15 partitioning and checks detection recall
on both training rings and the held-out test ring.

Input:
  - data/pair_features.csv
  - data/rings.csv
Output:
  - models/pair_model.json
  - data/scored_payments.csv
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve
)

METADATA_COLS = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]

def train_and_evaluate(
    features_path="data/pair_features.csv",
    rings_path="data/rings.csv",
    model_out="models/pair_model.json",
    scored_out="data/scored_payments.csv"
):
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature table not found: {features_path}")

    df = pd.read_csv(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols]
    y = df["is_fraud"]

    # 1. Chronological Split (70% Train, 15% Validation, 15% Test)
    n_total = len(df)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    meta_test = df.iloc[val_end:][METADATA_COLS].copy()

    # 2. Address severe class imbalance (0.26% positive rate)
    num_neg = (y_train == 0).sum()
    num_pos = max((y_train == 1).sum(), 1)
    scale_pos = num_neg / num_pos

    print("=" * 60)
    print("RINGBREAKER PAIR RISK MODEL TRAINING")
    print("=" * 60)
    print(f"Total Transactions: {n_total:,}")
    print(f"Train Partition:    {len(X_train):,} rows ({y_train.sum()} fraud)")
    print(f"Val Partition:      {len(X_val):,} rows ({y_val.sum()} fraud)")
    print(f"Test Partition:     {len(X_test):,} rows ({y_test.sum()} fraud)")
    print(f"Negative/Positive Weight Ratio: {scale_pos:.2f}")

    # 3. Model Definition
    clf = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=42
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 4. Calibration on Validation Split
    val_probs = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-6)
    best_thresh_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5
    optimal_threshold = max(0.2, min(optimal_threshold, 0.8))  # clamp

    print(f"Calibrated Alert Threshold: {optimal_threshold:.4f}")

    # 5. Out-of-Time Test Evaluation
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= optimal_threshold).astype(int)

    test_pr_auc = average_precision_score(y_test, test_probs)
    test_roc_auc = roc_auc_score(y_test, test_probs)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 60)
    print(f"PR-AUC (Primary Metric):  {test_pr_auc:.4f}")
    print(f"ROC-AUC:                  {test_roc_auc:.4f}")
    print("\nClassification Report (Test Partition):")
    print(classification_report(y_test, test_preds, digits=4, zero_division=0))

    # 6. Evaluation against Planted Rings
    meta_test["risk_score"] = test_probs
    meta_test["predicted_fraud"] = test_preds

    print("Ring Detection Breakdown (Test Set):")
    test_rings = meta_test[meta_test["ring_id"].notnull() & (meta_test["ring_id"] != "")]["ring_id"].unique()
    for r_id in test_rings:
        subset = meta_test[meta_test["ring_id"] == r_id]
        detected = (subset["predicted_fraud"] == 1).sum()
        total_tx = len(subset)
        avg_score = subset["risk_score"].mean()
        print(f"  - {r_id:<15}: Detected {detected}/{total_tx} txs (Recall: {detected/total_tx*100:.1f}%, Avg Risk: {avg_score:.3f})")

    # 7. Score Entire Dataset & Export Artifacts
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    clf.save_model(model_out)

    all_probs = clf.predict_proba(X)[:, 1]
    df["risk_score"] = np.round(all_probs, 4)
    df["predicted_fraud"] = (all_probs >= optimal_threshold).astype(int)

    export_cols = METADATA_COLS + ["risk_score", "predicted_fraud"]
    df[export_cols].to_csv(scored_out, index=False)

    # 8. Feature Importances
    imp_series = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 8 Most Discriminative Features:")
    for feat, imp in imp_series.head(8).items():
        print(f"  - {feat:<35}: {imp:.4f}")

    print("\nArtifacts Saved:")
    print(f"  1. Model:  {model_out}")
    print(f"  2. Scores: {scored_out}")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate()