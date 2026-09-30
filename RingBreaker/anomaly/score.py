"""
RingBreaker - Behavioural Anomaly Detection Inference & Evaluation
Module: anomaly/score.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support

from anomaly.eif_model import (
    FEATURE_COLUMNS,
    ExtendedIsolationForest,
    EIFNode,
    EIFTree,
    build_point_in_time_behavioral_features
)


def run_scoring_and_evaluation():
    model_path = os.path.join("models", "eif_model.pkl")
    schema_path = os.path.join("models", "eif_feature_schema.json")
    payments_path = os.path.join("data", "payments.csv")

    if not os.path.exists(model_path) or not os.path.exists(schema_path):
        raise FileNotFoundError("Missing EIF model or schema. Run anomaly/train_eif.py first.")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_cols = schema["features"]

    df_payments = pd.read_csv(payments_path)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])
    df_payments = df_payments.sort_values("timestamp").reset_index(drop=True)

    print("Generating point-in-time account behavioral features...")
    df_feat = build_point_in_time_behavioral_features(df_payments)

    X = df_feat[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    print("Scoring accounts with Extended Isolation Forest...")
    anomaly_scores = model.compute_anomaly_scores(X)

    df_scored = pd.DataFrame({
        "transaction_id": df_payments["transaction_id"],
        "timestamp": df_payments["timestamp"],
        "sender": df_payments["sender"],
        "receiver": df_payments["receiver"],
        "amount": df_payments["amount"],
        "sender_anomaly_score": np.round(anomaly_scores, 4)
    })

    # Account-level mapping
    account_score_map = dict(zip(df_scored["sender"], df_scored["sender_anomaly_score"]))
    df_scored["receiver_anomaly_score"] = df_scored["receiver"].map(account_score_map).fillna(0.15).round(4)
    df_scored["max_anomaly_score"] = df_scored[["sender_anomaly_score", "receiver_anomaly_score"]].max(axis=1)

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "anomaly_scores.csv")
    df_scored.to_csv(output_path, index=False)
    print(f"Anomaly scores saved to: {output_path}")

    if "is_fraud" in df_payments.columns:
        print("\n==========================================")
        print("          EIF PERFORMANCE REPORT          ")
        print("==========================================")
        y_true = df_payments["is_fraud"].values
        y_score = df_scored["max_anomaly_score"].values

        total_n = len(df_payments)
        train_end = int(0.70 * total_n)
        val_end = int(0.85 * total_n)

        # Unsupervised threshold from 95th percentile of training period
        train_scores = y_score[:train_end]
        chosen_threshold = float(np.percentile(train_scores, 95))

        test_true = y_true[val_end:]
        test_score = y_score[val_end:]

        print(f"Unsupervised Threshold (95th %-tile of Training Period): {chosen_threshold:.4f}\n")

        n_test_pos = int(np.sum(test_true))
        print(f"Held-out Test Period Size: {len(test_true)} transactions ({n_test_pos} fraud labels)")

        if n_test_pos > 0:
            roc_auc = roc_auc_score(test_true, test_score)
            precisions, recalls, _ = precision_recall_curve(test_true, test_score)
            pr_auc = auc(recalls, precisions)
            print(f" - ROC-AUC:   {roc_auc:.4f}")
            print(f" - PR-AUC:    {pr_auc:.4f}")

        test_preds = (test_score >= chosen_threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(test_true, test_preds, average="binary", zero_division=0)

        tn = np.sum((test_true == 0) & (test_preds == 0))
        fp = np.sum((test_true == 0) & (test_preds == 1))
        fpr = fp / (fp + tn + 1e-8)

        print(f" - Precision: {p:.4f}")
        print(f" - Recall:    {r:.4f}")
        print(f" - F1:        {f1:.4f}")
        print(f" - FPR:       {fpr:.4f}")

        if "ring_id" in df_payments.columns:
            print("\n------------------------------------------")
            print("         RING DETECTION BREAKDOWN         ")
            print("------------------------------------------")
            df_eval = df_scored.copy()
            df_eval["is_fraud"] = df_payments["is_fraud"]
            df_eval["ring_id"] = df_payments["ring_id"]

            rings = [r for r in df_eval["ring_id"].dropna().unique() if r not in ["NORMAL", "NONE", ""]]
            for ring in sorted(rings):
                ring_df = df_eval[df_eval["ring_id"] == ring]
                n_tx = len(ring_df)
                detected = (ring_df["max_anomaly_score"] >= chosen_threshold).sum()
                det_rate = (detected / n_tx) * 100 if n_tx > 0 else 0
                avg_anom = ring_df["max_anomaly_score"].mean()
                max_anom = ring_df["max_anomaly_score"].max()

                print(f"Ring: {ring:<15} | Tx: {n_tx:>3} | Det: {detected:>3} ({det_rate:>5.1f}%) | "
                      f"Avg Score: {avg_anom:.3f} | Max Score: {max_anom:.3f}")

            normal_df = df_eval[df_eval["is_fraud"] == 0]
            norm_flagged = (normal_df["max_anomaly_score"] >= chosen_threshold).sum()
            norm_flag_rate = (norm_flagged / len(normal_df)) * 100
            print("\n------------------------------------------")
            print("       NORMAL ACCOUNT BEHAVIOUR CHECK     ")
            print("------------------------------------------")
            print(f"Total Normal Transactions:  {len(normal_df)}")
            print(f"Normal False Alarms:        {norm_flagged} ({norm_flag_rate:.2f}%)")
            print(f"Mean Normal Anomaly Score:  {normal_df['max_anomaly_score'].mean():.4f}")
            print("==========================================\n")


if __name__ == "__main__":
    run_scoring_and_evaluation()