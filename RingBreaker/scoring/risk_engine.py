"""
RingBreaker - Integrated Risk Engine (3-Signal Fusion with Model Selection)
Module: scoring/risk_engine.py

Combines:
1. Transaction-level Pair Risk (0.60) - supports 'original' vs 'retrained' XGBoost models
2. Account-level Behavioural Anomaly (Extended Isolation Forest) (0.25)
3. Account-level Lockstep Coordination (DBSCAN) (0.15)

Evaluates on the exact 15% chronological held-out slice.
"""

import os
import sys
from pathlib import Path

# Ensure project root is available on sys.path before local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    confusion_matrix
)

from scoring.action import determine_action, ALLOW_THRESHOLD, BLOCK_THRESHOLD

# Configurable Weights (Unchanged)
PAIR_WEIGHT = 0.60
ANOMALY_WEIGHT = 0.25
LOCKSTEP_WEIGHT = 0.15


class RiskEngine:
    def __init__(
        self,
        pair_weight: float = PAIR_WEIGHT,
        anomaly_weight: float = ANOMALY_WEIGHT,
        lockstep_weight: float = LOCKSTEP_WEIGHT
    ):
        self.pair_weight = pair_weight
        self.anomaly_weight = anomaly_weight
        self.lockstep_weight = lockstep_weight

    def score(
        self,
        payment: Dict[str, Any],
        pair_risk: float,
        behavioural_anomaly: float,
        coordination_score: float = 0.0,
        graph_risk: Optional[float] = None
    ) -> Dict[str, Any]:
        overall = (
            self.pair_weight * pair_risk +
            self.anomaly_weight * behavioural_anomaly +
            self.lockstep_weight * coordination_score
        )
        overall = float(np.clip(overall, 0.0, 1.0))
        action = determine_action(overall)

        return {
            "transaction_id": payment.get("transaction_id", "UNKNOWN"),
            "risk_score": round(overall, 4),
            "pair_risk": round(float(pair_risk), 4),
            "behavioural_anomaly": round(float(behavioural_anomaly), 4),
            "coordination_score": round(float(coordination_score), 4),
            "action": action
        }


def get_pair_predictions(model_path: Path, df_pair_features: pd.DataFrame) -> pd.Series:
    """Computes pair risk scores directly from a specified XGBoost model file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names is None:
        metadata_cols = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]
        feature_names = [c for c in df_pair_features.columns if c not in metadata_cols]

    X = df_pair_features[feature_names].values
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df_pair_features.index, name="pair_risk").round(4)


def run_pipeline_integration(model_choice: str = "original"):
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"

    scored_payments_path = data_dir / "scored_payments.csv"
    pair_features_path = data_dir / "pair_features.csv"
    anomaly_scores_path = data_dir / "anomaly_scores.csv"
    lockstep_scores_path = data_dir / "lockstep_scores.csv"
    payments_raw_path = data_dir / "payments.csv"

    # Select model file and output destination
    if model_choice == "retrained":
        model_path = models_dir / "pair_model_retrained.json"
        output_path = data_dir / "risk_scores_retrained.csv"
    else:
        model_path = models_dir / "pair_model.json"
        output_path = data_dir / "risk_scores_original.csv"

    if not anomaly_scores_path.exists():
        raise FileNotFoundError(f"Missing {anomaly_scores_path}")
    if not lockstep_scores_path.exists():
        raise FileNotFoundError(f"Missing {lockstep_scores_path}. Run features/lockstep.py first.")

    df_anom = pd.read_csv(anomaly_scores_path)
    df_lock = pd.read_csv(lockstep_scores_path)
    df_raw = pd.read_csv(payments_raw_path) if payments_raw_path.exists() else None

    # Determine Pair Risk Scores
    if model_choice == "retrained":
        if not pair_features_path.exists():
            raise FileNotFoundError(f"Missing {pair_features_path} for scoring retrained model.")
        df_pair_feats = pd.read_csv(pair_features_path)
        df_pair_feats["pair_risk"] = get_pair_predictions(model_path, df_pair_feats)
        df_pair_sub = df_pair_feats[["transaction_id", "pair_risk"]].copy()
        if "timestamp" in df_pair_feats.columns:
            df_pair_sub["timestamp"] = df_pair_feats["timestamp"]
        if "sender" in df_pair_feats.columns:
            df_pair_sub["sender"] = df_pair_feats["sender"]
        if "receiver" in df_pair_feats.columns:
            df_pair_sub["receiver"] = df_pair_feats["receiver"]
    else:
        # Original model: use scored_payments.csv if present, else predict with pair_model.json
        if scored_payments_path.exists():
            df_pair = pd.read_csv(scored_payments_path)
            pair_col = next((c for c in ["pair_risk_score", "risk_score", "score", "pair_risk", "probability"] if c in df_pair.columns), None)
            df_pair_sub = df_pair[["transaction_id", pair_col]].copy()
            df_pair_sub.rename(columns={pair_col: "pair_risk"}, inplace=True)
            for opt in ["timestamp", "sender", "receiver", "amount"]:
                if opt in df_pair.columns:
                    df_pair_sub[opt] = df_pair[opt]
        else:
            df_pair_feats = pd.read_csv(pair_features_path)
            df_pair_feats["pair_risk"] = get_pair_predictions(model_path, df_pair_feats)
            df_pair_sub = df_pair_feats[["transaction_id", "pair_risk", "timestamp", "sender", "receiver"]].copy()

    # Merge Anomaly Scores via transaction_id
    anom_cols_to_use = ["transaction_id", "sender_anomaly_score", "receiver_anomaly_score", "max_anomaly_score"]
    for col in ["timestamp", "sender", "receiver", "amount"]:
        if col not in df_pair_sub.columns and col in df_anom.columns:
            anom_cols_to_use.append(col)

    df_merged = df_pair_sub.merge(df_anom[anom_cols_to_use], on="transaction_id", how="inner")
    df_merged.rename(columns={"max_anomaly_score": "behavioural_anomaly"}, inplace=True)

    # Join Lockstep Coordination via Sender & Receiver
    lockstep_map = dict(zip(df_lock["user_id"], df_lock["coordination_score"]))
    df_merged["sender_coordination_score"] = df_merged["sender"].map(lockstep_map).fillna(0.0000).round(4)
    df_merged["receiver_coordination_score"] = df_merged["receiver"].map(lockstep_map).fillna(0.0000).round(4)
    df_merged["coordination_score"] = df_merged[["sender_coordination_score", "receiver_coordination_score"]].max(axis=1)

    # Compute Intermediate Model Signals
    df_merged["risk_m1_pair"] = df_merged["pair_risk"].clip(0.0, 1.0).round(4)
    df_merged["risk_m2_pair_eif"] = (
        0.70 * df_merged["pair_risk"] +
        0.30 * df_merged["behavioural_anomaly"]
    ).clip(0.0, 1.0).round(4)

    # Compute Integrated 3-Signal Overall Risk
    df_merged["overall_risk"] = (
        PAIR_WEIGHT * df_merged["pair_risk"] +
        ANOMALY_WEIGHT * df_merged["behavioural_anomaly"] +
        LOCKSTEP_WEIGHT * df_merged["coordination_score"]
    ).clip(0.0, 1.0).round(4)

    # Assign Operational Action
    df_merged["action"] = df_merged["overall_risk"].apply(determine_action)

    # Save Output
    preferred_cols = [
        "transaction_id", "timestamp", "sender", "receiver", "amount",
        "pair_risk", "sender_anomaly_score", "receiver_anomaly_score", "behavioural_anomaly",
        "sender_coordination_score", "receiver_coordination_score", "coordination_score",
        "overall_risk", "action"
    ]
    final_cols = [c for c in preferred_cols if c in df_merged.columns]
    df_merged[final_cols].to_csv(output_path, index=False)

    # Sync primary risk_scores.csv when running original
    if model_choice == "original":
        df_merged[final_cols].to_csv(data_dir / "risk_scores.csv", index=False)

    # Terminal Report Header
    print("=================================================================")
    print(f"       RINGBREAKER RISK ENGINE (MODEL: {model_choice.upper()})")
    print("=================================================================")
    print(f"Model Path:               {model_path}")
    print(f"Transactions Scored:      {len(df_merged)}")
    print(f"Integration Formula:      0.60 * Pair + 0.25 * Anomaly + 0.15 * Lockstep")
    print(f"Action Thresholds:        ALLOW < 0.30 | 0.30 <= REVIEW < 0.70 | BLOCK >= 0.70")
    print(f"Output Saved To:          {output_path}")

    # Specific TX_0008825 Check
    tx_check = df_merged[df_merged["transaction_id"] == "TX_0008825"]
    if not tx_check.empty:
        r = tx_check.iloc[0]
        print(f"\n--- TARGET TRANSACTION AUDIT (TX_0008825) ---")
        print(f"  Transaction ID:         TX_0008825")
        print(f"  Pair Risk:              {r['pair_risk']:.4f}")
        print(f"  Behavioural Anomaly:    {r['behavioural_anomaly']:.4f}")
        print(f"  Coordination Score:     {r['coordination_score']:.4f}")
        print(f"  Overall Integrated Risk:{r['overall_risk']:.4f}")
        print(f"  Action Decision:        {r['action']}")

    # Evaluation on 15% Held-Out Slice
    if df_raw is not None and "is_fraud" in df_raw.columns:
        eval_cols = ["transaction_id", "is_fraud"]
        if "ring_id" in df_raw.columns:
            eval_cols.append("ring_id")
        if "timestamp" not in df_merged.columns and "timestamp" in df_raw.columns:
            eval_cols.append("timestamp")

        df_eval = df_merged.merge(df_raw[eval_cols], on="transaction_id", how="left")
        df_eval["timestamp"] = pd.to_datetime(df_eval["timestamp"])
        df_eval = df_eval.sort_values("timestamp").reset_index(drop=True)

        total_n = len(df_eval)
        val_end = int(0.85 * total_n)
        df_test = df_eval.iloc[val_end:].copy()
        y_test = df_test["is_fraud"].values

        def compute_metrics(y_true, scores, threshold):
            roc = roc_auc_score(y_true, scores)
            p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(r_curve, p_curve)
            preds = (scores >= threshold).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn + 1e-8)
            return {"ROC-AUC": roc, "PR-AUC": pr_auc, "Precision": p, "Recall": r, "F1": f1, "FPR": fpr}

        pair_m = compute_metrics(y_test, df_test["pair_risk"].values, BLOCK_THRESHOLD)
        eng_m = compute_metrics(y_test, df_test["overall_risk"].values, BLOCK_THRESHOLD)

        print("\n--- HELD-OUT PERFORMANCE (15% Chronological Slice) ---")
        comp_df = pd.DataFrame([pair_m, eng_m], index=[f"Pair Risk ({model_choice})", f"Integrated Engine ({model_choice})"])
        print(comp_df.round(4).to_string())

        # Action Distribution
        print("\n--- ACTION DISTRIBUTION (Full Dataset) ---")
        action_counts = df_merged["action"].value_counts()
        for act in ["ALLOW", "REVIEW", "BLOCK"]:
            cnt = action_counts.get(act, 0)
            pct = (cnt / len(df_merged)) * 100.0
            print(f"  {act:<8}: {cnt:>6} transactions ({pct:>5.2f}%)")

        # Ring-level breakdown
        if "ring_id" in df_eval.columns:
            print("\n--- RING-LEVEL DETECTION BREAKDOWN ---")
            rings = [r for r in df_eval["ring_id"].dropna().unique() if r not in ["NORMAL", "NONE", ""]]
            for r in sorted(rings):
                rdf = df_eval[df_eval["ring_id"] == r]
                n_tx = len(rdf)
                flagged = (rdf["overall_risk"] >= ALLOW_THRESHOLD).sum()
                blocked = (rdf["overall_risk"] >= BLOCK_THRESHOLD).sum()
                avg_risk = rdf["overall_risk"].mean()
                print(f"  Ring: {r:<15} | Tx: {n_tx:>2} | Flagged (>=0.30): {flagged:>2} | Blocked (>=0.70): {blocked:>2} | Avg Risk: {avg_risk:.4f}")

    print("=================================================================\n")
    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Integrated Risk Engine")
    parser.add_argument(
        "--model",
        type=str,
        choices=["original", "retrained"],
        default="original",
        help="Pair Risk Model state to score with: 'original' (models/pair_model.json) or 'retrained' (models/pair_model_retrained.json)"
    )
    args = parser.parse_args()

    run_pipeline_integration(model_choice=args.model)