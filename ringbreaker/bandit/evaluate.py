"""
RingBreaker - P2 Feature F23: Bandit Evaluation (Control vs. Bandit)
Module: bandit/evaluate.py

Evaluates the learned Bandit threshold policy against the Fixed Production Control
strictly on the untouched 15% chronological held-out slice.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix
)

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bandit.thresholds import POLICIES, action_from_risk, calculate_reward

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
RISK_SCORES_PATH = DATA_DIR / "risk_scores_retrained.csv"
ORIGINAL_RISK_PATH = DATA_DIR / "risk_scores_original.csv"

POLICY_PATH = MODELS_DIR / "bandit_policy.json"
EVALUATION_OUTPUT_PATH = DATA_DIR / "bandit_evaluation.json"
HELDOUT_SCORES_PATH = DATA_DIR / "bandit_heldout_scores.csv"


def evaluate_policy_on_slice(
    df_slice: pd.DataFrame,
    score_col: str,
    policy_name: str
) -> Dict[str, Any]:
    y_true = df_slice["is_fraud"].values
    n = len(df_slice)
    fraud_total = int(np.sum(y_true == 1))

    actions = [action_from_risk(float(r), policy_name) for r in df_slice[score_col]]
    rewards = [calculate_reward(act, y) for act, y in zip(actions, y_true)]

    # Hard block metrics (Action == BLOCK)
    preds_block = np.array([1 if a == "BLOCK" else 0 for a in actions])
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds_block, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_block, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-8)

    # Any flag (REVIEW or BLOCK)
    preds_flag = np.array([1 if a in ["REVIEW", "BLOCK"] else 0 for a in actions])
    flag_recall = float(np.sum((y_true == 1) & (preds_flag == 1)) / (fraud_total + 1e-8))

    allow_cnt = actions.count("ALLOW")
    review_cnt = actions.count("REVIEW")
    block_cnt = actions.count("BLOCK")

    total_reward = float(np.sum(rewards))
    avg_reward = float(total_reward / n)

    return {
        "policy": policy_name,
        "transactions": n,
        "fraud_total": fraud_total,
        "recall_hard_block": float(round(r, 4)),
        "recall_any_flag": float(round(flag_recall, 4)),
        "precision": float(round(p, 4)),
        "f1": float(round(f1, 4)),
        "fpr": float(round(fpr, 4)),
        "total_reward": round(total_reward, 2),
        "avg_reward_per_tx": round(avg_reward, 4),
        "allow_count": allow_cnt,
        "review_count": review_cnt,
        "block_count": block_cnt,
        "false_blocks": int(fp),
        "fraud_missed": int(fn),
        "actions": actions,
        "rewards": rewards
    }


def run_bandit_evaluation():
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Missing {POLICY_PATH}. Run bandit/train.py first.")

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy_data = json.load(f)

    selected_policy = policy_data["selected_policy"]

    risk_path = RISK_SCORES_PATH if RISK_SCORES_PATH.exists() else ORIGINAL_RISK_PATH
    if not risk_path.exists():
        raise FileNotFoundError(f"Missing risk scores at {risk_path}")

    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])
    df_payments = df_payments.sort_values("timestamp").reset_index(drop=True)

    df_risk = pd.read_csv(risk_path)
    score_col = "overall_risk" if "overall_risk" in df_risk.columns else "risk_score"

    merge_cols = ["transaction_id", score_col]
    df_merged = df_payments.merge(df_risk[merge_cols], on="transaction_id", how="inner")
    df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"])
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

    total_n = len(df_merged)
    val_end = int(0.85 * total_n)
    df_heldout = df_merged.iloc[val_end:].copy().reset_index(drop=True)

    # Evaluate CONTROL (Fixed Current Thresholds: 0.30 / 0.70)
    control_eval = evaluate_policy_on_slice(df_heldout, score_col, "current")

    # Evaluate BANDIT (Learned Threshold Policy)
    bandit_eval = evaluate_policy_on_slice(df_heldout, score_col, selected_policy)

    # Save detailed held-out scoring table
    df_heldout_out = df_heldout[["transaction_id", "timestamp", "sender", "receiver", "amount", score_col, "is_fraud", "ring_id"]].copy()
    df_heldout_out["control_action"] = control_eval["actions"]
    df_heldout_out["control_reward"] = control_eval["rewards"]
    df_heldout_out["bandit_policy"] = selected_policy
    df_heldout_out["bandit_action"] = bandit_eval["actions"]
    df_heldout_out["bandit_reward"] = bandit_eval["rewards"]
    df_heldout_out.to_csv(HELDOUT_SCORES_PATH, index=False)

    # Strip raw action arrays from JSON payload
    c_summary = {k: v for k, v in control_eval.items() if k not in ["actions", "rewards"]}
    b_summary = {k: v for k, v in bandit_eval.items() if k not in ["actions", "rewards"]}

    eval_payload = {
        "evaluation_slice": "Held-Out (15% Chronological)",
        "transactions_evaluated": len(df_heldout),
        "fraud_transactions": int(df_heldout["is_fraud"].sum()),
        "selected_policy": selected_policy,
        "learned_thresholds": POLICIES[selected_policy],
        "control_thresholds": POLICIES["current"],
        "control_metrics": c_summary,
        "bandit_metrics": b_summary
    }

    with open(EVALUATION_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    # Terminal Report
    print("=======================================================")
    print("          RINGBREAKER BANDIT EVALUATION                ")
    print("=======================================================")
    print(f"Held-Out Transactions:    {len(df_heldout)}")
    print(f"Held-Out Fraud Count:     {int(df_heldout['is_fraud'].sum())}")
    print(f"\nControl Policy:           current (REVIEW 0.30 / BLOCK 0.70)")
    print(f"Selected Bandit Policy:   {selected_policy} (REVIEW {POLICIES[selected_policy]['review_threshold']:.2f} / BLOCK {POLICIES[selected_policy]['block_threshold']:.2f})")

    print("\n-------------------------------------------------------")
    print("CONTROL VS BANDIT (Held-Out Slice)")
    print("-------------------------------------------------------")
    print(f"{'Metric':<25} {'Control':<15} {'Bandit':<15}")
    print("-" * 55)
    print(f"{'Recall (Hard Block)':<25} {c_summary['recall_hard_block']:<15.4f} {b_summary['recall_hard_block']:<15.4f}")
    print(f"{'Recall (Any Flag)':<25} {c_summary['recall_any_flag']:<15.4f} {b_summary['recall_any_flag']:<15.4f}")
    print(f"{'Precision':<25} {c_summary['precision']:<15.4f} {b_summary['precision']:<15.4f}")
    print(f"{'F1 Score':<25} {c_summary['f1']:<15.4f} {b_summary['f1']:<15.4f}")
    print(f"{'FPR':<25} {c_summary['fpr']:<15.4f} {b_summary['fpr']:<15.4f}")
    print(f"{'Total Reward':<25} {c_summary['total_reward']:<15.1f} {b_summary['total_reward']:<15.1f}")
    print(f"{'Avg Reward / Tx':<25} {c_summary['avg_reward_per_tx']:<15.4f} {b_summary['avg_reward_per_tx']:<15.4f}")
    print(f"{'ALLOW Count':<25} {c_summary['allow_count']:<15} {b_summary['allow_count']:<15}")
    print(f"{'REVIEW Count':<25} {c_summary['review_count']:<15} {b_summary['review_count']:<15}")
    print(f"{'BLOCK Count':<25} {c_summary['block_count']:<15} {b_summary['block_count']:<15}")
    print(f"{'False Blocks':<25} {c_summary['false_blocks']:<15} {b_summary['false_blocks']:<15}")
    print(f"{'Fraud Missed (Non-Block)':<25} {c_summary['fraud_missed']:<15} {b_summary['fraud_missed']:<15}")

    print("\n-------------------------------------------------------")
    print("INTERPRETATION")
    print("-------------------------------------------------------")
    print("The bandit experiment treats threshold policies as competing actions.")
    print("During sequential training, the system observes risk/context, selects a")
    print("threshold policy, receives simulated analyst feedback, and updates the")
    print("policy reward estimate.")
    print("\nThe current fixed thresholds remain the production control.")
    print("The bandit is an offline P2 experiment and does not modify production")
    print("RingBreaker behaviour.")

    print(f"\nArtifacts Saved:")
    print(f"  Evaluation JSON:        {EVALUATION_OUTPUT_PATH}")
    print(f"  Held-Out Scores CSV:    {HELDOUT_SCORES_PATH}")
    print("=======================================================")


if __name__ == "__main__":
    run_bandit_evaluation()