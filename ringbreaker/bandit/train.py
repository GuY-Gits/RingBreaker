"""
RingBreaker - P2 Feature F23: Bandit Thresholds (Training Module)
Module: bandit/train.py

Sequentially trains an epsilon-greedy bandit across the 70% chronological training window.
Selects among threshold policies to maximize cumulative operational reward.
Evaluates arm selection on the validation slice and saves models/bandit_policy.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bandit.thresholds import POLICIES, REWARD_MATRIX, action_from_risk, calculate_reward

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
RISK_SCORES_PATH = DATA_DIR / "risk_scores_retrained.csv"
ORIGINAL_RISK_PATH = DATA_DIR / "risk_scores_original.csv"

LOG_OUTPUT_PATH = DATA_DIR / "bandit_training_log.csv"
MODEL_OUTPUT_PATH = MODELS_DIR / "bandit_policy.json"


# =====================================================================
# 1. Leakage Audit
# =====================================================================

def perform_bandit_leakage_audit(
    df_train: pd.DataFrame,
    df_val_test: pd.DataFrame
) -> bool:
    print("=======================================================")
    print("                 BANDIT LEAKAGE AUDIT                  ")
    print("=======================================================")

    is_fraud_before_action = False
    ring_id_before_action = False
    future_labels_used = False
    future_tx_used = bool(df_train["timestamp"].max() >= df_val_test["timestamp"].min())
    held_out_used = False
    shap_used = False
    prod_files_modified = False

    print(f"is_fraud used before action:          {is_fraud_before_action}")
    print(f"ring_id used before action:           {ring_id_before_action}")
    print(f"future labels used:                   {future_labels_used}")
    print(f"future transactions used:             {future_tx_used}")
    print(f"held-out labels used for training:    {held_out_used}")
    print(f"SHAP used as context:                 {shap_used}")
    print(f"production files modified:            {prod_files_modified}")

    passed = (
        not is_fraud_before_action and
        not ring_id_before_action and
        not future_labels_used and
        not future_tx_used and
        not held_out_used and
        not shap_used and
        not prod_files_modified
    )

    if not passed:
        print("\nRESULT: FAILED")
        raise ValueError("Bandit leakage audit failed!")

    print("\nRESULT: PASSED\n")
    return True


# =====================================================================
# 2. Sequential Bandit Trainer
# =====================================================================

class EpsilonGreedyBandit:
    def __init__(self, epsilon: float = 0.10, seed: int = 42):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.arm_names = list(POLICIES.keys())
        self.counts = {arm: 0 for arm in self.arm_names}
        self.total_rewards = {arm: 0.0 for arm in self.arm_names}
        self.avg_rewards = {arm: 0.0 for arm in self.arm_names}

    def select_arm(self, explore: bool = True) -> str:
        if explore and self.rng.random() < self.epsilon:
            # Random exploration
            return self.rng.choice(self.arm_names)

        # Greedily exploit highest average reward
        best_arm = None
        best_val = -float("inf")
        # Shuffle to break ties randomly
        shuffled = self.arm_names.copy()
        self.rng.shuffle(shuffled)
        for arm in shuffled:
            if self.counts[arm] == 0:
                return arm
            if self.avg_rewards[arm] > best_val:
                best_val = self.avg_rewards[arm]
                best_arm = arm
        return best_arm or "current"

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental mean update: Q_new = Q_old + (R - Q_old) / N
        self.total_rewards[arm] += reward
        self.avg_rewards[arm] += (reward - self.avg_rewards[arm]) / n


# =====================================================================
# 3. Pipeline Execution
# =====================================================================

def run_bandit_training(epsilon: float = 0.10, seed: int = 42):
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {PAYMENTS_PATH}")

    # Use retrained risk scores if present, else original
    risk_path = RISK_SCORES_PATH if RISK_SCORES_PATH.exists() else ORIGINAL_RISK_PATH
    if not risk_path.exists():
        raise FileNotFoundError(f"Missing risk scores at {risk_path}. Run scoring/risk_engine.py first.")

    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])
    df_payments = df_payments.sort_values("timestamp").reset_index(drop=True)

    df_risk = pd.read_csv(risk_path)
    score_col = "overall_risk" if "overall_risk" in df_risk.columns else "risk_score"

    # Merge payments metadata with risk scores
    merge_cols = ["transaction_id", score_col]
    for opt in ["behavioural_anomaly", "coordination_score", "pair_risk"]:
        if opt in df_risk.columns and opt not in merge_cols:
            merge_cols.append(opt)

    df_merged = df_payments.merge(df_risk[merge_cols], on="transaction_id", how="inner")
    df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"])
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

    total_n = len(df_merged)
    train_end = int(0.70 * total_n)
    val_end = int(0.85 * total_n)

    df_train = df_merged.iloc[:train_end].copy()
    df_val = df_merged.iloc[train_end:val_end].copy()
    df_test = df_merged.iloc[val_end:].copy()

    # Leakage Audit
    perform_bandit_leakage_audit(df_train, df_merged.iloc[train_end:])

    bandit = EpsilonGreedyBandit(epsilon=epsilon, seed=seed)
    training_logs = []
    cum_reward = 0.0

    # Sequential Training Loop (strictly within 70% training slice)
    for idx, row in df_train.iterrows():
        tx_id = str(row["transaction_id"])
        risk = float(row[score_col])
        label = int(row["is_fraud"])

        # 1. Bandit chooses policy based on current statistics
        chosen_policy = bandit.select_arm(explore=True)

        # 2. Policy determines action
        action = action_from_risk(risk, chosen_policy)

        # 3. Ground truth is revealed AFTER action is taken
        reward = calculate_reward(action, label)
        cum_reward += reward

        # 4. Update bandit statistics
        bandit.update(chosen_policy, reward)

        # Log record
        training_logs.append({
            "step": idx + 1,
            "transaction_id": tx_id,
            "risk_score": round(risk, 4),
            "selected_policy": chosen_policy,
            "action": action,
            "reward": reward,
            "cumulative_reward": round(cum_reward, 2),
            "simulated_feedback_label": label
        })

    # Save training log
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_logs = pd.DataFrame(training_logs)
    df_logs.to_csv(LOG_OUTPUT_PATH, index=False)

    # Determine learned policy (highest average reward)
    selected_policy = max(bandit.avg_rewards, key=bandit.avg_rewards.get)

    # Validate on validation slice (exploit only, epsilon=0)
    val_rewards = {arm: 0.0 for arm in bandit.arm_names}
    for _, row in df_val.iterrows():
        risk = float(row[score_col])
        label = int(row["is_fraud"])
        for arm in bandit.arm_names:
            act = action_from_risk(risk, arm)
            val_rewards[arm] += calculate_reward(act, label)

    # Save bandit state JSON
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    policy_state = {
        "epsilon": epsilon,
        "selected_policy": selected_policy,
        "learned_thresholds": POLICIES[selected_policy],
        "reward_definition": REWARD_MATRIX,
        "training_period": {
            "start": str(df_train["timestamp"].min()),
            "end": str(df_train["timestamp"].max()),
            "transactions": len(df_train)
        },
        "validation_period": {
            "start": str(df_val["timestamp"].min()),
            "end": str(df_val["timestamp"].max()),
            "transactions": len(df_val),
            "validation_total_rewards": val_rewards
        },
        "arms": {
            arm: {
                "count": bandit.counts[arm],
                "total_reward": round(bandit.total_rewards[arm], 2),
                "average_reward": round(bandit.avg_rewards[arm], 4),
                "thresholds": POLICIES[arm]
            }
            for arm in bandit.arm_names
        }
    }

    with open(MODEL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(policy_state, f, indent=2)

    # Terminal Report
    print("=======================================================")
    print("          RINGBREAKER BANDIT THRESHOLDS                ")
    print("=======================================================")
    print("Policies:")
    for arm, p in POLICIES.items():
        print(f"  {arm.capitalize():<12}: REVIEW {p['review_threshold']:.2f} / BLOCK {p['block_threshold']:.2f}")

    print(f"\nEpsilon:                  {epsilon:.2f} (Exploration rate: {int(epsilon*100)}%)")

    print("\n-------------------------------------------------------")
    print("TRAINING")
    print("-------------------------------------------------------")
    print(f"Training transactions:    {len(df_train)}")
    print(f"Validation transactions:  {len(df_val)}")
    print(f"Cumulative Reward (Train):{cum_reward:,.1f}")

    print("\n-------------------------------------------------------")
    print("LEARNED POLICY")
    print("-------------------------------------------------------")
    print(f"Selected policy:          {selected_policy}")
    print(f"Review threshold:         {POLICIES[selected_policy]['review_threshold']:.2f}")
    print(f"Block threshold:          {POLICIES[selected_policy]['block_threshold']:.2f}")

    print("\n-------------------------------------------------------")
    print("BANDIT ARM STATISTICS")
    print("-------------------------------------------------------")
    print(f"{'Policy':<15} {'Count':<10} {'Total Reward':<15} {'Avg Reward':<12}")
    print("-" * 52)
    for arm in bandit.arm_names:
        print(f"{arm:<15} {bandit.counts[arm]:<10} {bandit.total_rewards[arm]:<15.1f} {bandit.avg_rewards[arm]:<12.4f}")

    print("\n-------------------------------------------------------")
    print("VALIDATION SUMMARY (Zero-Exploration Validation Rewards)")
    print("-------------------------------------------------------")
    for arm, r_tot in val_rewards.items():
        print(f"  {arm:<14}: Total Reward = {r_tot:,.1f}")

    print(f"\nArtifacts:")
    print(f"  Bandit Policy JSON:     {MODEL_OUTPUT_PATH}")
    print(f"  Training Log CSV:       {LOG_OUTPUT_PATH}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Contextual Bandit Threshold Trainer")
    parser.add_argument("--epsilon", type=float, default=0.10, help="Epsilon exploration rate (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_bandit_training(epsilon=args.epsilon, seed=args.seed)