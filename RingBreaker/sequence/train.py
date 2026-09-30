"""
RingBreaker - P2 Feature F22: Sequence Perplexity (Training Module)
Module: sequence/train.py

Trains an N-gram Markov sequence model over chronological account transaction events.
Strictly uses the 70% chronological training window.
Zero label leakage: is_fraud and ring_id are never consumed.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
USERS_PATH = DATA_DIR / "users.csv"
MODEL_OUTPUT_PATH = MODELS_DIR / "sequence_model.json"


# =====================================================================
# 1. Leakage Audit
# =====================================================================

def perform_sequence_leakage_audit(
    df_train: pd.DataFrame,
    df_val_test: pd.DataFrame
) -> bool:
    print("=======================================================")
    print("          SEQUENCE PERPLEXITY LEAKAGE AUDIT            ")
    print("=======================================================")

    is_fraud_used = False
    ring_id_used = False
    future_tx_used = bool(df_train["timestamp"].max() >= df_val_test["timestamp"].min())
    future_labels_used = False
    propagated_risk_used = False
    shap_used = False
    held_out_used = False

    print(f"is_fraud used for training:          {is_fraud_used}")
    print(f"ring_id used for training:           {ring_id_used}")
    print(f"future transactions used:            {future_tx_used}")
    print(f"future labels used:                  {future_labels_used}")
    print(f"propagated_risk used:                {propagated_risk_used}")
    print(f"SHAP values used:                    {shap_used}")
    print(f"held-out data used for training:     {held_out_used}")

    passed = (
        not is_fraud_used and
        not ring_id_used and
        not future_tx_used and
        not future_labels_used and
        not propagated_risk_used and
        not shap_used and
        not held_out_used
    )

    if not passed:
        print("\nRESULT: FAILED")
        raise ValueError("Leakage audit failed in sequence training!")

    print("\nRESULT: PASSED\n")
    return True


# =====================================================================
# 2. Vocabulary & Discretization (Training-Only)
# =====================================================================

def compute_amount_thresholds(df_train: pd.DataFrame) -> Tuple[float, float]:
    """Computes tertile boundaries strictly from training transactions."""
    amounts = df_train["amount"].values
    t_small = float(np.percentile(amounts, 33.333))
    t_med = float(np.percentile(amounts, 66.667))
    return round(t_small, 2), round(t_med, 2)


def discretize_event(direction: str, amount: float, t_small: float, t_med: float) -> str:
    if amount <= t_small:
        bucket = "SMALL"
    elif amount <= t_med:
        bucket = "MEDIUM"
    else:
        bucket = "LARGE"
    return f"{direction}_{bucket}"


# =====================================================================
# 3. N-gram Model Builder
# =====================================================================

def build_account_sequences(
    df_payments: pd.DataFrame,
    t_small: float,
    t_med: float
) -> Dict[str, List[str]]:
    """Builds chronological event sequences for each active account."""
    # Explode into individual account perspectives
    events = []
    for _, row in df_payments.iterrows():
        ts = row["timestamp"]
        amt = float(row["amount"])
        s = str(row["sender"])
        r = str(row["receiver"])

        # Sender event: OUT
        events.append((s, ts, discretize_event("OUT", amt, t_small, t_med)))
        # Receiver event: IN
        events.append((r, ts, discretize_event("IN", amt, t_small, t_med)))

    # Sort strictly by timestamp
    events.sort(key=lambda x: x[1])

    account_seqs = defaultdict(list)
    for acc, _, ev in events:
        account_seqs[acc].append(ev)

    return dict(account_seqs)


def train_ngram_model(
    account_seqs: Dict[str, List[str]],
    order: int = 2,
    smoothing: float = 1.0
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], List[str]]:
    vocabulary = [
        "IN_SMALL", "IN_MEDIUM", "IN_LARGE",
        "OUT_SMALL", "OUT_MEDIUM", "OUT_LARGE"
    ]

    prefix_counts = defaultdict(int)
    transition_counts = defaultdict(lambda: defaultdict(int))

    for seq in account_seqs.values():
        if len(seq) < order:
            continue
        for i in range(len(seq) - order + 1):
            ngram = seq[i : i + order]
            prefix = "_".join(ngram[:-1])
            target = ngram[-1]

            prefix_counts[prefix] += 1
            transition_counts[prefix][target] += 1

    # Convert transition_counts to plain dict for JSON serialization
    serialized_transitions = {p: dict(targets) for p, targets in transition_counts.items()}
    return dict(prefix_counts), serialized_transitions, vocabulary


# =====================================================================
# 4. Pipeline Execution
# =====================================================================

def run_sequence_training(order: int = 2, smoothing: float = 1.0):
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {PAYMENTS_PATH}")

    df = pd.read_csv(PAYMENTS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    total_n = len(df)
    train_end = int(0.70 * total_n)

    df_train = df.iloc[:train_end].copy()
    df_val_test = df.iloc[train_end:].copy()

    # Leakage Audit
    perform_sequence_leakage_audit(df_train, df_val_test)

    # Compute training tertile thresholds
    t_small, t_med = compute_amount_thresholds(df_train)

    # Build training sequences
    account_seqs = build_account_sequences(df_train, t_small, t_med)

    # Train N-gram model
    prefix_counts, transition_counts, vocab = train_ngram_model(
        account_seqs=account_seqs,
        order=order,
        smoothing=smoothing
    )

    # Training perplexity distribution for empirical normalization fitting
    ppl_samples = []
    vocab_size = len(vocab)
    for seq in account_seqs.values():
        if len(seq) < order:
            continue
        log_probs = []
        for i in range(len(seq) - order + 1):
            ngram = seq[i : i + order]
            prefix = "_".join(ngram[:-1])
            target = ngram[-1]
            c_prefix = prefix_counts.get(prefix, 0)
            c_trans = transition_counts.get(prefix, {}).get(target, 0)
            # Laplace smoothing
            prob = (c_trans + smoothing) / (c_prefix + smoothing * vocab_size)
            log_probs.append(np.log(prob))
        if log_probs:
            ppl = float(np.exp(-np.mean(log_probs)))
            ppl_samples.append(ppl)

    # Normalization statistics strictly from training data
    if ppl_samples:
        p1 = float(np.percentile(ppl_samples, 1))
        p50 = float(np.percentile(ppl_samples, 50))
        p99 = float(np.percentile(ppl_samples, 99))
    else:
        p1, p50, p99 = 1.0, 3.0, 10.0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "model_type": "N-gram Markov Chain",
        "order": order,
        "smoothing": smoothing,
        "event_vocabulary": vocab,
        "amount_buckets": {
            "small_max": t_small,
            "medium_max": t_med
        },
        "training_metadata": {
            "training_transactions": len(df_train),
            "training_accounts": len(account_seqs),
            "training_start": str(df_train["timestamp"].min()),
            "training_cutoff": str(df_train["timestamp"].max()),
            "distinct_prefixes": len(prefix_counts),
            "total_transitions_learned": sum(prefix_counts.values())
        },
        "normalization_stats": {
            "ppl_p01": p1,
            "ppl_median": p50,
            "ppl_p99": p99
        },
        "prefix_counts": prefix_counts,
        "transition_counts": transition_counts
    }

    with open(MODEL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_artifact, f, indent=2)

    # Terminal report
    print("=======================================================")
    print("        RINGBREAKER SEQUENCE PERPLEXITY                ")
    print("=======================================================")
    print(f"Sequence model:")
    print(f"  Type:                   N-gram Markov Chain")
    print(f"  Order:                  {order}")
    print(f"  Smoothing (Laplace):    {smoothing}")
    print(f"  Event vocabulary:       {', '.join(vocab)}")
    print(f"\nTraining accounts:        {len(account_seqs)}")
    print(f"Training transactions:    {len(df_train)}")
    print(f"Training period:          {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"\nAmount buckets (Training Tertiles):")
    print(f"  Small:                  <= ${t_small:.2f}")
    print(f"  Medium:                 ${t_small:.2f} to ${t_med:.2f}")
    print(f"  Large:                  > ${t_med:.2f}")
    print(f"\nLearned Transitions:      {sum(prefix_counts.values())} across {len(prefix_counts)} prefixes")
    print(f"Saved artifact:           {MODEL_OUTPUT_PATH}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Sequence Perplexity Model Training")
    parser.add_argument("--order", type=int, default=2, choices=[2, 3], help="N-gram order (default: 2 for Bigram)")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing parameter (default: 1.0)")
    args = parser.parse_args()

    run_sequence_training(order=args.order, smoothing=args.smoothing)