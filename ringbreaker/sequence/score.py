"""
RingBreaker - P2 Feature F22: Sequence Perplexity (Scoring & Evaluation Module)
Module: sequence/score.py

Loads models/sequence_model.json and scores transaction sequences.
Generates:
- data/sequence_scores.csv (Account-level perplexity and normalized anomaly)
- data/sequence_transaction_scores.csv (Transaction-level max counterparty score)
- data/sequence_combined_scores.csv (Optional Pair + Sequence experiment)
- data/sequence_evaluation.json (Structured evaluation report)
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
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    confusion_matrix
)

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
USERS_PATH = DATA_DIR / "users.csv"
RINGS_PATH = DATA_DIR / "rings.csv"
RETRAINED_SCORES_PATH = DATA_DIR / "risk_scores_retrained.csv"
ORIGINAL_SCORES_PATH = DATA_DIR / "risk_scores_original.csv"

MODEL_PATH = MODELS_DIR / "sequence_model.json"
ACCOUNT_SCORES_PATH = DATA_DIR / "sequence_scores.csv"
TX_SCORES_PATH = DATA_DIR / "sequence_transaction_scores.csv"
COMBINED_SCORES_PATH = DATA_DIR / "sequence_combined_scores.csv"
EVALUATION_PATH = DATA_DIR / "sequence_evaluation.json"

ANOMALY_THRESHOLD = 0.80


# =====================================================================
# 1. Sequence Perplexity Calculator
# =====================================================================

def calculate_sequence_perplexity(
    seq: List[str],
    order: int,
    vocab: List[str],
    prefix_counts: Dict[str, int],
    transition_counts: Dict[str, Dict[str, int]],
    smoothing: float
) -> Tuple[float, bool]:
    """
    Computes perplexity = exp(-1/M * sum(log P(e_t | h))).
    If sequence length < order, returns default neutral perplexity and insufficient_history=True.
    """
    if len(seq) < order:
        return 3.0, True

    vocab_size = len(vocab)
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

    if not log_probs:
        return 3.0, True

    nll = -float(np.mean(log_probs))
    ppl = float(np.exp(nll))
    return round(ppl, 4), False


def normalize_perplexity(ppl: float, p01: float, p99: float) -> float:
    """Clips and scales raw perplexity to [0.0, 1.0] using training percentiles."""
    if p99 <= p01:
        return 0.50
    clipped = np.clip(ppl, p01, p99)
    norm = (clipped - p01) / (p99 - p01)
    return float(round(norm, 4))


# =====================================================================
# 2. Main Scoring Pipeline
# =====================================================================

def run_sequence_scoring():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing sequence model at {MODEL_PATH}. Run sequence/train.py first.")
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing payments data at {PAYMENTS_PATH}")

    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model = json.load(f)

    order = model["order"]
    smoothing = model["smoothing"]
    vocab = model["event_vocabulary"]
    t_small = model["amount_buckets"]["small_max"]
    t_med = model["amount_buckets"]["medium_max"]
    p01 = model["normalization_stats"]["ppl_p01"]
    p99 = model["normalization_stats"]["ppl_p99"]

    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])
    df_payments = df_payments.sort_values("timestamp").reset_index(drop=True)

    total_n = len(df_payments)
    train_end = int(0.70 * total_n)
    val_end = int(0.85 * total_n)

    # 1. Build Chronological Sequences Across Entire History
    events = []
    for _, row in df_payments.iterrows():
        ts = row["timestamp"]
        amt = float(row["amount"])
        s = str(row["sender"])
        r = str(row["receiver"])

        def bucket_event(direction, amount):
            if amount <= t_small:
                b = "SMALL"
            elif amount <= t_med:
                b = "MEDIUM"
            else:
                b = "LARGE"
            return f"{direction}_{b}"

        events.append((s, ts, bucket_event("OUT", amt)))
        events.append((r, ts, bucket_event("IN", amt)))

    events.sort(key=lambda x: x[1])

    account_sequences = defaultdict(list)
    for acc, _, ev in events:
        account_sequences[acc].append(ev)

    # Collect fraud accounts and ring associations for offline evaluation
    fraud_txs = df_payments[df_payments["is_fraud"] == 1]
    fraud_accounts = set(fraud_txs["sender"]).union(set(fraud_txs["receiver"]))
    account_ring_map = {}
    for _, row in fraud_txs.iterrows():
        ring = str(row.get("ring_id", ""))
        if ring and ring not in ["NORMAL", "NONE", "nan"]:
            account_ring_map[row["sender"]] = ring
            account_ring_map[row["receiver"]] = ring

    # 2. Score Accounts
    account_records = []
    insufficient_count = 0

    all_accounts = sorted(list(account_sequences.keys()))
    for acc in all_accounts:
        seq = account_sequences[acc]
        ppl, is_short = calculate_sequence_perplexity(
            seq=seq,
            order=order,
            vocab=vocab,
            prefix_counts=model["prefix_counts"],
            transition_counts=model["transition_counts"],
            smoothing=smoothing
        )

        if is_short:
            insufficient_count += 1
            norm_score = 0.50
        else:
            norm_score = normalize_perplexity(ppl, p01, p99)

        account_records.append({
            "account_id": acc,
            "transaction_count": len(seq),
            "sequence_length": len(seq),
            "sequence_perplexity": ppl,
            "sequence_anomaly_score": norm_score,
            "insufficient_sequence_history": is_short,
            "is_fraud_account": 1 if acc in fraud_accounts else 0,
            "ring_id": account_ring_map.get(acc, "NONE")
        })

    df_account_scores = pd.DataFrame(account_records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_account_scores.to_csv(ACCOUNT_SCORES_PATH, index=False)

    # 3. Compute Transaction-Level Scores
    acc_map = df_account_scores.set_index("account_id")
    ppl_dict = acc_map["sequence_perplexity"].to_dict()
    anom_dict = acc_map["sequence_anomaly_score"].to_dict()

    tx_rows = []
    for _, row in df_payments.iterrows():
        tx_id = str(row["transaction_id"])
        s = str(row["sender"])
        r = str(row["receiver"])

        s_ppl = ppl_dict.get(s, 3.0)
        r_ppl = ppl_dict.get(r, 3.0)
        s_anom = anom_dict.get(s, 0.50)
        r_anom = anom_dict.get(r, 0.50)
        max_anom = max(s_anom, r_anom)

        tx_rows.append({
            "transaction_id": tx_id,
            "timestamp": str(row["timestamp"]),
            "sender": s,
            "receiver": r,
            "sender_sequence_perplexity": s_ppl,
            "receiver_sequence_perplexity": r_ppl,
            "sender_sequence_anomaly": s_anom,
            "receiver_sequence_anomaly": r_anom,
            "transaction_sequence_score": max_anom,
            "is_fraud": int(row.get("is_fraud", 0)),
            "ring_id": str(row.get("ring_id", "NONE"))
        })

    df_tx_scores = pd.DataFrame(tx_rows)
    df_tx_scores.to_csv(TX_SCORES_PATH, index=False)

    # 4. Optional Pair + Sequence Experimental Combination
    pair_source_path = RETRAINED_SCORES_PATH if RETRAINED_SCORES_PATH.exists() else ORIGINAL_SCORES_PATH
    df_combined = None
    if pair_source_path.exists():
        df_pair = pd.read_csv(pair_source_path)
        pair_col = "pair_risk" if "pair_risk" in df_pair.columns else "risk_score"
        df_merged_exp = df_tx_scores.merge(df_pair[["transaction_id", pair_col]], on="transaction_id", how="inner")
        df_merged_exp.rename(columns={pair_col: "pair_risk"}, inplace=True)
        # Experimental combination: 80% pair + 20% sequence
        df_merged_exp["combined_score"] = (
            0.80 * df_merged_exp["pair_risk"] +
            0.20 * df_merged_exp["transaction_sequence_score"]
        ).clip(0.0, 1.0).round(4)
        df_merged_exp.to_csv(COMBINED_SCORES_PATH, index=False)
        df_combined = df_merged_exp

    # 5. Held-Out Evaluation (15% Chronological Slice)
    df_heldout = df_tx_scores.iloc[val_end:].copy()
    y_test = df_heldout["is_fraud"].values
    seq_scores_test = df_heldout["transaction_sequence_score"].values

    def compute_eval_metrics(y_true, scores, threshold):
        roc = roc_auc_score(y_true, scores)
        p_c, r_c, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(r_c, p_c)
        preds = (scores >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-8)
        return {"ROC-AUC": roc, "PR-AUC": pr_auc, "Precision": p, "Recall": r, "F1": f1, "FPR": fpr}

    seq_metrics = compute_eval_metrics(y_test, seq_scores_test, threshold=ANOMALY_THRESHOLD)

    # Evaluate comparison with Pair model if available
    pair_metrics = None
    comb_metrics = None
    if df_combined is not None:
        df_comb_heldout = df_combined.iloc[val_end:].copy()
        pair_metrics = compute_eval_metrics(y_test, df_comb_heldout["pair_risk"].values, threshold=0.70)
        comb_metrics = compute_eval_metrics(y_test, df_comb_heldout["combined_score"].values, threshold=0.70)

    # 6. Ring-Level Analysis
    ring_analysis = {}
    rings_in_data = [r for r in df_tx_scores["ring_id"].unique() if r not in ["NORMAL", "NONE", "nan", ""]]
    for r in sorted(rings_in_data):
        r_df = df_tx_scores[df_tx_scores["ring_id"] == r]
        unique_accs = set(r_df["sender"]).union(set(r_df["receiver"]))
        avg_anom = float(round(r_df["transaction_sequence_score"].mean(), 4))
        max_anom = float(round(r_df["transaction_sequence_score"].max(), 4))
        ring_analysis[r] = {
            "transactions": len(r_df),
            "accounts": len(unique_accs),
            "avg_sequence_anomaly": avg_anom,
            "max_sequence_anomaly": max_anom
        }

    # Save Evaluation JSON
    eval_payload = {
        "evaluation_split": "Held-Out (15% Chronological)",
        "held_out_transactions": len(df_heldout),
        "held_out_fraud_transactions": int(np.sum(y_test == 1)),
        "accounts_total": len(df_account_scores),
        "accounts_with_insufficient_history": insufficient_count,
        "experimental_threshold": ANOMALY_THRESHOLD,
        "sequence_metrics": seq_metrics,
        "ring_analysis": ring_analysis
    }
    if pair_metrics:
        eval_payload["pair_metrics"] = pair_metrics
        eval_payload["combined_metrics"] = comb_metrics

    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    # ==========================================================
    # Terminal Report
    # ==========================================================
    heldout_accs = set(df_heldout["sender"]).union(set(df_heldout["receiver"]))

    print("=======================================================")
    print("        SEQUENCE PERPLEXITY EVALUATION                 ")
    print("=======================================================")
    print(f"Held-Out Accounts:        {len(heldout_accs)}")
    print(f"Held-Out Transactions:    {len(df_heldout)}")
    print(f"Held-Out Fraud Count:     {int(np.sum(y_test == 1))}")
    print(f"Accounts (Short History): {insufficient_count} / {len(df_account_scores)}")

    print(f"\nSequence ROC-AUC:         {seq_metrics['ROC-AUC']:.4f}")
    print(f"Sequence PR-AUC:          {seq_metrics['PR-AUC']:.4f}")

    print(f"\nExperimental Threshold:   {ANOMALY_THRESHOLD:.2f}")
    print(f"Precision:                {seq_metrics['Precision']:.4f}")
    print(f"Recall:                   {seq_metrics['Recall']:.4f}")
    print(f"F1:                       {seq_metrics['F1']:.4f}")
    print(f"FPR:                      {seq_metrics['FPR']:.4f}")

    print("\n-------------------------------------------------------")
    print("RING ANALYSIS (Sequence Anomaly Profile)")
    print("-------------------------------------------------------")
    for r, data in ring_analysis.items():
        print(f"{r}:")
        print(f"  Transactions: {data['transactions']} | Accounts: {data['accounts']}")
        print(f"  Avg anomaly:  {data['avg_sequence_anomaly']:.4f}")
        print(f"  Max anomaly:  {data['max_sequence_anomaly']:.4f}")

    if pair_metrics and comb_metrics:
        print("\n-------------------------------------------------------")
        print("PAIR MODEL VS PAIR + SEQUENCE (EXPERIMENTAL ONLY)")
        print("-------------------------------------------------------")
        comp_df = pd.DataFrame([pair_metrics, comb_metrics], index=["Pair Model", "Pair + Sequence (80/20)"])
        print(comp_df.round(4).to_string())

    print("\n-------------------------------------------------------")
    print("INTERPRETATION")
    print("-------------------------------------------------------")
    print("Sequence perplexity provides an independent behavioural signal.")
    print("High perplexity indicates that an account's transaction-order pattern")
    print("is unusual compared with behaviour observed during training.")
    print("\nThis experiment does not replace the Pair Risk Model.")
    print("It is an auxiliary P2 signal that can potentially complement")
    print("transaction-level and graph-based fraud detection.")

    print(f"\nArtifacts:")
    print(f"  Account Scores:         {ACCOUNT_SCORES_PATH}")
    print(f"  Transaction Scores:     {TX_SCORES_PATH}")
    print(f"  Evaluation JSON:        {EVALUATION_PATH}")
    if df_combined is not None:
        print(f"  Combined Scores (Exp):  {COMBINED_SCORES_PATH}")
    print("=======================================================\n")


if __name__ == "__main__":
    run_sequence_scoring()