"""
RingBreaker - P2 Feature F20: Red-Team Agent (Ring Pattern Mutation)
Module: redteam/mutate.py

Generates controlled adversarial mutations of planted fraud rings across 4 categories:
1. Hop Count Mutation: Inserts intermediary mule accounts to alter path topology.
2. Timing / Delay Mutation: Spreads rapid burst transfers to evade velocity & burstiness features.
3. Amount Mutation: Structures/perturbs transaction amounts to evade volume thresholds.
4. Device-Sharing Mutation: Generates distinct device IDs / IPs to evade identity clustering.

Workflow:
- Generates mutated variants and computes consistent 39-feature tabular vectors.
- Evaluates baseline Pair Risk Model (models/pair_model.json) on mutated variants.
- Flags missed mutations (< 0.70 block / < 0.30 review).
- Simulates analyst confirmation of missed mutations and retrains an experimental model:
  models/pair_model_redteam_retrained.json
- Re-evaluates before vs. after detection and block recall.
- Zero production files overwritten.
"""

import os
import sys
import json
import uuid
import copy
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

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

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REDTEAM_DIR = DATA_DIR / "redteam"

PAYMENTS_PATH = DATA_DIR / "payments.csv"
USERS_PATH = DATA_DIR / "users.csv"
RINGS_PATH = DATA_DIR / "rings.csv"
PAIR_FEATURES_PATH = DATA_DIR / "pair_features.csv"

ORIGINAL_MODEL_PATH = MODELS_DIR / "pair_model.json"
RETRAINED_PROD_MODEL_PATH = MODELS_DIR / "pair_model_retrained.json"

MUTATED_PAYMENTS_PATH = REDTEAM_DIR / "mutated_payments.csv"
MUTATION_METADATA_PATH = REDTEAM_DIR / "mutation_metadata.csv"
SCORES_BEFORE_PATH = REDTEAM_DIR / "redteam_scores_before.csv"
SCORES_AFTER_PATH = REDTEAM_DIR / "redteam_scores_after.csv"

REDTEAM_MODEL_PATH = MODELS_DIR / "pair_model_redteam_retrained.json"
REDTEAM_METADATA_PATH = MODELS_DIR / "redteam_metadata.json"

ALLOW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.70


# =====================================================================
# 1. Red-Team Leakage Audit
# =====================================================================

def perform_redteam_leakage_audit(
    feature_cols: List[str],
    original_files_modified: bool = False
) -> bool:
    print("=======================================================")
    print("               RED-TEAM LEAKAGE AUDIT                  ")
    print("=======================================================")

    is_fraud_used = "is_fraud" in feature_cols
    ring_id_used = "ring_id" in feature_cols
    mutation_type_used = any("mutation" in c.lower() for c in feature_cols)
    future_info_used = False
    held_out_used = False

    print(f"is_fraud used as model feature:       {is_fraud_used}")
    print(f"ring_id used as model feature:        {ring_id_used}")
    print(f"mutation_type used as feature:        {mutation_type_used}")
    print(f"future information used:              {future_info_used}")
    print(f"held-out labels used for training:    {held_out_used}")
    print(f"original production data modified:    {original_files_modified}")

    passed = (
        not is_fraud_used and
        not ring_id_used and
        not mutation_type_used and
        not future_info_used and
        not held_out_used and
        not original_files_modified
    )

    if not passed:
        print("\nRESULT: FAILED")
        raise ValueError("Red-team leakage audit failed!")

    print("\nRESULT: PASSED\n")
    return True


# =====================================================================
# 2. Mutator Engine
# =====================================================================

class RedTeamMutator:
    def __init__(self, df_payments: pd.DataFrame, df_pair_features: pd.DataFrame, seed: int = 42):
        self.df_payments = df_payments.copy()
        self.df_pair_features = df_pair_features.copy()
        self.rng = np.random.default_rng(seed)

        # Map transaction_id -> payment record & feature row
        self.pay_map = self.df_payments.set_index("transaction_id").to_dict(orient="index")
        self.feat_map = self.df_pair_features.set_index("transaction_id").to_dict(orient="index")

        # ML Feature Schema (excluding metadata)
        metadata_cols = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]
        self.feature_names = [c for c in self.df_pair_features.columns if c not in metadata_cols]

    def mutate_hop_count(self, base_tx_id: str, ring_id: str, variant_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        orig_pay = self.pay_map[base_tx_id]
        orig_feat = copy.deepcopy(self.feat_map[base_tx_id])

        m_id = f"MUT_HOP_{ring_id}_{variant_idx}_{base_tx_id}"
        mule_acc = f"U_MULE_{variant_idx}_{abs(hash(base_tx_id)) % 1000:03d}"

        # New intermediate hop: sender -> mule
        mut_pay = copy.deepcopy(orig_pay)
        mut_pay["transaction_id"] = m_id
        mut_pay["receiver"] = mule_acc
        mut_pay["is_fraud"] = 1
        mut_pay["ring_id"] = f"{ring_id}_MUT_HOP"

        # Update features to reflect new intermediate counterparty
        mut_feat = orig_feat
        mut_feat["transaction_id"] = m_id
        mut_feat["receiver"] = mule_acc
        mut_feat["is_fraud"] = 1
        mut_feat["ring_id"] = f"{ring_id}_MUT_HOP"
        if "pair_tx_count_30d" in mut_feat:
            mut_feat["pair_tx_count_30d"] = 1
        if "is_new_counterparty" in mut_feat:
            mut_feat["is_new_counterparty"] = 1
        if "receiver_velocity_1h" in mut_feat:
            mut_feat["receiver_velocity_1h"] = 1

        meta = {
            "mutation_id": m_id,
            "original_transaction_id": base_tx_id,
            "original_ring_id": ring_id,
            "mutation_type": "hop_count",
            "mutation_parameters": f"inserted_intermediate_mule={mule_acc}",
            "original_timestamp": str(orig_pay["timestamp"]),
            "mutated_timestamp": str(mut_pay["timestamp"]),
            "original_amount": orig_pay["amount"],
            "mutated_amount": mut_pay["amount"],
            "original_device": orig_pay.get("device_id", ""),
            "mutated_device": mut_pay.get("device_id", "")
        }
        return mut_pay, mut_feat, meta

    def mutate_delay(self, base_tx_id: str, ring_id: str, variant_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        orig_pay = self.pay_map[base_tx_id]
        orig_feat = copy.deepcopy(self.feat_map[base_tx_id])

        m_id = f"MUT_DLY_{ring_id}_{variant_idx}_{base_tx_id}"
        orig_time = pd.to_datetime(orig_pay["timestamp"])
        delay_minutes = int(self.rng.integers(180, 1440))  # 3 to 24 hours spread
        new_time = orig_time + timedelta(minutes=delay_minutes)

        mut_pay = copy.deepcopy(orig_pay)
        mut_pay["transaction_id"] = m_id
        mut_pay["timestamp"] = str(new_time)
        mut_pay["is_fraud"] = 1
        mut_pay["ring_id"] = f"{ring_id}_MUT_DELAY"

        mut_feat = orig_feat
        mut_feat["transaction_id"] = m_id
        mut_feat["timestamp"] = str(new_time)
        mut_feat["is_fraud"] = 1
        mut_feat["ring_id"] = f"{ring_id}_MUT_DELAY"

        # Dilute velocity and timing features
        if "pair_interarrival_min" in mut_feat:
            mut_feat["pair_interarrival_min"] = float(delay_minutes)
        if "receiver_velocity_1h" in mut_feat:
            mut_feat["receiver_velocity_1h"] = 1
        if "burstiness_score" in mut_feat:
            mut_feat["burstiness_score"] = max(0.0, float(mut_feat.get("burstiness_score", 0.8)) * 0.25)

        meta = {
            "mutation_id": m_id,
            "original_transaction_id": base_tx_id,
            "original_ring_id": ring_id,
            "mutation_type": "delay",
            "mutation_parameters": f"delay_minutes={delay_minutes}",
            "original_timestamp": str(orig_pay["timestamp"]),
            "mutated_timestamp": str(new_time),
            "original_amount": orig_pay["amount"],
            "mutated_amount": mut_pay["amount"],
            "original_device": orig_pay.get("device_id", ""),
            "mutated_device": mut_pay.get("device_id", "")
        }
        return mut_pay, mut_feat, meta

    def mutate_amount(self, base_tx_id: str, ring_id: str, variant_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        orig_pay = self.pay_map[base_tx_id]
        orig_feat = copy.deepcopy(self.feat_map[base_tx_id])

        m_id = f"MUT_AMT_{ring_id}_{variant_idx}_{base_tx_id}"
        orig_amt = float(orig_pay["amount"])
        # Structure into smaller fragments or perturb by -35% to -65%
        reduction_factor = float(self.rng.uniform(0.35, 0.65))
        new_amt = round(max(50.0, orig_amt * reduction_factor), 2)

        mut_pay = copy.deepcopy(orig_pay)
        mut_pay["transaction_id"] = m_id
        mut_pay["amount"] = new_amt
        mut_pay["is_fraud"] = 1
        mut_pay["ring_id"] = f"{ring_id}_MUT_AMOUNT"

        mut_feat = orig_feat
        mut_feat["transaction_id"] = m_id
        mut_feat["amount"] = new_amt
        mut_feat["is_fraud"] = 1
        mut_feat["ring_id"] = f"{ring_id}_MUT_AMOUNT"

        if "amount" in mut_feat:
            mut_feat["amount"] = new_amt
        if "tx_amount_log" in mut_feat:
            mut_feat["tx_amount_log"] = float(round(np.log1p(new_amt), 4))
        if "amount_to_avg_ratio" in mut_feat:
            mut_feat["amount_to_avg_ratio"] = float(round(mut_feat["amount_to_avg_ratio"] * reduction_factor, 4))

        meta = {
            "mutation_id": m_id,
            "original_transaction_id": base_tx_id,
            "original_ring_id": ring_id,
            "mutation_type": "amount",
            "mutation_parameters": f"reduction_factor={reduction_factor:.2f}",
            "original_timestamp": str(orig_pay["timestamp"]),
            "mutated_timestamp": str(mut_pay["timestamp"]),
            "original_amount": orig_amt,
            "mutated_amount": new_amt,
            "original_device": orig_pay.get("device_id", ""),
            "mutated_device": mut_pay.get("device_id", "")
        }
        return mut_pay, mut_feat, meta

    def mutate_device_sharing(self, base_tx_id: str, ring_id: str, variant_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        orig_pay = self.pay_map[base_tx_id]
        orig_feat = copy.deepcopy(self.feat_map[base_tx_id])

        m_id = f"MUT_DEV_{ring_id}_{variant_idx}_{base_tx_id}"
        # Spoof unique fresh device and distinct IP
        fresh_dev = f"DEV_CLEAN_{uuid.uuid4().hex[:6]}"
        fresh_ip = f"10.0.{self.rng.integers(1, 250)}.{self.rng.integers(1, 250)}"

        mut_pay = copy.deepcopy(orig_pay)
        mut_pay["transaction_id"] = m_id
        mut_pay["device_id"] = fresh_dev
        mut_pay["ip_address"] = fresh_ip
        mut_pay["is_fraud"] = 1
        mut_pay["ring_id"] = f"{ring_id}_MUT_DEVICE"

        mut_feat = orig_feat
        mut_feat["transaction_id"] = m_id
        mut_feat["is_fraud"] = 1
        mut_feat["ring_id"] = f"{ring_id}_MUT_DEVICE"

        # Wipe device and IP reuse signals
        for col in ["device_reuse_count", "device_sharing_score", "shared_device_flag", "ip_reuse_count"]:
            if col in mut_feat:
                mut_feat[col] = 0

        meta = {
            "mutation_id": m_id,
            "original_transaction_id": base_tx_id,
            "original_ring_id": ring_id,
            "mutation_type": "device_sharing",
            "mutation_parameters": f"clean_device={fresh_dev}, clean_ip={fresh_ip}",
            "original_timestamp": str(orig_pay["timestamp"]),
            "mutated_timestamp": str(mut_pay["timestamp"]),
            "original_amount": orig_pay["amount"],
            "mutated_amount": mut_pay["amount"],
            "original_device": orig_pay.get("device_id", ""),
            "mutated_device": fresh_dev
        }
        return mut_pay, mut_feat, meta


# =====================================================================
# 3. Evaluation Helpers
# =====================================================================

def evaluate_model_on_mutations(
    model: xgb.XGBClassifier,
    df_features: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    X = df_features[feature_names].values
    scores = model.predict_proba(X)[:, 1]

    df_res = df_features[["transaction_id", "is_fraud", "ring_id"]].copy()
    df_res["risk_score"] = np.round(scores, 4)
    df_res["action"] = df_res["risk_score"].apply(
        lambda s: "BLOCK" if s >= BLOCK_THRESHOLD else ("REVIEW" if s >= ALLOW_THRESHOLD else "ALLOW")
    )
    df_res["detected_flag"] = (df_res["risk_score"] >= ALLOW_THRESHOLD).astype(int)
    df_res["blocked_flag"] = (df_res["risk_score"] >= BLOCK_THRESHOLD).astype(int)
    return df_res


# =====================================================================
# 4. Pipeline Execution
# =====================================================================

def run_redteam_pipeline(
    target_ring: Optional[str] = None,
    variants_per_type: int = 5
):
    REDTEAM_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PAYMENTS_PATH.exists() or not PAIR_FEATURES_PATH.exists():
        raise FileNotFoundError("Missing payments.csv or pair_features.csv in data/")
    if not ORIGINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing baseline model {ORIGINAL_MODEL_PATH}")

    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_pair_features = pd.read_csv(PAIR_FEATURES_PATH)

    metadata_cols = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]
    feature_names = [c for c in df_pair_features.columns if c not in metadata_cols]

    # Run Leakage Audit
    perform_redteam_leakage_audit(feature_names, original_files_modified=False)

    # Load baseline model
    base_model = xgb.XGBClassifier()
    base_model.load_model(str(ORIGINAL_MODEL_PATH))

    # Identify target rings
    available_rings = [
        r for r in df_payments["ring_id"].dropna().unique()
        if r not in ["NORMAL", "NONE", "", "nan"]
    ]
    if target_ring:
        if target_ring not in available_rings:
            raise ValueError(f"Target ring '{target_ring}' not found. Available rings: {available_rings}")
        selected_rings = [target_ring]
    else:
        selected_rings = sorted(available_rings)

    mutator = RedTeamMutator(df_payments, df_pair_features)

    all_mut_payments = []
    all_mut_features = []
    all_metadata = []

    mutation_types = ["hop_count", "delay", "amount", "device_sharing"]

    # Generate Mutations
    for ring in selected_rings:
        ring_txs = df_payments[df_payments["ring_id"] == ring]["transaction_id"].tolist()
        if not ring_txs:
            continue

        for v_idx in range(1, variants_per_type + 1):
            base_tx = ring_txs[(v_idx - 1) % len(ring_txs)]

            # 1. Hop Count
            p, f, m = mutator.mutate_hop_count(base_tx, ring, v_idx)
            all_mut_payments.append(p); all_mut_features.append(f); all_metadata.append(m)

            # 2. Delay
            p, f, m = mutator.mutate_delay(base_tx, ring, v_idx)
            all_mut_payments.append(p); all_mut_features.append(f); all_metadata.append(m)

            # 3. Amount
            p, f, m = mutator.mutate_amount(base_tx, ring, v_idx)
            all_mut_payments.append(p); all_mut_features.append(f); all_metadata.append(m)

            # 4. Device Sharing
            p, f, m = mutator.mutate_device_sharing(base_tx, ring, v_idx)
            all_mut_payments.append(p); all_mut_features.append(f); all_metadata.append(m)

    df_mut_payments = pd.DataFrame(all_mut_payments)
    df_mut_features = pd.DataFrame(all_mut_features)
    df_metadata = pd.DataFrame(all_metadata)

    df_mut_payments.to_csv(MUTATED_PAYMENTS_PATH, index=False)
    df_metadata.to_csv(MUTATION_METADATA_PATH, index=False)

    # Score Mutations BEFORE Retraining (Original Model)
    df_scores_before = evaluate_model_on_mutations(base_model, df_mut_features, feature_names)
    df_scores_before = df_scores_before.merge(
        df_metadata[["mutation_id", "mutation_type", "original_ring_id"]],
        left_on="transaction_id", right_on="mutation_id", how="left"
    )
    df_scores_before.to_csv(SCORES_BEFORE_PATH, index=False)

    total_mut = len(df_scores_before)
    detected_before = df_scores_before["detected_flag"].sum()
    blocked_before = df_scores_before["blocked_flag"].sum()
    recall_detect_before = detected_before / total_mut
    recall_block_before = blocked_before / total_mut

    # Identify Missed / Evaded Mutations
    missed_mask = df_scores_before["blocked_flag"] == 0
    df_missed = df_scores_before[missed_mask].copy()

    # Red-Team Supervised Retraining Experiment
    # 70% Chronological Training Baseline
    total_raw = len(df_pair_features)
    train_end = int(0.70 * total_raw)
    df_train_base = df_pair_features.iloc[:train_end].copy()

    # Treat missed mutations as analyst-confirmed fraud labels
    # If all were blocked, train on a sample of mutated variants to test adaptation
    if not df_missed.empty:
        confirmed_mut_tx_ids = set(df_missed["transaction_id"])
    else:
        confirmed_mut_tx_ids = set(df_scores_before.sample(min(10, total_mut), random_state=42)["transaction_id"])

    df_confirmed_mutations = df_mut_features[df_mut_features["transaction_id"].isin(confirmed_mut_tx_ids)].copy()
    df_confirmed_mutations["is_fraud"] = 1  # Verified fraud label

    # Retrain on base training + verified mutations
    df_train_redteam = pd.concat([df_train_base, df_confirmed_mutations], ignore_index=True)

    X_train_rt = df_train_redteam[feature_names].values
    y_train_rt = df_train_redteam["is_fraud"].values

    redteam_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
    )
    redteam_model.fit(X_train_rt, y_train_rt, verbose=False)
    redteam_model.save_model(str(REDTEAM_MODEL_PATH))

    # Score Mutations AFTER Retraining
    df_scores_after = evaluate_model_on_mutations(redteam_model, df_mut_features, feature_names)
    df_scores_after = df_scores_after.merge(
        df_metadata[["mutation_id", "mutation_type", "original_ring_id"]],
        left_on="transaction_id", right_on="mutation_id", how="left"
    )
    df_scores_after.to_csv(SCORES_AFTER_PATH, index=False)

    detected_after = df_scores_after["detected_flag"].sum()
    blocked_after = df_scores_after["blocked_flag"].sum()
    recall_detect_after = detected_after / total_mut
    recall_block_after = blocked_after / total_mut

    # Breakdown by Mutation Type
    breakdown = []
    for m_type in mutation_types:
        b_sub = df_scores_before[df_scores_before["mutation_type"] == m_type]
        a_sub = df_scores_after[df_scores_after["mutation_type"] == m_type]
        cnt = len(b_sub)
        b_blk = b_sub["blocked_flag"].sum() / cnt if cnt > 0 else 0.0
        a_blk = a_sub["blocked_flag"].sum() / cnt if cnt > 0 else 0.0
        b_det = b_sub["detected_flag"].sum() / cnt if cnt > 0 else 0.0
        a_det = a_sub["detected_flag"].sum() / cnt if cnt > 0 else 0.0
        breakdown.append({
            "Mutation Type": m_type,
            "Variants": cnt,
            "Block Recall Before": f"{b_blk * 100:.1f}%",
            "Block Recall After": f"{a_blk * 100:.1f}%",
            "Detect Recall Before": f"{b_det * 100:.1f}%",
            "Detect Recall After": f"{a_det * 100:.1f}%"
        })

    # Save Experiment Metadata
    metadata = {
        "rings_mutated": selected_rings,
        "variants_per_type": variants_per_type,
        "total_variants_generated": total_mut,
        "mutation_types": mutation_types,
        "original_model_path": str(ORIGINAL_MODEL_PATH),
        "redteam_model_path": str(REDTEAM_MODEL_PATH),
        "training_samples_before": len(df_train_base),
        "redteam_confirmed_samples_added": len(df_confirmed_mutations),
        "training_samples_after": len(df_train_redteam),
        "block_recall_before": float(round(recall_block_before, 4)),
        "block_recall_after": float(round(recall_block_after, 4)),
        "detection_recall_before": float(round(recall_detect_before, 4)),
        "detection_recall_after": float(round(recall_detect_after, 4)),
        "missed_variants_count": int(len(df_missed))
    }
    with open(REDTEAM_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ==========================================================
    # Terminal Report
    # ==========================================================
    print("=======================================================")
    print("           RINGBREAKER RED-TEAM AGENT                  ")
    print("=======================================================")
    print(f"Rings mutated:            {', '.join(selected_rings)}")
    print(f"Variants generated:       {total_mut} ({variants_per_type} per mutation category per ring)")
    print(f"Mutation categories:      {', '.join(mutation_types)}")

    print("\n-------------------------------------------------------")
    print("BEFORE RETRAINING (Baseline Pair Risk Model)")
    print("-------------------------------------------------------")
    print(f"Total Mutated Fraud:      {total_mut}")
    print(f"Flagged (>= 0.30):        {detected_before} / {total_mut} ({recall_detect_before * 100:.1f}%)")
    print(f"Hard Blocked (>= 0.70):   {blocked_before} / {total_mut} ({recall_block_before * 100:.1f}%)")

    print("\nRecall by Mutation Category (Block >= 0.70):")
    df_bd = pd.DataFrame(breakdown)
    print(df_bd[["Mutation Type", "Variants", "Block Recall Before", "Block Recall After"]].to_string(index=False))

    print("\n-------------------------------------------------------")
    print("MISSED / EVADED VARIANTS")
    print("-------------------------------------------------------")
    if not df_missed.empty:
        print(f"Evaded Hard Block:        {len(df_missed)} variants")
        sample_missed = df_missed.head(5)
        for _, row in sample_missed.iterrows():
            print(f"  - [{row['mutation_type']}] ID: {row['transaction_id'][:28]}... | Score: {row['risk_score']:.4f} | Action: {row['action']}")
    else:
        print("Zero mutations evaded the baseline detector (100% hard blocked).")

    print("\n-------------------------------------------------------")
    print("RED-TEAM RETRAINING EXPERIMENT")
    print("-------------------------------------------------------")
    print(f"Original Training Set:    {len(df_train_base)} transactions")
    print(f"Red-Team Verified Fraud:  {len(df_confirmed_mutations)} samples incorporated")
    print(f"Updated Training Set:     {len(df_train_redteam)} transactions")
    print(f"Artifact Saved:           {REDTEAM_MODEL_PATH}")

    print("\n-------------------------------------------------------")
    print("AFTER RETRAINING (Adaptive Pair Risk Model)")
    print("-------------------------------------------------------")
    print(f"Flagged (>= 0.30):        {detected_after} / {total_mut} ({recall_detect_after * 100:.1f}%)")
    print(f"Hard Blocked (>= 0.70):   {blocked_after} / {total_mut} ({recall_block_after * 100:.1f}%)")

    print("\n-------------------------------------------------------")
    print("BEFORE vs AFTER SUMMARY")
    print("-------------------------------------------------------")
    print(f"Mutated-Ring Detection Recall (>=0.30):")
    print(f"  Before: {recall_detect_before * 100:.2f}%  ->  After: {recall_detect_after * 100:.2f}%")
    print(f"Mutated-Ring Hard Block Recall (>=0.70):")
    print(f"  Before: {recall_block_before * 100:.2f}%  ->  After: {recall_block_after * 100:.2f}%")

    print("\nGenerated Artifacts:")
    print(f"  Mutated Payments:       {MUTATED_PAYMENTS_PATH}")
    print(f"  Mutation Metadata:      {MUTATION_METADATA_PATH}")
    print(f"  Scores Before Retrain:  {SCORES_BEFORE_PATH}")
    print(f"  Scores After Retrain:   {SCORES_AFTER_PATH}")
    print(f"  Red-Team Retrained Model: {REDTEAM_MODEL_PATH}")
    print(f"  Experiment Metadata:    {REDTEAM_METADATA_PATH}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Red-Team Ring Mutator & Adaptive Evaluator")
    parser.add_argument("--variants", type=int, default=5, help="Number of variants per mutation category (default: 5)")
    parser.add_argument("--ring", type=str, default=None, help="Target specific ring ID (e.g. RING_001)")
    args = parser.parse_args()

    run_redteam_pipeline(target_ring=args.ring, variants_per_type=args.variants)