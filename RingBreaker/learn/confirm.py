"""
RingBreaker - Analyst Confirmation & Feedback Action Loop
Module: learn/confirm.py

Connects:
Analyst Action (CONFIRM_FRAUD)
    -> Temporal Risk Propagation (learn/propagate.py)
    -> Supervised Retraining (learn/retrain.py)
    -> Case File Status Update (data/case_files/ALERT_<TX_ID>.json)

Also supports:
MARK_FALSE_POSITIVE (Updates case status without model retraining)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
CASE_FILES_DIR = DATA_DIR / "case_files"
PROPAGATED_RISK_PATH = DATA_DIR / "propagated_risk.csv"
RETRAIN_METADATA_PATH = PROJECT_ROOT / "models" / "retrain_metadata.json"
RETRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "pair_model_retrained.json"

# Re-use existing propagation, retraining, and case file builders
from learn.propagate import run_propagation
from learn.retrain import run_retrain
from explain.case_file import run_case_file_pipeline


# =====================================================================
# 1. Case File Status Updater
# =====================================================================

def update_case_file_status(
    tx_id: str,
    new_status: str,
    confirmation_meta: Optional[Dict[str, Any]] = None
) -> Path:
    """Updates the analyst section of the corresponding case file JSON."""
    CASE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    case_path = CASE_FILES_DIR / f"ALERT_{tx_id}.json"

    # If the case file doesn't exist yet on disk, build it first
    if not case_path.exists():
        run_case_file_pipeline(target_tx=tx_id)

    # Read existing case file
    with open(case_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    # Update analyst block
    if "analyst" not in case_data:
        case_data["analyst"] = {}

    case_data["analyst"]["status"] = new_status
    case_data["analyst"]["updated_at"] = datetime.now().isoformat()

    if confirmation_meta:
        for k, v in confirmation_meta.items():
            case_data["analyst"][k] = v

    # Write back updated JSON
    with open(case_path, "w", encoding="utf-8") as f:
        json.dump(case_data, f, indent=2)

    return case_path


# =====================================================================
# 2. Confirm Fraud Workflow
# =====================================================================

def confirm_fraud(
    tx_id: str,
    confirmed_at_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the full Confirm-Fraud feedback loop:
    1. Validates transaction exists.
    2. Runs graph propagation strictly up to confirmation timestamp.
    3. Runs supervised retraining incorporating the confirmed fraud sample.
    4. Updates the alert case file to CONFIRMED_FRAUD.
    """
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {PAYMENTS_PATH}")

    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])

    tx_match = df_payments[df_payments["transaction_id"] == tx_id]
    if tx_match.empty:
        raise ValueError(f"Transaction ID '{tx_id}' not found in {PAYMENTS_PATH}")

    tx_row = tx_match.iloc[0]
    tx_timestamp = tx_row["timestamp"]
    sender = str(tx_row["sender"])
    receiver = str(tx_row["receiver"])

    print("=======================================================")
    print("       RINGBREAKER ANALYST CONFIRMATION                ")
    print("=======================================================")
    print(f"Transaction: {tx_id}")
    print(f"Action:      CONFIRM_FRAUD\n")
    print("Analyst confirmation accepted.")

    # 1. Run Propagation
    print("\n--- 1. TRIGGERING TEMPORAL RISK PROPAGATION ---")
    df_prop = run_propagation(
        confirmed_tx_id=tx_id,
        confirmed_at_str=confirmed_at_str
    )
    n_seeds = (df_prop["hop_distance"] == 0).sum() if not df_prop.empty else 0
    n_affected = len(df_prop)

    # 2. Run Supervised Retraining
    print("\n--- 2. TRIGGERING SUPERVISED PAIR RETRAINING ---")
    run_retrain(confirmed_tx_id=tx_id)

    # Read retrain metadata
    retrain_meta = {}
    if RETRAIN_METADATA_PATH.exists():
        with open(RETRAIN_METADATA_PATH, "r", encoding="utf-8") as f:
            retrain_meta = json.load(f)

    prev_samples = retrain_meta.get("training_samples_total", 0) - 1
    new_samples = retrain_meta.get("training_samples_total", 0)
    orig_risk = retrain_meta.get("confirmed_transaction_scoring", {}).get("original_risk", 0.0)
    retrained_risk = retrain_meta.get("confirmed_transaction_scoring", {}).get("retrained_risk", 0.0)

    # 3. Update Case File
    confirmation_meta = {
        "confirmed_at": str(datetime.now().isoformat()),
        "confirmed_transaction_id": tx_id,
        "transaction_timestamp": str(tx_timestamp),
        "propagation_output": str(PROPAGATED_RISK_PATH.relative_to(PROJECT_ROOT)),
        "affected_accounts_count": int(n_affected),
        "retrained_model": str(RETRAINED_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "retrained_risk_score": float(retrained_risk)
    }
    case_path = update_case_file_status(tx_id, "CONFIRMED_FRAUD", confirmation_meta)

    # Summary
    print("\n=======================================================")
    print("       RINGBREAKER CONFIRMATION SUMMARY                ")
    print("=======================================================")
    print(f"Transaction:             {tx_id}")
    print(f"Action:                  CONFIRM_FRAUD")
    print(f"Transaction Timestamp:   {tx_timestamp}")
    print("\nPropagation:")
    print(f"  Seeds:                 {n_seeds} ({sender}, {receiver})")
    print(f"  Affected accounts:     {n_affected}")
    print(f"  Artifact:              {PROPAGATED_RISK_PATH}")
    print("\nRetraining:")
    print(f"  Previous training:     {prev_samples} samples")
    print(f"  New training:          {new_samples} samples")
    print(f"  Artifact:              {RETRAINED_MODEL_PATH}")
    print("\nModel Prediction (TX):")
    print(f"  Original risk:         {orig_risk:.4f}")
    print(f"  Retrained risk:        {retrained_risk:.4f}")
    print("\nCase status:")
    print(f"  Status:                CONFIRMED_FRAUD")
    print(f"  Updated Case File:     {case_path}")
    print("=======================================================\n")

    return {
        "status": "CONFIRMED_FRAUD",
        "transaction_id": tx_id,
        "propagation_affected": n_affected,
        "original_risk": orig_risk,
        "retrained_risk": retrained_risk,
        "case_file": str(case_path)
    }


# =====================================================================
# 3. Mark False Positive Workflow
# =====================================================================

def mark_false_positive(tx_id: str) -> Dict[str, Any]:
    """Updates the case file status to FALSE_POSITIVE without retraining or propagation."""
    print("=======================================================")
    print("       RINGBREAKER ANALYST ACTION                      ")
    print("=======================================================")
    print(f"Transaction: {tx_id}")
    print(f"Action:      MARK_FALSE_POSITIVE\n")

    meta = {
        "marked_false_positive_at": str(datetime.now().isoformat()),
        "note": "Analyst determined alert is not fraudulent. No model retraining applied."
    }
    case_path = update_case_file_status(tx_id, "FALSE_POSITIVE", meta)

    print(f"Case status updated to: FALSE_POSITIVE")
    print(f"Updated Case File:      {case_path}")
    print("=======================================================\n")

    return {
        "status": "FALSE_POSITIVE",
        "transaction_id": tx_id,
        "case_file": str(case_path)
    }


# =====================================================================
# 4. CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Analyst Action Feedback Loop")
    parser.add_argument("--transaction", type=str, required=True, help="Transaction ID to act upon (e.g., TX_0008825)")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["confirm", "false_positive"],
        help="Analyst decision: 'confirm' or 'false_positive'"
    )
    parser.add_argument(
        "--confirmed-at",
        type=str,
        default=None,
        help="Explicit analyst confirmation timestamp (e.g. '2026-03-24 16:00:00')"
    )
    args = parser.parse_args()

    if args.action == "confirm":
        confirm_fraud(args.transaction, confirmed_at_str=args.confirmed_at)
    elif args.action == "false_positive":
        mark_false_positive(args.transaction)