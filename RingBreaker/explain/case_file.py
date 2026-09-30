"""
RingBreaker - Analyst Case File & Alert Evidence Layer
Module: explain/case_file.py

Aggregates:
1. Retrained Risk Engine decision (data/risk_scores_retrained.csv)
2. Raw transaction and device attributes (data/payments.csv)
3. Historical account registration context (data/users.csv)
4. SHAP model feature attributions (data/shap_explanations/ with lazy on-demand computation)
5. Known / planted ground-truth ring metadata (data/rings.csv)

Outputs:
- data/case_files/ALERT_<TX_ID>.json
- data/alerts.json (summary index of all generated alerts)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import xgboost as xgb
import shap

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
USERS_PATH = DATA_DIR / "users.csv"
RINGS_PATH = DATA_DIR / "rings.csv"
RISK_SCORES_PATH = DATA_DIR / "risk_scores_retrained.csv"
PAIR_FEATURES_PATH = DATA_DIR / "pair_features.csv"
RETRAINED_MODEL_PATH = MODELS_DIR / "pair_model_retrained.json"
ORIGINAL_MODEL_PATH = MODELS_DIR / "pair_model.json"
SHAP_DIR = DATA_DIR / "shap_explanations"
CASE_FILES_DIR = DATA_DIR / "case_files"
ALERTS_SUMMARY_PATH = DATA_DIR / "alerts.json"

DEFAULT_TARGET_TX = "TX_0008825"


# =====================================================================
# 1. Cached SHAP Explainer & Just-In-Time Generator
# =====================================================================

class ShapExplainerCache:
    """Singleton cache for XGBoost model, TreeExplainer, and pair features matrix."""
    _instance = None

    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.df_features = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        if self.explainer is not None:
            return

        # Prefer retrained model matching data/risk_scores_retrained.csv
        model_path = RETRAINED_MODEL_PATH if RETRAINED_MODEL_PATH.exists() else ORIGINAL_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Neither {RETRAINED_MODEL_PATH} nor {ORIGINAL_MODEL_PATH} exists.")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))

        # Load pair features table
        if not PAIR_FEATURES_PATH.exists():
            raise FileNotFoundError(f"Missing required pair features file: {PAIR_FEATURES_PATH}")

        self.df_features = pd.read_csv(PAIR_FEATURES_PATH)

        # Extract strict 39 ML feature names from model booster
        booster = self.model.get_booster()
        feature_names = booster.feature_names
        if feature_names is None:
            metadata_cols = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]
            feature_names = [c for c in self.df_features.columns if c not in metadata_cols]
        self.feature_names = feature_names

        # Verify no label leakage
        forbidden = ["is_fraud", "ring_id", "propagated_risk", "shap"]
        assert not any(f in self.feature_names for f in forbidden), "Forbidden labels in SHAP feature set!"

        # Initialize TreeExplainer in raw margin (log-odds) space
        self.explainer = shap.TreeExplainer(self.model)

    def explain_transaction(self, tx_id: str) -> Optional[Dict[str, Any]]:
        self.initialize()

        tx_rows = self.df_features[self.df_features["transaction_id"] == tx_id]
        if tx_rows.empty:
            return None

        row = tx_rows.iloc[0]
        X_vec = row[self.feature_names].values.reshape(1, -1).astype(np.float64)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_vec)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # positive fraud class
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        base_val = float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value)

        factors = []
        for feat, shap_val, feat_val in zip(self.feature_names, sv, X_vec[0]):
            factors.append({
                "feature": feat,
                "shap_value": float(round(shap_val, 4)),
                "feature_value": float(round(feat_val, 4))
            })

        increasing = [f for f in factors if f["shap_value"] > 0]
        reducing = [f for f in factors if f["shap_value"] < 0]

        increasing.sort(key=lambda x: x["shap_value"], reverse=True)
        reducing.sort(key=lambda x: x["shap_value"])

        explanation_payload = {
            "transaction_id": tx_id,
            "base_value": round(base_val, 4),
            "risk_increasing_factors": increasing[:5],
            "risk_reducing_factors": reducing[:5],
            "all_factors_count": len(factors)
        }

        # Persist to data/shap_explanations/{tx_id}.json
        SHAP_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SHAP_DIR / f"{tx_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(explanation_payload, f, indent=2)

        return {
            "available": True,
            "risk_increasing_factors": increasing[:5],
            "risk_reducing_factors": reducing[:5]
        }


def load_shap_explanation(tx_id: str) -> Dict[str, Any]:
    """
    Retrieves stored SHAP explanation for tx_id.
    If missing, on-demand computes and persists it via ShapExplainerCache.
    """
    candidate_paths = [
        SHAP_DIR / f"{tx_id}.json",
        SHAP_DIR / f"ALERT_{tx_id}.json",
        SHAP_DIR / f"shap_{tx_id}.json"
    ]

    for p in candidate_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    increasing = [x for x in data if x.get("shap_value", 0.0) > 0]
                    reducing = [x for x in data if x.get("shap_value", 0.0) < 0]
                    increasing.sort(key=lambda x: x.get("shap_value", 0.0), reverse=True)
                    reducing.sort(key=lambda x: x.get("shap_value", 0.0))
                    return {
                        "available": True,
                        "risk_increasing_factors": increasing[:5],
                        "risk_reducing_factors": reducing[:5]
                    }

                elif isinstance(data, dict):
                    if "risk_increasing_factors" in data:
                        return {
                            "available": True,
                            "risk_increasing_factors": data["risk_increasing_factors"][:5],
                            "risk_reducing_factors": data.get("risk_reducing_factors", [])[:5]
                        }
                    elif "feature_attributions" in data:
                        raw_attrs = data["feature_attributions"]
                        inc = [{"feature": k, "shap_value": round(float(v), 4)} for k, v in raw_attrs.items() if v > 0]
                        red = [{"feature": k, "shap_value": round(float(v), 4)} for k, v in raw_attrs.items() if v < 0]
                        inc.sort(key=lambda x: x["shap_value"], reverse=True)
                        red.sort(key=lambda x: x["shap_value"])
                        return {
                            "available": True,
                            "risk_increasing_factors": inc[:5],
                            "risk_reducing_factors": red[:5]
                        }
            except Exception:
                pass

    # Compute on demand if file was not pre-existing
    try:
        cache = ShapExplainerCache.get_instance()
        generated = cache.explain_transaction(tx_id)
        if generated is not None:
            return generated
    except Exception as e:
        print(f"Warning: Could not compute SHAP for {tx_id}: {e}", file=sys.stderr)

    return {
        "available": False,
        "risk_increasing_factors": [],
        "risk_reducing_factors": []
    }


# =====================================================================
# 2. Case File Builder
# =====================================================================

def build_case_file(
    tx_id: str,
    row_risk: pd.Series,
    df_payments: pd.DataFrame,
    df_users: Optional[pd.DataFrame],
    df_rings: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """Constructs a deterministic, complete analyst case file for a single transaction."""
    alert_id = f"ALERT_{tx_id}"

    # Transaction metadata
    tx_match = df_payments[df_payments["transaction_id"] == tx_id]
    tx_row = tx_match.iloc[0] if not tx_match.empty else None

    timestamp = str(tx_row["timestamp"]) if tx_row is not None and "timestamp" in tx_row else str(row_risk.get("timestamp", "UNKNOWN"))
    sender = str(tx_row["sender"]) if tx_row is not None and "sender" in tx_row else str(row_risk.get("sender", "UNKNOWN"))
    receiver = str(tx_row["receiver"]) if tx_row is not None and "receiver" in tx_row else str(row_risk.get("receiver", "UNKNOWN"))
    amount = float(tx_row["amount"]) if tx_row is not None and "amount" in tx_row else float(row_risk.get("amount", 0.0))
    device_id = str(tx_row["device_id"]) if tx_row is not None and "device_id" in tx_row else "NOT_AVAILABLE"
    ip_address = str(tx_row["ip_address"]) if tx_row is not None and "ip_address" in tx_row else "NOT_AVAILABLE"

    # Evaluation / Demo Ground-Truth (strictly isolated)
    demo_ground_truth = None
    if tx_row is not None:
        demo_ground_truth = {
            "is_fraud_planted": int(tx_row.get("is_fraud", 0)),
            "ring_id_planted": str(tx_row.get("ring_id", "NONE")),
            "note": "Ground truth simulator label (evaluation metadata only - was not used for scoring)"
        }

    # Account context from users.csv
    sender_context = {}
    receiver_context = {}
    if df_users is not None:
        s_user = df_users[df_users["user_id"] == sender]
        if not s_user.empty:
            sender_context = {k: v for k, v in s_user.iloc[0].to_dict().items() if k != "user_id"}
        r_user = df_users[df_users["user_id"] == receiver]
        if not r_user.empty:
            receiver_context = {k: v for k, v in r_user.iloc[0].to_dict().items() if k != "user_id"}

    # Model Risk Metrics (exact values from risk_scores_retrained.csv)
    pair_risk = float(row_risk.get("pair_risk", 0.0))
    behavioural_anomaly = float(row_risk.get("behavioural_anomaly", 0.0))
    sender_anom = float(row_risk.get("sender_anomaly_score", 0.0))
    receiver_anom = float(row_risk.get("receiver_anomaly_score", 0.0))
    coordination_score = float(row_risk.get("coordination_score", 0.0))
    sender_coord = float(row_risk.get("sender_coordination_score", 0.0))
    receiver_coord = float(row_risk.get("receiver_coordination_score", 0.0))
    overall_risk = float(row_risk.get("overall_risk", 0.0))
    action = str(row_risk.get("action", "REVIEW"))

    # SHAP Explanations (loaded or generated on-demand)
    shap_info = load_shap_explanation(tx_id)

    # Ring Context from rings.csv
    ring_planted_id = str(tx_row.get("ring_id", "")) if tx_row is not None else ""
    ring_context = {
        "associated_ring_id": None,
        "ring_type": None,
        "members": [],
        "description": "No known fraud syndicate association"
    }

    if df_rings is not None and ring_planted_id and ring_planted_id not in ["NORMAL", "NONE", "nan", ""]:
        ring_match = df_rings[df_rings["ring_id"] == ring_planted_id]
        if not ring_match.empty:
            r_meta = ring_match.iloc[0]
            members_raw = r_meta.get("members", [])
            if isinstance(members_raw, str):
                try:
                    members_parsed = json.loads(members_raw)
                except Exception:
                    members_parsed = [m.strip() for m in members_raw.split(";") if m.strip()]
            else:
                members_parsed = list(members_raw) if isinstance(members_raw, (list, tuple)) else []

            ring_context = {
                "associated_ring_id": ring_planted_id,
                "ring_type": str(r_meta.get("ring_type", r_meta.get("pattern", "syndicate_cluster"))),
                "members": members_parsed,
                "description": f"Known demo syndicate association from simulator ({ring_planted_id})"
            }

    # Human-readable evidence synthesis
    behavioural_signals = []
    if behavioural_anomaly >= 0.50:
        behavioural_signals.append(
            f"Elevated behavioral anomaly (max EIF score: {behavioural_anomaly:.4f}) across counterparties "
            f"(Sender: {sender_anom:.4f}, Receiver: {receiver_anom:.4f})"
        )
    else:
        behavioural_signals.append(f"Baseline behavioral profile (EIF score: {behavioural_anomaly:.4f})")

    network_signals = []
    if coordination_score >= 0.50:
        network_signals.append(
            f"Participating in synchronized lockstep micro-cohort (coordination score: {coordination_score:.4f}, "
            f"Sender: {sender_coord:.4f}, Receiver: {receiver_coord:.4f})"
        )
    else:
        network_signals.append(f"No significant multi-account behavioral synchronization (score: {coordination_score:.4f})")

    return {
        "alert_id": alert_id,
        "transaction_id": tx_id,
        "timestamp": timestamp,
        "transaction": {
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "device_id": device_id,
            "ip_address": ip_address
        },
        "account_context": {
            "sender_profile": sender_context,
            "receiver_profile": receiver_context
        },
        "risk": {
            "pair_risk": round(pair_risk, 4),
            "sender_anomaly_score": round(sender_anom, 4),
            "receiver_anomaly_score": round(receiver_anom, 4),
            "behavioural_anomaly": round(behavioural_anomaly, 4),
            "sender_coordination_score": round(sender_coord, 4),
            "receiver_coordination_score": round(receiver_coord, 4),
            "coordination_score": round(coordination_score, 4),
            "overall_risk": round(overall_risk, 4),
            "action": action,
            "fusion_formula": "0.60 * pair_risk + 0.25 * behavioural_anomaly + 0.15 * coordination_score"
        },
        "evidence": {
            "risk_increasing_factors": shap_info["risk_increasing_factors"],
            "risk_reducing_factors": shap_info["risk_reducing_factors"],
            "behavioural_signals": behavioural_signals,
            "network_signals": network_signals
        },
        "ring_context": ring_context,
        "demo_ground_truth": demo_ground_truth,
        "analyst": {
            "status": "PENDING",
            "available_actions": [
                "CONFIRM_FRAUD",
                "MARK_FALSE_POSITIVE"
            ]
        }
    }


# =====================================================================
# 3. Terminal Renderer
# =====================================================================

def render_case_file_terminal(case: Dict[str, Any]):
    tx = case["transaction"]
    risk = case["risk"]
    evidence = case["evidence"]
    ring = case["ring_context"]
    analyst = case["analyst"]

    print("=======================================================")
    print("           RINGBREAKER FRAUD CASE FILE                 ")
    print("=======================================================")
    print(f"Alert ID:              {case['alert_id']}")
    print(f"Transaction:           {case['transaction_id']}")
    print(f"Timestamp:             {case['timestamp']}")
    print(f"Sender:                {tx['sender']}")
    print(f"Receiver:              {tx['receiver']}")
    print(f"Amount:                ${tx['amount']:,.2f}")
    print(f"Device ID:             {tx['device_id']}")
    print(f"IP Address:            {tx['ip_address']}")

    print("\n-------------------------------------------------------")
    print("RISK ASSESSMENT")
    print("-------------------------------------------------------")
    print(f"Pair Risk:             {risk['pair_risk']:.4f}")
    print(f"Behavioural Anomaly:   {risk['behavioural_anomaly']:.4f}")
    print(f"Coordination Score:    {risk['coordination_score']:.4f}")
    print(f"Overall Risk:          {risk['overall_risk']:.4f}")
    print(f"Action:                {risk['action']}")

    print("\n-------------------------------------------------------")
    print("TOP RISK-INCREASING FACTORS (SHAP)")
    print("-------------------------------------------------------")
    inc_factors = evidence.get("risk_increasing_factors", [])
    if inc_factors:
        for idx, item in enumerate(inc_factors, 1):
            feat = item.get("feature", "Unknown")
            shap_val = item.get("shap_value", 0.0)
            feat_val = item.get("feature_value", None)
            val_str = f" (Value: {feat_val})" if feat_val is not None else ""
            print(f"[{idx}] {feat:<28} | SHAP: +{shap_val:.4f}{val_str}")
    else:
        print("  - Feature SHAP explanations not pre-computed for this transaction.")

    red_factors = evidence.get("risk_reducing_factors", [])
    if red_factors:
        print("\nTOP RISK-REDUCING FACTORS (SHAP):")
        for idx, item in enumerate(red_factors, 1):
            feat = item.get("feature", "Unknown")
            shap_val = item.get("shap_value", 0.0)
            feat_val = item.get("feature_value", None)
            val_str = f" (Value: {feat_val})" if feat_val is not None else ""
            print(f"[-] {feat:<28} | SHAP: {shap_val:.4f}{val_str}")

    print("\n-------------------------------------------------------")
    print("BEHAVIOURAL & NETWORK EVIDENCE")
    print("-------------------------------------------------------")
    for b in evidence.get("behavioural_signals", []):
        print(f"  * {b}")
    for n in evidence.get("network_signals", []):
        print(f"  * {n}")

    print("\n-------------------------------------------------------")
    print("RING CONTEXT (Known / Demo Association)")
    print("-------------------------------------------------------")
    if ring.get("associated_ring_id"):
        print(f"Associated Ring ID:    {ring['associated_ring_id']}")
        print(f"Ring Pattern / Type:   {ring['ring_type']}")
        members = ring.get("members", [])
        print(f"Ring Members:          {', '.join(members) if members else 'None specified'}")
    else:
        print("No prior syndicate or ring cluster recorded.")

    print("\n-------------------------------------------------------")
    print("ANALYST ACTION")
    print("-------------------------------------------------------")
    print(f"Status:                {analyst['status']}")
    print("Available actions:")
    for idx, act in enumerate(analyst["available_actions"], 1):
        print(f"  [{idx}] {act}")
    print("=======================================================\n")


# =====================================================================
# 4. Pipeline Execution
# =====================================================================

def run_case_file_pipeline(
    target_tx: Optional[str] = None,
    limit: Optional[int] = None
):
    if not RISK_SCORES_PATH.exists():
        raise FileNotFoundError(f"Missing required retrained risk scores at {RISK_SCORES_PATH}. Run scoring/risk_engine.py --model retrained first.")

    df_risk = pd.read_csv(RISK_SCORES_PATH)
    df_payments = pd.read_csv(PAYMENTS_PATH) if PAYMENTS_PATH.exists() else pd.DataFrame()
    df_users = pd.read_csv(USERS_PATH) if USERS_PATH.exists() else None
    df_rings = pd.read_csv(RINGS_PATH) if RINGS_PATH.exists() else None

    CASE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Single Transaction Target Mode
    if target_tx:
        match = df_risk[df_risk["transaction_id"] == target_tx]
        if match.empty:
            raise ValueError(f"Transaction ID '{target_tx}' not found in {RISK_SCORES_PATH}")

        case = build_case_file(target_tx, match.iloc[0], df_payments, df_users, df_rings)
        out_file = CASE_FILES_DIR / f"{case['alert_id']}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2)

        render_case_file_terminal(case)
        print(f"Case file saved to: {out_file}")
        return

    # 2. Batch Alert Generation Mode: BLOCK first, then REVIEW
    block_df = df_risk[df_risk["action"] == "BLOCK"].copy()
    review_df = df_risk[df_risk["action"] == "REVIEW"].copy()
    candidate_df = pd.concat([block_df, review_df], ignore_index=True)

    if limit is not None and limit > 0:
        candidate_df = candidate_df.head(limit)

    alerts_summary = []
    generated_count = 0

    print(f"Generating case files for {len(candidate_df)} suspicious alerts (BLOCK/REVIEW)...")

    for _, row in candidate_df.iterrows():
        tx_id = str(row["transaction_id"])
        case = build_case_file(tx_id, row, df_payments, df_users, df_rings)

        out_file = CASE_FILES_DIR / f"{case['alert_id']}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2)

        alerts_summary.append({
            "alert_id": case["alert_id"],
            "transaction_id": case["transaction_id"],
            "timestamp": case["timestamp"],
            "sender": case["transaction"]["sender"],
            "receiver": case["transaction"]["receiver"],
            "amount": case["transaction"]["amount"],
            "overall_risk": case["risk"]["overall_risk"],
            "action": case["risk"]["action"],
            "shap_available": len(case["evidence"]["risk_increasing_factors"]) > 0,
            "case_file_path": str(out_file.relative_to(PROJECT_ROOT)),
            "associated_ring": case["ring_context"]["associated_ring_id"]
        })
        generated_count += 1

    with open(ALERTS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts_summary, f, indent=2)

    print(f"Successfully generated {generated_count} case files in: {CASE_FILES_DIR}")
    print(f"Alerts index saved to: {ALERTS_SUMMARY_PATH}")

    if alerts_summary:
        first_tx = alerts_summary[0]["transaction_id"]
        first_match = df_risk[df_risk["transaction_id"] == first_tx].iloc[0]
        sample_case = build_case_file(first_tx, first_match, df_payments, df_users, df_rings)
        print("\n--- SAMPLE GENERATED ALERT ---")
        render_case_file_terminal(sample_case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Analyst Case File Generator")
    parser.add_argument("--transaction", type=str, default=None, help="Generate / display case file for a specific transaction ID")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of case files to generate in batch mode")
    args = parser.parse_args()

    if args.transaction is None and args.limit is None:
        run_case_file_pipeline(limit=10)
    else:
        run_case_file_pipeline(target_tx=args.transaction, limit=args.limit)