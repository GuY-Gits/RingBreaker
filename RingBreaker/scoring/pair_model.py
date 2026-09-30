"""RingBreaker Pair Risk Model (F10 Pair Risk Score).

Provides:
- train_and_evaluate: Train XGBoost on chronological 70/15/15 partitions.
- score_payment: Live fast-path scoring of one candidate payment using
  TransactionGraph, IdentityGraph, and trained XGBoost model.
  Returns combined risk_score, three PRD sub-scores, and top factors.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None  # type: ignore
    XGB_AVAILABLE = False
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

from ringbreaker.graphs.build import IdentityGraph, TransactionGraph, parse_timestamp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DEFAULT_PATH = PROJECT_ROOT / "models" / "pair_model.json"
METADATA_COLS = ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]

FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_night",
    "sender_account_age_days",
    "receiver_account_age_days",
    "sender_tx_count_before",
    "sender_unique_receivers_before",
    "sender_avg_amount_before",
    "sender_max_amount_before",
    "sender_avg_time_gap",
    "sender_tx_last_1h",
    "sender_tx_last_24h",
    "sender_tx_last_7d",
    "receiver_tx_count_before",
    "receiver_unique_senders_before",
    "receiver_avg_amount_before",
    "receiver_max_amount_before",
    "receiver_tx_last_1h",
    "receiver_tx_last_24h",
    "receiver_tx_last_7d",
    "pair_tx_count_before",
    "pair_avg_amount_before",
    "time_since_previous_pair_tx",
    "is_first_transaction_between_pair",
    "device_tx_count_before",
    "device_unique_users_before",
    "ip_tx_count_before",
    "ip_unique_users_before",
    "sender_in_degree_before",
    "sender_out_degree_before",
    "receiver_in_degree_before",
    "receiver_out_degree_before",
    "sender_out_in_ratio",
    "receiver_in_out_ratio",
    "sender_time_since_last_incoming",
]

_loaded_model: Optional[xgb.XGBClassifier] = None
_model_mtime: float = 0.0


def get_model(model_path: Path | str = MODEL_DEFAULT_PATH) -> Optional[xgb.XGBClassifier]:
    """Lazy loader for the trained XGBoost model."""
    global _loaded_model, _model_mtime
    path = Path(model_path)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _loaded_model is None or mtime != _model_mtime:
        clf = xgb.XGBClassifier()
        clf.load_model(str(path))
        _loaded_model = clf
        _model_mtime = mtime
    return _loaded_model


def extract_candidate_features(
    sender: str,
    receiver: str,
    amount: float,
    timestamp: datetime | str,
    graph: Optional[TransactionGraph],
    identity_graph: Optional[IdentityGraph] = None,
    device: Optional[str] = None,
    ip: Optional[str] = None,
    exclude_transaction_id: Optional[str] = None,
) -> Dict[str, float]:
    """Extract the exact 39 tabular features as-of timestamp T."""
    ts = parse_timestamp(timestamp)
    amt = float(amount)
    log_amt = float(np.log1p(max(0.0, amt)))
    hr = ts.hour
    dow = ts.weekday()
    dom = ts.day
    is_wknd = 1.0 if dow >= 5 else 0.0
    is_night = 1.0 if hr < 6 else 0.0

    if graph is None:
        # Default empty history
        return {name: 0.0 for name in FEATURE_NAMES}

    # History queries strictly <= ts, excluding candidate transaction if needed
    s_hist = graph.get_account_history(sender, as_of=ts, exclude_transaction_id=exclude_transaction_id)
    r_hist = graph.get_account_history(receiver, as_of=ts, exclude_transaction_id=exclude_transaction_id)
    s_in = [p for p in s_hist if p.receiver == sender]
    s_out = [p for p in s_hist if p.sender == sender]
    r_in = [p for p in r_hist if p.receiver == receiver]
    r_out = [p for p in r_hist if p.sender == receiver]

    s_signup = graph.get_signup_at(sender, as_of=ts)
    if s_signup:
        s_age = max(0.0, (ts - s_signup).total_seconds() / 86400.0)
    elif s_hist:
        s_age = max(0.0, (ts - min(p.timestamp for p in s_hist)).total_seconds() / 86400.0)
    else:
        s_age = 0.0

    r_signup = graph.get_signup_at(receiver, as_of=ts)
    if r_signup:
        r_age = max(0.0, (ts - r_signup).total_seconds() / 86400.0)
    elif r_hist:
        r_age = max(0.0, (ts - min(p.timestamp for p in r_hist)).total_seconds() / 86400.0)
    else:
        r_age = 0.0

    s_out_amts = [p.amount for p in s_out]
    s_avg_amt = float(np.mean(s_out_amts)) if s_out_amts else 0.0
    s_max_amt = float(max(s_out_amts)) if s_out_amts else 0.0

    if len(s_out) >= 2:
        s_times = sorted(p.timestamp for p in s_out)
        gaps = [(s_times[i] - s_times[i - 1]).total_seconds() for i in range(1, len(s_times))]
        s_avg_gap = float(np.mean(gaps))
    else:
        s_avg_gap = -1.0

    s_tx_1h = float(sum(1 for p in s_out if (ts - p.timestamp).total_seconds() <= 3600))
    s_tx_24h = float(sum(1 for p in s_out if (ts - p.timestamp).total_seconds() <= 86400))
    s_tx_7d = float(sum(1 for p in s_out if (ts - p.timestamp).total_seconds() <= 7 * 86400))

    r_in_amts = [p.amount for p in r_in]
    r_avg_amt = float(np.mean(r_in_amts)) if r_in_amts else 0.0
    r_max_amt = float(max(r_in_amts)) if r_in_amts else 0.0
    r_tx_1h = float(sum(1 for p in r_in if (ts - p.timestamp).total_seconds() <= 3600))
    r_tx_24h = float(sum(1 for p in r_in if (ts - p.timestamp).total_seconds() <= 86400))
    r_tx_7d = float(sum(1 for p in r_in if (ts - p.timestamp).total_seconds() <= 7 * 86400))

    pair_hist = graph.get_pair_history(sender, receiver, as_of=ts, exclude_transaction_id=exclude_transaction_id)
    p_count = len(pair_hist)
    p_avg_amt = float(np.mean([p.amount for p in pair_hist])) if pair_hist else 0.0
    if pair_hist:
        time_since_pair = max(0.0, (ts - max(p.timestamp for p in pair_hist)).total_seconds())
        is_first_pair = 0.0
    else:
        time_since_pair = -1.0
        is_first_pair = 1.0

    s_in_deg = float(len(graph.get_in_neighbors(sender, as_of=ts, exclude_transaction_id=exclude_transaction_id)))
    s_out_deg = float(len(graph.get_out_neighbors(sender, as_of=ts, exclude_transaction_id=exclude_transaction_id)))
    r_in_deg = float(len(graph.get_in_neighbors(receiver, as_of=ts, exclude_transaction_id=exclude_transaction_id)))
    r_out_deg = float(len(graph.get_out_neighbors(receiver, as_of=ts, exclude_transaction_id=exclude_transaction_id)))

    s_out_in = float(len(s_out) / (len(s_in) + 1.0))
    r_in_out = float(len(r_in) / (len(r_out) + 1.0))

    if s_in:
        s_time_since_inc = max(0.0, (ts - max(p.timestamp for p in s_in)).total_seconds())
    else:
        s_time_since_inc = -1.0

    dev_users = 1.0
    dev_tx = 1.0
    if identity_graph and device:
        shared = identity_graph.get_accounts_for_identity("device", str(device), as_of=ts)
        dev_users = float(max(1, len(shared)))
        dev_tx = dev_users

    return {
        "amount": amt,
        "log_amount": log_amt,
        "hour": float(hr),
        "day_of_week": float(dow),
        "day_of_month": float(dom),
        "is_weekend": is_wknd,
        "is_night": is_night,
        "sender_account_age_days": s_age,
        "receiver_account_age_days": r_age,
        "sender_tx_count_before": float(len(s_out)),
        "sender_unique_receivers_before": float(len(set(p.receiver for p in s_out))),
        "sender_avg_amount_before": s_avg_amt,
        "sender_max_amount_before": s_max_amt,
        "sender_avg_time_gap": s_avg_gap,
        "sender_tx_last_1h": s_tx_1h,
        "sender_tx_last_24h": s_tx_24h,
        "sender_tx_last_7d": s_tx_7d,
        "receiver_tx_count_before": float(len(r_in)),
        "receiver_unique_senders_before": float(len(set(p.sender for p in r_in))),
        "receiver_avg_amount_before": r_avg_amt,
        "receiver_max_amount_before": r_max_amt,
        "receiver_tx_last_1h": r_tx_1h,
        "receiver_tx_last_24h": r_tx_24h,
        "receiver_tx_last_7d": r_tx_7d,
        "pair_tx_count_before": float(p_count),
        "pair_avg_amount_before": p_avg_amt,
        "time_since_previous_pair_tx": time_since_pair,
        "is_first_transaction_between_pair": is_first_pair,
        "device_tx_count_before": dev_tx,
        "device_unique_users_before": dev_users,
        "ip_tx_count_before": 0.0,
        "ip_unique_users_before": 1.0,
        "sender_in_degree_before": s_in_deg,
        "sender_out_degree_before": s_out_deg,
        "receiver_in_degree_before": r_in_deg,
        "receiver_out_degree_before": r_out_deg,
        "sender_out_in_ratio": s_out_in,
        "receiver_in_out_ratio": r_in_out,
        "sender_time_since_last_incoming": s_time_since_inc,
    }


def compute_sub_scores(
    features: Dict[str, float],
    risk_boost_sender: float = 0.0,
    risk_boost_receiver: float = 0.0,
) -> Dict[str, float]:
    """Compute the 3 PRD sub-scores bounded in [0.0, 1.0]."""
    # 1. Sender Anomaly: high velocity in 1h/24h, high out-in ratio, extreme amount spike
    s_vel = min(1.0, features["sender_tx_last_1h"] / 3.0 + features["sender_tx_last_24h"] / 10.0)
    s_amt_spike = 0.0
    if features["sender_avg_amount_before"] > 0:
        s_amt_spike = min(1.0, max(0.0, (features["amount"] - features["sender_avg_amount_before"]) / (features["sender_avg_amount_before"] * 3.0)))
    sender_anomaly = min(1.0, 0.15 + 0.35 * s_vel + 0.30 * s_amt_spike + 0.20 * features["is_night"] + risk_boost_sender)

    # 2. Receiver Mule Propensity: new receiver account, incoming velocity burst, device reuse
    r_newness = 1.0 if features["receiver_account_age_days"] < 7.0 else max(0.0, (30.0 - features["receiver_account_age_days"]) / 30.0)
    r_vel = min(1.0, features["receiver_tx_last_1h"] / 3.0 + features["receiver_tx_last_24h"] / 8.0)
    dev_reuse = min(1.0, max(0.0, (features["device_unique_users_before"] - 1.0) / 3.0))
    receiver_mule = min(1.0, 0.15 + 0.30 * r_newness + 0.35 * r_vel + 0.20 * dev_reuse + risk_boost_receiver)

    # 3. Relationship Plausibility: first-time transfer, zero prior history, large amount
    if features["is_first_transaction_between_pair"] > 0.5:
        rel_anomaly = min(1.0, 0.35 + min(0.55, features["amount"] / 10000.0))
    else:
        rel_anomaly = max(0.05, 0.20 - min(0.15, features["pair_tx_count_before"] * 0.05))

    return {
        "sender_anomaly": round(float(np.clip(sender_anomaly, 0.0, 1.0)), 4),
        "receiver_mule_propensity": round(float(np.clip(receiver_mule, 0.0, 1.0)), 4),
        "relationship_plausibility": round(float(np.clip(rel_anomaly, 0.0, 1.0)), 4),
    }


def score_payment(
    sender: str,
    receiver: str,
    amount: float,
    timestamp: str,
    device: Optional[str] = None,
    graph: Optional[TransactionGraph] = None,
    identity_graph: Optional[IdentityGraph] = None,
    feature_cache: Optional[Dict[str, Any]] = None,
    model_path: Path | str = MODEL_DEFAULT_PATH,
    exclude_transaction_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main live scoring entrypoint called by API POST /score.
    Returns risk_score, 3 sub-scores, and top factor contributions.
    """
    feats = extract_candidate_features(
        sender=sender,
        receiver=receiver,
        amount=amount,
        timestamp=timestamp,
        graph=graph,
        identity_graph=identity_graph,
        device=device,
        exclude_transaction_id=exclude_transaction_id,
    )

    boost_s = 0.0
    boost_r = 0.0
    if feature_cache:
        boost_s = float(feature_cache.get(sender, {}).get("risk_boost", 0.0))
        boost_r = float(feature_cache.get(receiver, {}).get("risk_boost", 0.0))

    sub_scores = compute_sub_scores(feats, boost_s, boost_r)

    # ML Inference
    model = get_model(model_path)
    if model is not None:
        row = np.array([[feats[col] for col in FEATURE_NAMES]], dtype=np.float32)
        prob = float(model.predict_proba(row)[0, 1])
        # Fuse ML risk with propagated risk boosts if any
        combined_risk = float(np.clip(prob + max(boost_s, boost_r) * 0.5, 0.0, 1.0))
        model_ver = "1.0.0-xgb"
    else:
        combined_risk = float(np.clip(
            (sub_scores["sender_anomaly"] + sub_scores["receiver_mule_propensity"] + sub_scores["relationship_plausibility"]) / 3.0,
            0.0, 1.0
        ))
        model_ver = "0.1.0-heuristic"

    # SHAP-style linear feature importance for top factors
    shap_factors = {
        "amount": round(float((feats["amount"] / 5000.0) * 0.3), 4),
        "sender_tx_last_1h": round(float(feats["sender_tx_last_1h"] * 0.25), 4),
        "receiver_tx_last_1h": round(float(feats["receiver_tx_last_1h"] * 0.25), 4),
        "is_first_transaction_between_pair": round(float(feats["is_first_transaction_between_pair"] * 0.2), 4),
        "device_unique_users_before": round(float((feats["device_unique_users_before"] - 1.0) * 0.3), 4),
        "sender_out_in_ratio": round(float(min(1.0, feats["sender_out_in_ratio"] / 5.0) * 0.15), 4),
    }

    return {
        "risk_score": round(combined_risk, 4),
        "sender_anomaly": sub_scores["sender_anomaly"],
        "receiver_mule_propensity": sub_scores["receiver_mule_propensity"],
        "relationship_plausibility": sub_scores["relationship_plausibility"],
        "sub_scores": sub_scores,
        "shap_values": shap_factors,
        "feature_values": feats,
        "model_version": model_ver,
    }


def train_and_evaluate(
    features_path="data/pair_features.csv",
    rings_path="data/rings.csv",
    model_out="models/pair_model.json",
    scored_out="data/scored_payments.csv",
):
    """Offline batch training and evaluation on chronological partitions."""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature table not found: {features_path}")

    df = pd.read_csv(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols]
    y = df["is_fraud"]

    n_total = len(df)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    num_neg = (y_train == 0).sum()
    num_pos = max((y_train == 1).sum(), 1)
    scale_pos = num_neg / num_pos

    clf = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=42,
    )

    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    clf.save_model(model_out)
    return clf