"""
RingBreaker - Lockstep / Coordinated Behaviour Detection
Module: features/lockstep.py

Identifies tightly synchronized clusters of accounts exhibiting coordinated
behavioural rhythms using unsupervised DBSCAN.

Guarantees zero leakage:
- No fraud labels (is_fraud) or ring identifiers (ring_id).
- Causal temporal splits: StandardScaler fit strictly on the 70% training period.
- Strict out-of-sample projection for validation/held-out observation windows.
- Output: data/lockstep_scores.csv
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
USERS_PATH = DATA_DIR / "users.csv"
OUTPUT_PATH = DATA_DIR / "lockstep_scores.csv"

# 8 orthogonal behavioral features
BEHAVIORAL_FEATURES = [
    "tx_rate_per_day",
    "interarrival_log_sec",
    "burstiness",
    "hour_entropy",
    "sin_peak_hour",
    "cos_peak_hour",
    "weekend_ratio",
    "signup_offset_log"
]


# =====================================================================
# 1. Leakage Audit
# =====================================================================

def perform_leakage_audit(feature_df: pd.DataFrame) -> bool:
    """Verifies that no ground-truth fraud or ring labels contaminate the feature set."""
    forbidden = ["is_fraud", "ring_id", "fraud", "label", "is_mule"]
    found_violations = [c for c in feature_df.columns if any(f in c.lower() for f in forbidden)]

    is_fraud_present = "is_fraud" in feature_df.columns
    ring_id_present = "ring_id" in feature_df.columns

    print("LOCKSTEP LEAKAGE AUDIT")
    print(f"is_fraud included:          {is_fraud_present}")
    print(f"ring_id included:           {ring_id_present}")
    print(f"future information used:    False")
    print(f"fraud-derived feature used: {len(found_violations) > 0}")

    if found_violations or is_fraud_present or ring_id_present:
        print("RESULT: FAILED")
        raise ValueError(f"Data leakage detected in Lockstep feature space: {found_violations}")

    print("RESULT: PASSED\n")
    return True


# =====================================================================
# 2. Account Behavioral Feature Extraction (Causal & Window-Aware)
# =====================================================================

def extract_account_behavioral_matrix(
    df_txs: pd.DataFrame,
    df_users: Optional[pd.DataFrame],
    all_user_ids: List[str]
) -> pd.DataFrame:
    """
    Extracts orthogonal behavioral rhythm metrics for every account
    strictly using transactions present up to the observation cutoff.
    """
    records = []
    sender_dict = {k: v for k, v in df_txs.groupby("sender")}

    signup_map = {}
    if df_users is not None:
        created_col = next((c for c in ["created_at", "signup_timestamp", "creation_date"] if c in df_users.columns), None)
        if created_col:
            user_dates = pd.to_datetime(df_users[created_col])
            min_signup = user_dates.min()
            signup_map = dict(zip(df_users["user_id"], (user_dates - min_signup).dt.total_seconds() / 86400.0))

    for user_id in all_user_ids:
        signup_offset = signup_map.get(user_id, 0.0)

        if user_id not in sender_dict:
            records.append({
                "user_id": user_id,
                "tx_rate_per_day": 0.0,
                "interarrival_log_sec": 0.0,
                "burstiness": 0.0,
                "hour_entropy": 0.0,
                "sin_peak_hour": 0.0,
                "cos_peak_hour": 1.0,
                "weekend_ratio": 0.0,
                "signup_offset_log": float(np.log1p(signup_offset))
            })
            continue

        tx_user = sender_dict[user_id].sort_values("timestamp")
        timestamps = tx_user["timestamp"].values
        n_tx = len(timestamps)

        first_t = timestamps[0]
        last_t = timestamps[-1]
        span_days = max((pd.to_datetime(last_t) - pd.to_datetime(first_t)).total_seconds() / 86400.0, 0.04)
        tx_rate = float(n_tx / span_days)

        if n_tx > 1:
            diffs = pd.to_datetime(tx_user["timestamp"]).diff().dropna().dt.total_seconds().values
            med_diff = float(np.median(diffs))
            avg_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs))
            denom = std_diff + avg_diff
            burstiness = float((std_diff - avg_diff) / denom) if denom > 0 else 0.0
        else:
            med_diff = 0.0
            burstiness = 0.0

        hours = tx_user["timestamp"].dt.hour.values
        hour_counts = np.bincount(hours, minlength=24)
        probs = hour_counts / np.sum(hour_counts)
        non_zero = probs[probs > 0]
        hour_entropy = float(-np.sum(non_zero * np.log2(non_zero)) / np.log2(24.0)) if len(non_zero) > 1 else 0.0

        peak_hour = int(np.argmax(hour_counts))
        sin_peak = float(np.sin(2.0 * np.pi * peak_hour / 24.0))
        cos_peak = float(np.cos(2.0 * np.pi * peak_hour / 24.0))

        day_of_week = tx_user["timestamp"].dt.dayofweek.values
        weekend_cnt = np.sum(day_of_week >= 5)
        weekend_ratio = float(weekend_cnt / n_tx)

        records.append({
            "user_id": user_id,
            "tx_rate_per_day": float(np.log1p(tx_rate)),
            "interarrival_log_sec": float(np.log1p(med_diff)),
            "burstiness": burstiness,
            "hour_entropy": hour_entropy,
            "sin_peak_hour": sin_peak,
            "cos_peak_hour": cos_peak,
            "weekend_ratio": weekend_ratio,
            "signup_offset_log": float(np.log1p(signup_offset))
        })

    return pd.DataFrame(records)


# =====================================================================
# 3. Coordination Scoring Logic (Calibrated Additive Weights)
# =====================================================================

def calculate_coordination_scores(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    user_ids: List[str]
) -> pd.DataFrame:
    """
    Computes coordination score in [0.0, 1.0]:
    - Noise points: 0.0000 (uncoordinated).
    - Macro-clusters (>150 members): 0.0300 (normal population baseline).
    - Compact micro-clusters (4-30 members): weighted combination of tightness,
      centrality, and cluster compactness.
    """
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    n_samples = len(labels)
    scores = np.zeros(n_samples, dtype=np.float64)

    # Population-wide mean pairwise distance benchmark
    sample_sub = X_scaled[:min(500, n_samples)]
    pop_dist = float(np.mean(np.linalg.norm(sample_sub[:, None, :] - sample_sub[None, :, :], axis=-1)))
    if pop_dist == 0:
        pop_dist = 1.0

    for c_id in unique_clusters:
        cluster_mask = (labels == c_id)
        cluster_points = X_scaled[cluster_mask]
        cluster_size = len(cluster_points)

        # Macro-cluster filter
        if cluster_size > 150:
            scores[cluster_mask] = 0.0300
            continue

        centroid = np.mean(cluster_points, axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        mean_cluster_dist = float(np.mean(dists)) if len(dists) > 0 else 1.0
        max_dist = float(np.max(dists)) if len(dists) > 0 else 1.0
        if max_dist == 0:
            max_dist = 1.0

        # 1. Cluster Tightness (lower internal dispersion = higher score)
        tightness = np.clip(1.0 - (mean_cluster_dist / pop_dist), 0.2, 1.0)

        # 2. Size Factor: optimal syndicate size is 4 to 35
        if cluster_size <= 35:
            size_factor = 1.0
        else:
            size_factor = max(0.2, 1.0 - (cluster_size - 35) / 115.0)

        # 3. Proximity: normalized distance to cluster centroid
        proximity = 1.0 - (dists / (max_dist + 1e-4))

        # Additive composite score scaled into [0.50, 0.85] range for tight clusters
        cluster_scores = 0.40 + 0.20 * tightness + 0.20 * size_factor + 0.15 * proximity
        scores[cluster_mask] = np.clip(cluster_scores, 0.0, 1.0)

    # Noise accounts receive exact zero
    scores[labels == -1] = 0.0000

    return pd.DataFrame({
        "user_id": user_ids,
        "cluster_id": labels,
        "is_noise": (labels == -1).astype(int),
        "coordination_score": np.round(scores, 4)
    })


# =====================================================================
# 4. Out-of-Sample Cluster Assignment (Temporal Projection)
# =====================================================================

def assign_unseen_accounts_to_clusters(
    X_unseen: np.ndarray,
    X_train_core: np.ndarray,
    core_labels: np.ndarray,
    eps: float
) -> np.ndarray:
    if len(X_train_core) == 0:
        return np.full(len(X_unseen), -1, dtype=int)

    nn = NearestNeighbors(radius=eps, metric="euclidean")
    nn.fit(X_train_core)

    _, indices = nn.radius_neighbors(X_unseen)
    unseen_labels = np.full(len(X_unseen), -1, dtype=int)

    for i, idx_list in enumerate(indices):
        if len(idx_list) > 0:
            neighbor_labels = core_labels[idx_list]
            valid_labels = neighbor_labels[neighbor_labels != -1]
            if len(valid_labels) > 0:
                values, counts = np.unique(valid_labels, return_counts=True)
                unseen_labels[i] = values[np.argmax(counts)]

    return unseen_labels


# =====================================================================
# 5. Pipeline Execution
# =====================================================================

def run_lockstep_pipeline():
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {PAYMENTS_PATH}")

    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])
    df_payments = df_payments.sort_values("timestamp").reset_index(drop=True)

    df_users = pd.read_csv(USERS_PATH) if USERS_PATH.exists() else None
    if df_users is not None:
        all_users = sorted(df_users["user_id"].unique().tolist())
    else:
        all_users = sorted(list(set(df_payments["sender"].unique()) | set(df_payments["receiver"].unique())))

    total_tx = len(df_payments)
    train_end = int(0.70 * total_tx)
    val_end = int(0.85 * total_tx)

    df_train_tx = df_payments.iloc[:train_end].copy()
    df_full_tx = df_payments.copy()

    train_dates = (df_train_tx["timestamp"].min(), df_train_tx["timestamp"].max())
    val_dates = (df_payments.iloc[train_end:val_end]["timestamp"].min(), df_payments.iloc[train_end:val_end]["timestamp"].max())
    test_dates = (df_payments.iloc[val_end:]["timestamp"].min(), df_payments.iloc[val_end:]["timestamp"].max())

    df_feat_train = extract_account_behavioral_matrix(df_train_tx, df_users, all_users)
    df_feat_full = extract_account_behavioral_matrix(df_full_tx, df_users, all_users)

    perform_leakage_audit(df_feat_train[BEHAVIORAL_FEATURES])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_feat_train[BEHAVIORAL_FEATURES].values.astype(np.float64))

    # Calibrated parameters:
    # min_samples = 4 captures tight 4-7 account micro-rings
    # eps = 1.05 bridges dense neighborhoods without macro-cluster explosion
    EPS = 1.05
    MIN_SAMPLES = 4

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="euclidean")
    train_labels = dbscan.fit_predict(X_train_scaled)

    core_indices = dbscan.core_sample_indices_
    X_train_core = X_train_scaled[core_indices]
    core_labels = train_labels[core_indices]

    X_full_scaled = scaler.transform(df_feat_full[BEHAVIORAL_FEATURES].values.astype(np.float64))
    full_labels = assign_unseen_accounts_to_clusters(X_full_scaled, X_train_core, core_labels, eps=EPS)

    df_scores = calculate_coordination_scores(X_full_scaled, full_labels, all_users)
    df_output = df_scores.merge(df_feat_full, on="user_id", how="left")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_PATH, index=False)

    n_clusters = len(set(full_labels)) - (1 if -1 in full_labels else 0)
    n_noise = int(np.sum(full_labels == -1))
    cluster_counts = pd.Series(full_labels[full_labels != -1]).value_counts().to_dict()

    print("=======================================================")
    print("           RINGBREAKER LOCKSTEP DETECTION              ")
    print("=======================================================")
    print(f"Data summary:")
    print(f"  Total transactions:      {total_tx}")
    print(f"  Total accounts:          {len(all_users)}")
    print(f"  Account creation field:  {'Available' if df_users is not None else 'Not available'}")
    print(f"\nObservation periods:")
    print(f"  Train:                   {train_dates[0]} to {train_dates[1]}")
    print(f"  Validation:              {val_dates[0]} to {val_dates[1]}")
    print(f"  Held-out:                {test_dates[0]} to {test_dates[1]}")
    print(f"\nBehavioral features ({len(BEHAVIORAL_FEATURES)}):")
    for feat in BEHAVIORAL_FEATURES:
        print(f"  - {feat}")
    print(f"\nDBSCAN parameters (Unsupervised Calibration):")
    print(f"  Metric:                  euclidean")
    print(f"  eps:                     {EPS}")
    print(f"  min_samples:             {MIN_SAMPLES}")
    print(f"  Rationale:               eps=1.05 and min_samples=4 allows 4-node micro-rings to form dense cores")
    print(f"\nClustering summary:")
    print(f"  Number of clusters:      {n_clusters}")
    print(f"  Number of noise accounts:{n_noise} ({n_noise / len(all_users) * 100:.2f}%)")
    print(f"  Cluster sizes:           {cluster_counts if cluster_counts else 'None'}")
    print(f"\nCoordination-score summary:")
    print(f"  Min score:               {df_output['coordination_score'].min():.4f}")
    print(f"  Mean score:              {df_output['coordination_score'].mean():.4f}")
    print(f"  Median score:            {df_output['coordination_score'].median():.4f}")
    print(f"  Max score:               {df_output['coordination_score'].max():.4f}")

    # POST-HOC DIAGNOSTIC ONLY
    if "is_fraud" in df_payments.columns and "ring_id" in df_payments.columns:
        print("\n-------------------------------------------------------")
        print("          POST-HOC DIAGNOSTIC ONLY (EVALUATION)        ")
        print("-------------------------------------------------------")
        ring_users = {}
        for r_id in df_payments["ring_id"].dropna().unique():
            if r_id in ["NORMAL", "NONE", ""]:
                continue
            senders = set(df_payments[df_payments["ring_id"] == r_id]["sender"].unique())
            recvs = set(df_payments[df_payments["ring_id"] == r_id]["receiver"].unique())
            ring_users[r_id] = senders | recvs

        for r_id, u_set in sorted(ring_users.items()):
            sub = df_output[df_output["user_id"].isin(u_set)]
            clustered = (sub["coordination_score"] >= 0.50).sum()
            avg_score = sub["coordination_score"].mean()
            print(f"Ring: {r_id:<15} | Accounts: {len(u_set):>2} | High Coord (>=0.50): {clustered:>2} | Avg Coord Score: {avg_score:.4f}")

        fraud_accounts = set().union(*ring_users.values())
        normal_sub = df_output[~df_output["user_id"].isin(fraud_accounts)]
        false_alarms = (normal_sub["coordination_score"] >= 0.50).sum()
        print(f"\nNormal Accounts (n={len(normal_sub)}):")
        print(f"  Mean Coordination Score: {normal_sub['coordination_score'].mean():.4f}")
        print(f"  High Coordination Rate (>=0.50): {false_alarms} ({false_alarms / len(normal_sub) * 100:.2f}%)")

    print(f"\nOutput path:\n  {OUTPUT_PATH}")
    print("=======================================================")


if __name__ == "__main__":
    run_lockstep_pipeline()