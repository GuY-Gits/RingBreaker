"""F15: lockstep / sleeper-batch clustering.

DBSCAN on a small as-of feature vector per account (PRD: "cluster accounts on
signup time and activity rhythm"):

- signup time (hours since epoch)
- circular mean hour-of-day of activity (cos, sin)
- circular dispersion of activity hour (0 = always at the same time of day)
- log median gap between the account's payments (activity rhythm)

DBSCAN groups accounts that look alike, which on its own also groups ordinary
users. Each cluster is therefore scored on three interpretable lockstep
signals, averaged into ``lockstep_score``:

- signup compactness: 1 - signup span / 14 days
- rhythm tightness:  1 - mean hour dispersion / 1.0
- internal activity: share of members' outgoing payments that go to members

A cluster is ``suspicious`` when that score is >= 0.6 and it has 3-60 members.
Clusters are detection signals, not fraud labels. Only accounts with at least
one payment visible at ``as_of`` are clustered.
Also provides batch CSV pipeline execution for offline evaluation.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ringbreaker.graphs.build import TransactionGraph, parse_timestamp

DEFAULT_EPS = 0.3
DEFAULT_MIN_SAMPLES = 4
SUSPICIOUS_SCORE = 0.6
MAX_SUSPICIOUS_SIZE = 60
FEATURE_NAMES = [
    "signup_hours",
    "hour_cos",
    "hour_sin",
    "hour_dispersion",
    "log_median_gap_hours",
]


def account_lockstep_vector(
    graph: TransactionGraph,
    account_id: str,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
) -> dict[str, float] | None:
    as_of_ts = parse_timestamp(as_of)
    account_id = str(account_id)
    history = graph.get_account_history(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    if not history:
        return None
    signup = graph.get_signup_at(account_id, as_of=as_of_ts)
    if signup is None:
        signup = min(p.timestamp for p in history)
    angles = [2 * math.pi * (p.timestamp.hour + p.timestamp.minute / 60.0) / 24.0 for p in history]
    c = float(np.mean(np.cos(angles)))
    s = float(np.mean(np.sin(angles)))
    r = math.hypot(c, s)
    dispersion = math.sqrt(-2.0 * math.log(max(r, 1e-9)))
    times = sorted(p.timestamp.timestamp() for p in history)
    gaps = np.diff(times) / 3600.0
    median_gap = float(np.median(gaps)) if len(gaps) else 0.0
    return {
        "signup_hours": signup.timestamp() / 3600.0,
        "hour_cos": c,
        "hour_sin": s,
        "hour_dispersion": min(dispersion, 3.0),
        "log_median_gap_hours": math.log1p(median_gap),
    }


def detect_lockstep(
    graph: TransactionGraph,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    account_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cluster accounts; return one dict per cluster with size >= min_samples."""
    ids = account_ids if account_ids is not None else graph.account_ids(as_of=as_of)
    rows: list[tuple[str, dict[str, float]]] = []
    for account_id in ids:
        vector = account_lockstep_vector(
            graph, account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
        if vector is not None:
            rows.append((account_id, vector))
    if len(rows) < min_samples:
        return []
    matrix = np.array([[row[1][name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    scaled = StandardScaler().fit_transform(matrix)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label < 0:
            continue
        clusters.setdefault(int(label), []).append(idx)
    results: list[dict[str, Any]] = []
    for cluster_id, indices in sorted(clusters.items()):
        members = [rows[i][0] for i in indices]
        member_set = set(members)
        signup_hours = matrix[indices, 0]
        span_days = float(signup_hours.max() - signup_hours.min()) / 24.0
        dispersion = float(matrix[indices, 3].mean())
        total_out = internal_out = 0
        for m in members:
            for p in graph.outgoing_payments(m, as_of=as_of, exclude_transaction_id=exclude_transaction_id):
                total_out += 1
                internal_out += p.receiver in member_set
        internal_ratio = internal_out / total_out if total_out else 0.0
        compactness = max(0.0, 1.0 - span_days / 14.0)
        rhythm = max(0.0, 1.0 - dispersion / 1.0)
        score = float((compactness + rhythm + internal_ratio) / 3.0)
        member_features = {rows[i][0]: rows[i][1] for i in indices}
        results.append(
            {
                "cluster_id": cluster_id,
                "members": members,
                "cluster_size": len(members),
                "lockstep_score": score,
                "suspicious": bool(
                    score >= SUSPICIOUS_SCORE and 3 <= len(members) <= MAX_SUSPICIOUS_SIZE
                ),
                "feature_names": FEATURE_NAMES,
                "member_features": member_features,
                "evidence": {
                    "eps": eps,
                    "min_samples": min_samples,
                    "algorithm": "DBSCAN",
                    "signup_span_days": span_days,
                    "mean_hour_dispersion": dispersion,
                    "internal_payment_ratio": internal_ratio,
                    "internal_payments": internal_out,
                    "outgoing_payments": total_out,
                    "signup_compactness": compactness,
                    "rhythm_tightness": rhythm,
                },
            }
        )
    return results


def account_lockstep_features(
    graph: TransactionGraph,
    account_id: str,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, Any]:
    """Single-account lockstep membership features for Person 1 scoring models."""
    account_id = str(account_id)
    clusters = detect_lockstep(
        graph,
        as_of=as_of,
        exclude_transaction_id=exclude_transaction_id,
        eps=eps,
        min_samples=min_samples,
    )
    for cluster in clusters:
        if account_id in cluster["members"]:
            return {
                "in_lockstep_cluster": 1,
                "lockstep_cluster_size": cluster["cluster_size"],
                "lockstep_cluster_score": cluster["lockstep_score"],
                "lockstep_cluster_id": cluster["cluster_id"],
            }
    return {
        "in_lockstep_cluster": 0,
        "lockstep_cluster_size": 0,
        "lockstep_cluster_score": 0.0,
        "lockstep_cluster_id": -1,
    }


# =====================================================================
# Offline Batch CSV Execution
# =====================================================================

BEHAVIORAL_FEATURES = [
    "tx_rate_per_day",
    "interarrival_log_sec",
    "burstiness",
    "hour_entropy",
    "sin_peak_hour",
    "cos_peak_hour",
    "weekend_ratio",
    "signup_offset_log",
]


def perform_leakage_audit(feature_df: pd.DataFrame) -> bool:
    """Verifies that no ground-truth fraud or ring labels contaminate the feature set."""
    forbidden = ["is_fraud", "ring_id", "fraud", "label", "is_mule"]
    found_violations = [c for c in feature_df.columns if any(f in c.lower() for f in forbidden)]
    is_fraud_present = "is_fraud" in feature_df.columns
    ring_id_present = "ring_id" in feature_df.columns
    if found_violations or is_fraud_present or ring_id_present:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {found_violations}")
    return True