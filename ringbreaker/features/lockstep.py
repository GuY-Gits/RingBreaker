"""F15: lockstep / sleeper-batch clustering.

DBSCAN on a small as-of feature vector per account:

- signup time (hours since epoch)
- activity rhythm (mean inter-event hours, std of inter-event hours)
- dormancy (hours since last payment as of T)
- mean hour-of-day of activity

Clusters are detection signals, not fraud labels. Noise points (DBSCAN -1) are
omitted. Accounts with no signup and no history are skipped.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ringbreaker.graphs.build import TransactionGraph, parse_timestamp

DEFAULT_EPS = 0.8
DEFAULT_MIN_SAMPLES = 2


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
    signup = graph.get_signup_at(account_id, as_of=as_of_ts)
    if signup is None and history:
        signup = min(p.timestamp for p in history)
    if signup is None:
        return None
    times = sorted(p.timestamp for p in history)
    if times:
        gaps = [
            (times[i] - times[i - 1]).total_seconds() / 3600.0
            for i in range(1, len(times))
        ]
        last = times[-1]
    else:
        gaps = []
        last = signup
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    std_gap = float(np.std(gaps, ddof=1)) if len(gaps) >= 2 else 0.0
    dormancy = max(0.0, (as_of_ts - last).total_seconds() / 3600.0)
    hours = [ts.hour + ts.minute / 60.0 for ts in times]
    mean_hour = float(np.mean(hours)) if hours else 0.0
    return {
        "signup_hours": signup.timestamp() / 3600.0,
        "mean_inter_event_hours": mean_gap,
        "std_inter_event_hours": std_gap,
        "dormancy_hours": dormancy,
        "mean_hour_of_day": mean_hour,
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
    feature_names = [
        "signup_hours",
        "mean_inter_event_hours",
        "std_inter_event_hours",
        "dormancy_hours",
        "mean_hour_of_day",
    ]
    matrix = np.array([[row[1][name] for name in feature_names] for row in rows], dtype=float)
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
        subset = scaled[indices]
        centroid = subset.mean(axis=0)
        tightness = float(np.mean(np.linalg.norm(subset - centroid, axis=1)))
        score = float(1.0 / (1.0 + tightness))
        member_features = {rows[i][0]: rows[i][1] for i in indices}
        results.append(
            {
                "cluster_id": cluster_id,
                "members": members,
                "cluster_size": len(members),
                "lockstep_score": score,
                "feature_names": feature_names,
                "member_features": member_features,
                "evidence": {
                    "eps": eps,
                    "min_samples": min_samples,
                    "mean_scaled_distance_to_centroid": tightness,
                    "algorithm": "DBSCAN",
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
