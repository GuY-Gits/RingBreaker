"""F15: lockstep / sleeper-batch clustering.

Candidates come from DBSCAN on a small as-of feature vector per account, plus
signup cohorts (accounts created within 7 days of each other that pay each
other — found even when camouflage payments hide them from DBSCAN) (PRD: "cluster accounts on
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
- identity share (when an identity graph is given): share of members that
  share a device, phone, IP or address with another member

A cluster is ``suspicious`` when that score is >= 0.6 and it has 3-60 members.
Clusters are detection signals, not fraud labels. Only accounts with at least
one payment visible at ``as_of`` are clustered.
Also provides batch CSV pipeline execution for offline evaluation.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
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
    cache: dict | None = None,
) -> dict[str, float] | None:
    as_of_ts = parse_timestamp(as_of)
    account_id = str(account_id)
    # The vector only changes when the account gets a payment or its signup
    # changes, and only grows stale if as_of moves backwards.
    use_cache = cache is not None and exclude_transaction_id is None
    if use_cache:
        key = (graph.payment_count(account_id), graph.get_signup_at(account_id, as_of=as_of_ts))
        hit = cache.get(account_id)
        if hit is not None and hit[0] == key and hit[1] <= as_of_ts:
            return hit[2]
    history = graph.get_account_history(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    if not history:
        return None
    all_visible = max(p.timestamp for p in history) <= as_of_ts
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
    vector = {
        "signup_hours": signup.timestamp() / 3600.0,
        "hour_cos": c,
        "hour_sin": s,
        "hour_dispersion": min(dispersion, 3.0),
        "log_median_gap_hours": math.log1p(median_gap),
    }
    if use_cache and all_visible:
        # Only reusable if every payment was visible; remember the as_of it was built at.
        cache[account_id] = (key, as_of_ts, vector)
    return vector


def detect_lockstep(
    graph: TransactionGraph,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    account_ids: list[str] | None = None,
    active_within_days: int | None = 30,
    identity_graph: Any = None,
    vector_cache: dict | None = None,
) -> list[dict[str, Any]]:
    """Cluster accounts; return one dict per cluster with size >= min_samples."""
    as_of_ts = parse_timestamp(as_of)
    ids = account_ids if account_ids is not None else graph.account_ids(as_of=as_of)
    if active_within_days is not None and account_ids is None:
        since = as_of_ts - timedelta(days=active_within_days)
        recent_payments = graph.payments(as_of=as_of_ts, since=since)
        if recent_payments:
            active_ids = {p.sender for p in recent_payments} | {p.receiver for p in recent_payments}
            ids = [a for a in ids if a in active_ids]

    rows: list[tuple[str, dict[str, float]]] = []
    for account_id in ids:
        vector = account_lockstep_vector(
            graph, account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id,
            cache=vector_cache,
        )
        if vector is not None:
            rows.append((account_id, vector))
    vectors = dict(rows)

    candidates: list[tuple[str, list[str]]] = []
    # 1. Behavioural lookalikes: DBSCAN on the per-account vector.
    if len(rows) >= min_samples:
        matrix = np.array([[row[1][name] for name in FEATURE_NAMES] for row in rows], dtype=float)
        scaled = StandardScaler().fit_transform(matrix)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
        clusters: dict[int, list[str]] = {}
        for idx, label in enumerate(labels):
            if label >= 0:
                clusters.setdefault(int(label), []).append(rows[idx][0])
        candidates += [("DBSCAN", members) for _, members in sorted(clusters.items())]
    # 2. Signup cohorts: accounts created within COHORT_DAYS of each other that
    #    pay each other. Camouflage payments can hide a sleeper batch from
    #    DBSCAN, but not from its own internal payment graph.
    candidates += [("signup_cohort", c) for c in _signup_cohorts(
        graph, set(vectors), as_of_ts, exclude_transaction_id, min_samples)]

    results: list[dict[str, Any]] = []
    seen: set[frozenset] = set()
    for cluster_id, (algorithm, members) in enumerate(candidates):
        key = frozenset(members)
        if key in seen:
            continue
        seen.add(key)
        results.append(_score_cluster(
            graph, cluster_id, algorithm, members, vectors, as_of_ts,
            exclude_transaction_id, eps, min_samples, identity_graph,
        ))
    return results


COHORT_DAYS = 7.0


def _signup_cohorts(
    graph: TransactionGraph,
    ids: set[str],
    as_of: datetime,
    exclude_transaction_id: str | None,
    min_size: int,
) -> list[list[str]]:
    signup = {a: graph.get_signup_at(a, as_of=as_of) for a in ids}
    parent = {a: a for a in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in graph.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id):
        s, r = p.sender, p.receiver
        if s not in parent or r not in parent or signup.get(s) is None or signup.get(r) is None:
            continue
        if abs((signup[s] - signup[r]).total_seconds()) <= COHORT_DAYS * 86400:
            parent[find(s)] = find(r)
    groups: dict[str, list[str]] = {}
    for a in ids:
        groups.setdefault(find(a), []).append(a)
    return [sorted(g) for g in groups.values() if min_size <= len(g) <= MAX_SUSPICIOUS_SIZE]


def _score_cluster(
    graph: TransactionGraph,
    cluster_id: int,
    algorithm: str,
    members: list[str],
    vectors: dict[str, dict[str, float]],
    as_of: datetime,
    exclude_transaction_id: str | None,
    eps: float,
    min_samples: int,
    identity_graph: Any = None,
) -> dict[str, Any]:
    member_set = set(members)
    signups = [vectors[m]["signup_hours"] for m in members]
    span_days = float(max(signups) - min(signups)) / 24.0
    total_out = internal_out = 0
    internal_angles: list[float] = []
    for m in members:
        for p in graph.outgoing_payments(m, as_of=as_of, exclude_transaction_id=exclude_transaction_id):
            total_out += 1
            if p.receiver in member_set:
                internal_out += 1
                internal_angles.append(2 * math.pi * (p.timestamp.hour + p.timestamp.minute / 60.0) / 24.0)
    internal_ratio = internal_out / total_out if total_out else 0.0
    # Rhythm is measured on the members' payments to each other (the coordinated
    # behaviour); fall back to per-account dispersion when there are none.
    if len(internal_angles) >= 3:
        r = math.hypot(float(np.mean(np.cos(internal_angles))), float(np.mean(np.sin(internal_angles))))
        dispersion = min(math.sqrt(-2.0 * math.log(max(r, 1e-9))), 3.0)
    else:
        dispersion = float(np.mean([vectors[m]["hour_dispersion"] for m in members]))
    compactness = max(0.0, 1.0 - span_days / 14.0)
    rhythm = max(0.0, 1.0 - dispersion / 1.0)
    components = [compactness, rhythm, internal_ratio]
    # Synthetic identities share devices/phones/addresses; colleagues or
    # friends who joined together and pay on a rhythm usually do not.
    identity_share = None
    if identity_graph is not None:
        sharing = 0
        for m in members:
            for ident in identity_graph.get_identities(m, as_of=as_of):
                others = identity_graph.get_accounts_for_identity(
                    ident["identity_type"], ident["identity_value"], as_of=as_of)
                if any(o != m and o in member_set for o in others):
                    sharing += 1
                    break
        identity_share = sharing / len(members)
        components.append(identity_share)
    score = float(sum(components) / len(components))
    return {
        "cluster_id": cluster_id,
        "members": members,
        "cluster_size": len(members),
        "lockstep_score": score,
        "suspicious": bool(score >= SUSPICIOUS_SCORE and 3 <= len(members) <= MAX_SUSPICIOUS_SIZE),
        "feature_names": FEATURE_NAMES,
        "member_features": {m: vectors[m] for m in members},
        "evidence": {
            "eps": eps,
            "min_samples": min_samples,
            "algorithm": algorithm,
            "signup_span_days": span_days,
            "mean_hour_dispersion": dispersion,
            "internal_payment_ratio": internal_ratio,
            "internal_payments": internal_out,
            "outgoing_payments": total_out,
            "signup_compactness": compactness,
            "rhythm_tightness": rhythm,
            "identity_share": identity_share,
        },
    }


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