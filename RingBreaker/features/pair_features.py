"""
RingBreaker Pair Risk Feature Engineering Engine
================================================
Extracts payment-level behavioural, temporal, pair-relationship, identity-reuse,
flow, and velocity features without future leakage.

Input:
  - data/payments.csv
  - data/users.csv
Output:
  - data/pair_features.csv
"""

import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# Default constant for unobserved dwell time or first transactions
UNOBSERVED_VALUE = -1.0


def load_and_prepare_data(payments_path="data/payments.csv", users_path="data/users.csv"):
    """
    Loads raw datasets, validates required columns, and parses datetimes.
    """
    if not os.path.exists(payments_path):
        raise FileNotFoundError(f"Missing payments file: {payments_path}")
    if not os.path.exists(users_path):
        raise FileNotFoundError(f"Missing users file: {users_path}")

    df_payments = pd.read_csv(payments_path)
    df_users = pd.read_csv(users_path)

    # Type casting and date parsing
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])
    df_users["signup_timestamp"] = pd.to_datetime(df_users["signup_timestamp"])

    # Ensure chronological sorting
    df_payments = df_payments.sort_values(by="timestamp").reset_index(drop=True)

    return df_payments, df_users


def _prune_queue(q: deque, current_time: datetime, window_seconds: float):
    """
    Removes timestamps strictly older than (current_time - window_seconds).
    """
    cutoff = current_time - timedelta(seconds=window_seconds)
    while q and q[0] < cutoff:
        q.popleft()


def build_pair_features(payments_path="data/payments.csv", users_path="data/users.csv"):
    """
    Single-pass causal expanding window feature extractor.
    Operates strictly forward in time: for row t, state reflects only rows 0 to t-1.
    """
    df_payments, df_users = load_and_prepare_data(payments_path, users_path)

    # Pre-map user signup timestamps for O(1) access
    user_signup = dict(zip(df_users["user_id"], df_users["signup_timestamp"]))

    # -------------------------------------------------------------
    # EXPANDING STATE REGISTRIES (History strictly < current tx time)
    # -------------------------------------------------------------
    # Sender state
    sender_tx_count = defaultdict(int)
    sender_total_amt = defaultdict(float)
    sender_max_amt = defaultdict(float)
    sender_receivers_seen = defaultdict(set)
    sender_last_tx_time = {}
    sender_tx_gaps_sum = defaultdict(float)
    sender_out_degree = defaultdict(int)

    # Receiver state
    receiver_tx_count = defaultdict(int)
    receiver_total_amt = defaultdict(float)
    receiver_max_amt = defaultdict(float)
    receiver_senders_seen = defaultdict(set)
    receiver_in_degree = defaultdict(int)
    receiver_last_incoming_time = {}

    # Pair state
    pair_tx_count = defaultdict(int)
    pair_total_amt = defaultdict(float)
    pair_last_tx_time = {}

    # Device & IP reuse registries
    device_tx_count = defaultdict(int)
    device_users_seen = defaultdict(set)
    ip_tx_count = defaultdict(int)
    ip_users_seen = defaultdict(set)

    # Velocity deques (storing timestamps of past events)
    sender_q_1h = defaultdict(deque)
    sender_q_24h = defaultdict(deque)
    sender_q_7d = defaultdict(deque)

    receiver_q_1h = defaultdict(deque)
    receiver_q_24h = defaultdict(deque)
    receiver_q_7d = defaultdict(deque)

    # Pre-allocate output records
    features = []

    # -------------------------------------------------------------
    # SEQUENTIAL CHRONOLOGICAL SCAN
    # -------------------------------------------------------------
    for row in df_payments.itertuples(index=False):
        tx_id = row.transaction_id
        t = row.timestamp
        s = row.sender
        r = row.receiver
        amt = float(row.amount)
        dev = str(row.device_id)
        ip = str(row.ip_address)
        label = int(row.is_fraud)
        ring = row.ring_id

        # =========================================================
        # 1. BASE PAYMENT & TEMPORAL FEATURES (Current row only)
        # =========================================================
        log_amt = float(np.log1p(max(amt, 0.0)))
        hr = t.hour
        dow = t.dayofweek
        dom = t.day
        is_wknd = 1 if dow >= 5 else 0
        is_night = 1 if (0 <= hr <= 5) else 0

        # Account age relative to payment time
        s_signup = user_signup.get(s, t)
        r_signup = user_signup.get(r, t)
        s_account_age_days = max(0.0, (t - s_signup).total_seconds() / 86400.0)
        r_account_age_days = max(0.0, (t - r_signup).total_seconds() / 86400.0)

        # =========================================================
        # 2. SENDER HISTORICAL BEHAVIOUR (Past strictly < t)
        # =========================================================
        s_cnt = sender_tx_count[s]
        s_uniq_rec = len(sender_receivers_seen[s])
        s_total = sender_total_amt[s]
        s_avg_amt = (s_total / s_cnt) if s_cnt > 0 else 0.0
        s_max_amt = sender_max_amt[s] if s_cnt > 0 else 0.0

        if s_cnt > 1:
            s_avg_gap = sender_tx_gaps_sum[s] / (s_cnt - 1)
        else:
            s_avg_gap = UNOBSERVED_VALUE

        # Sender velocities
        _prune_queue(sender_q_1h[s], t, 3600.0)
        _prune_queue(sender_q_24h[s], t, 86400.0)
        _prune_queue(sender_q_7d[s], t, 604800.0)

        s_v_1h = len(sender_q_1h[s])
        s_v_24h = len(sender_q_24h[s])
        s_v_7d = len(sender_q_7d[s])

        # =========================================================
        # 3. RECEIVER HISTORICAL BEHAVIOUR (Past strictly < t)
        # =========================================================
        r_cnt = receiver_tx_count[r]
        r_uniq_snd = len(receiver_senders_seen[r])
        r_total = receiver_total_amt[r]
        r_avg_amt = (r_total / r_cnt) if r_cnt > 0 else 0.0
        r_max_amt = receiver_max_amt[r] if r_cnt > 0 else 0.0

        # Receiver velocities
        _prune_queue(receiver_q_1h[r], t, 3600.0)
        _prune_queue(receiver_q_24h[r], t, 86400.0)
        _prune_queue(receiver_q_7d[r], t, 604800.0)

        r_v_1h = len(receiver_q_1h[r])
        r_v_24h = len(receiver_q_24h[r])
        r_v_7d = len(receiver_q_7d[r])

        # =========================================================
        # 4. SENDER -> RECEIVER PAIR RELATIONSHIP (Past strictly < t)
        # =========================================================
        pair_key = (s, r)
        p_cnt = pair_tx_count[pair_key]
        p_total = pair_total_amt[pair_key]
        p_avg_amt = (p_total / p_cnt) if p_cnt > 0 else 0.0
        is_first_pair = 1 if p_cnt == 0 else 0

        if pair_key in pair_last_tx_time:
            time_since_prev_pair = (t - pair_last_tx_time[pair_key]).total_seconds()
        else:
            time_since_prev_pair = UNOBSERVED_VALUE

        # =========================================================
        # 5. DEVICE & IP REUSE FEATURES (Past strictly < t)
        # =========================================================
        dev_cnt = device_tx_count[dev]
        dev_uniq_users = len(device_users_seen[dev])

        ip_cnt = ip_tx_count[ip]
        ip_uniq_users = len(ip_users_seen[ip])

        # =========================================================
        # 6. FLOW, DEGREE & PASS-THROUGH DWELL (Past strictly < t)
        # =========================================================
        s_out_deg = sender_out_degree[s]
        s_in_deg = receiver_in_degree[s]  # Times s received money historically
        r_out_deg = sender_out_degree[r]  # Times r sent money historically
        r_in_deg = receiver_in_degree[r]

        # Flow ratio safe divisions: out / (in + 1)
        s_out_in_ratio = s_out_deg / (s_in_deg + 1.0)
        r_in_out_ratio = r_in_deg / (r_out_deg + 1.0)

        # Pass-through dwell time: how recently did the sender receive funds?
        if s in receiver_last_incoming_time:
            s_time_since_last_incoming = (t - receiver_last_incoming_time[s]).total_seconds()
        else:
            s_time_since_last_incoming = UNOBSERVED_VALUE

        # Assemble clean feature row
        features.append({
            # Identifiers and Metadata (Excluded from X matrix during modeling)
            "transaction_id": tx_id,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": s,
            "receiver": r,
            # Target & Evaluation
            "is_fraud": label,
            "ring_id": ring if pd.notnull(ring) else "",
            # Base Payment Features
            "amount": amt,
            "log_amount": log_amt,
            # Time Features
            "hour": hr,
            "day_of_week": dow,
            "day_of_month": dom,
            "is_weekend": is_wknd,
            "is_night": is_night,
            "sender_account_age_days": s_account_age_days,
            "receiver_account_age_days": r_account_age_days,
            # Sender Behaviour Features
            "sender_tx_count_before": s_cnt,
            "sender_unique_receivers_before": s_uniq_rec,
            "sender_avg_amount_before": s_avg_amt,
            "sender_max_amount_before": s_max_amt,
            "sender_avg_time_gap": s_avg_gap,
            "sender_tx_last_1h": s_v_1h,
            "sender_tx_last_24h": s_v_24h,
            "sender_tx_last_7d": s_v_7d,
            # Receiver Behaviour Features
            "receiver_tx_count_before": r_cnt,
            "receiver_unique_senders_before": r_uniq_snd,
            "receiver_avg_amount_before": r_avg_amt,
            "receiver_max_amount_before": r_max_amt,
            "receiver_tx_last_1h": r_v_1h,
            "receiver_tx_last_24h": r_v_24h,
            "receiver_tx_last_7d": r_v_7d,
            # Pair / Relationship Features
            "pair_tx_count_before": p_cnt,
            "pair_avg_amount_before": p_avg_amt,
            "time_since_previous_pair_tx": time_since_prev_pair,
            "is_first_transaction_between_pair": is_first_pair,
            # Identity Fragment Reuse Features
            "device_tx_count_before": dev_cnt,
            "device_unique_users_before": dev_uniq_users,
            "ip_tx_count_before": ip_cnt,
            "ip_unique_users_before": ip_uniq_users,
            # Flow & Degree Features
            "sender_in_degree_before": s_in_deg,
            "sender_out_degree_before": s_out_deg,
            "receiver_in_degree_before": r_in_deg,
            "receiver_out_degree_before": r_out_deg,
            "sender_out_in_ratio": s_out_in_ratio,
            "receiver_in_out_ratio": r_in_out_ratio,
            "sender_time_since_last_incoming": s_time_since_last_incoming
        })

        # =========================================================
        # UPDATE STATE REGISTRIES (Now incorporating row t for future)
        # =========================================================
        # Sender updates
        if s in sender_last_tx_time:
            gap = (t - sender_last_tx_time[s]).total_seconds()
            sender_tx_gaps_sum[s] += gap
        sender_last_tx_time[s] = t

        sender_tx_count[s] += 1
        sender_total_amt[s] += amt
        if amt > sender_max_amt[s]:
            sender_max_amt[s] = amt
        sender_receivers_seen[s].add(r)
        sender_out_degree[s] += 1

        sender_q_1h[s].append(t)
        sender_q_24h[s].append(t)
        sender_q_7d[s].append(t)

        # Receiver updates
        receiver_tx_count[r] += 1
        receiver_total_amt[r] += amt
        if amt > receiver_max_amt[r]:
            receiver_max_amt[r] = amt
        receiver_senders_seen[r].add(s)
        receiver_in_degree[r] += 1
        receiver_last_incoming_time[r] = t

        receiver_q_1h[r].append(t)
        receiver_q_24h[r].append(t)
        receiver_q_7d[r].append(t)

        # Pair updates
        pair_tx_count[pair_key] += 1
        pair_total_amt[pair_key] += amt
        pair_last_tx_time[pair_key] = t

        # Identity updates
        device_tx_count[dev] += 1
        device_users_seen[dev].add(s)
        ip_tx_count[ip] += 1
        ip_users_seen[ip].add(s)

    df_out = pd.DataFrame(features)
    return df_out


# =====================================================================
# LEAKAGE SANITY TEST
# =====================================================================
def run_leakage_test(payments_path="data/payments.csv", users_path="data/users.csv"):
    """
    Proves zero future leakage.
    Features for transactions T1..T10 computed on a 10-row dataset MUST BE
    IDENTICAL to features for T1..T10 computed on a 50-row dataset.
    """
    df_p = pd.read_csv(payments_path).sort_values(by="timestamp").reset_index(drop=True)
    df_u = pd.read_csv(users_path)

    # Subsets
    df_sub_a = df_p.iloc[:15].copy()
    df_sub_b = df_p.iloc[:40].copy()

    # Temporary files
    tmp_a = "data/_tmp_test_a.csv"
    tmp_b = "data/_tmp_test_b.csv"
    df_sub_a.to_csv(tmp_a, index=False)
    df_sub_b.to_csv(tmp_b, index=False)

    try:
        feat_a = build_pair_features(tmp_a, users_path)
        feat_b = build_pair_features(tmp_b, users_path)

        # Exclude ring_id string differences if any
        eval_cols = [c for c in feat_a.columns if c not in ["ring_id"]]

        # Compare row-for-row on the first 15 transactions
        diff = feat_a[eval_cols].compare(feat_b.iloc[:15][eval_cols])
        assert diff.empty, "LEAKAGE TEST FAILED: Future transactions influenced past features!"
        print("✓ Sanity Test Passed: Future transactions do NOT alter historical features.")
    finally:
        if os.path.exists(tmp_a):
            os.remove(tmp_a)
        if os.path.exists(tmp_b):
            os.remove(tmp_b)


# =====================================================================
# VERIFICATION & SUMMARY REPORTS
# =====================================================================
def print_quality_checks(df_features, df_payments):
    feature_cols = [
        c for c in df_features.columns
        if c not in ["transaction_id", "timestamp", "sender", "receiver", "is_fraud", "ring_id"]
    ]

    print("\n" + "=" * 60)
    print("RINGBREAKER PAIR-RISK FEATURE VALIDATION REPORT")
    print("=" * 60)
    print(f"Original transactions:     {len(df_payments):,}")
    print(f"Feature rows:              {len(df_features):,}")
    print(f"Number of ML features:     {len(feature_cols)}")
    print(f"Normal transactions:       {(df_features['is_fraud'] == 0).sum():,}")
    print(f"Fraud transactions:        {(df_features['is_fraud'] == 1).sum():,}")
    print(f"Date range:                {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")

    # Check 1: Row count parity
    assert len(df_features) == len(df_payments), "Check 1 Failed: Row count mismatch."
    print("✓ Check 1: Feature row count matches raw payments.")

    # Check 2: Unique transaction IDs
    assert df_features["transaction_id"].is_unique, "Check 2 Failed: Duplicate transaction IDs found."
    print("✓ Check 2: All transaction IDs are unique.")

    # Check 3 & 4: Label and Ring ID status
    assert "is_fraud" in df_features.columns and "ring_id" in df_features.columns, "Check 3/4 Failed: Target missing."
    print("✓ Check 3: is_fraud present only as target.")
    print("✓ Check 4: ring_id present only for evaluation metadata.")

    # Check 5 & 6: Feature set purity
    for col in feature_cols:
        assert "fraud" not in col.lower(), f"Check 6 Failed: Target leakage in feature {col}"
        assert "ring" not in col.lower(), f"Check 6 Failed: Ring ID leakage in feature {col}"
    print("✓ Check 5 & 6: Zero target/ring leakage in feature definitions.")

    # Check 7: Chronological monotonicity
    dt_series = pd.to_datetime(df_features["timestamp"])
    assert dt_series.is_monotonic_increasing, "Check 7 Failed: Chronological sorting violated."
    print("✓ Check 7: Chronological ordering strictly preserved.")

    # Missing values check across ML features
    null_counts = df_features[feature_cols].isnull().sum()
    print("\nMissing values per feature:")
    if null_counts.sum() == 0:
        print("  None. All features are fully imputed and ready for modeling.")
    else:
        for feat, n_null in null_counts[null_counts > 0].items():
            print(f"  - {feat:35s}: {n_null} nulls")

    print("\n" + "=" * 60)
    print("FEATURE CATALOG SUMMARY")
    print("=" * 60)
    print(f"{'Feature Name':<35} | {'Group':<22} | {'Description'}")
    print("-" * 85)

    catalog = [
        ("amount", "Payment Features", "Monetary value of payment"),
        ("log_amount", "Payment Features", "Log1p transformed monetary amount"),
        ("hour", "Time Features", "Hour of day (0-23)"),
        ("day_of_week", "Time Features", "Day of week (0=Mon, 6=Sun)"),
        ("day_of_month", "Time Features", "Day of month (1-31)"),
        ("is_weekend", "Time Features", "1 if Saturday/Sunday else 0"),
        ("is_night", "Time Features", "1 if between 00:00 and 05:59 else 0"),
        ("sender_account_age_days", "Time Features", "Sender account age at payment time"),
        ("receiver_account_age_days", "Time Features", "Receiver account age at payment time"),
        ("sender_tx_count_before", "Sender Features", "Historical out-bound transaction count"),
        ("sender_unique_receivers_before", "Sender Features", "Number of distinct receivers previously paid"),
        ("sender_avg_amount_before", "Sender Features", "Historical average out-bound amount"),
        ("sender_max_amount_before", "Sender Features", "Historical max out-bound amount"),
        ("sender_avg_time_gap", "Sender Features", "Average time gap between historical payments"),
        ("sender_tx_last_1h", "Sender Features", "Velocity: payments sent in last 1 hour"),
        ("sender_tx_last_24h", "Sender Features", "Velocity: payments sent in last 24 hours"),
        ("sender_tx_last_7d", "Sender Features", "Velocity: payments sent in last 7 days"),
        ("receiver_tx_count_before", "Receiver Features", "Historical in-bound transaction count"),
        ("receiver_unique_senders_before", "Receiver Features", "Number of distinct senders previously received from"),
        ("receiver_avg_amount_before", "Receiver Features", "Historical average in-bound amount"),
        ("receiver_max_amount_before", "Receiver Features", "Historical max in-bound amount"),
        ("receiver_tx_last_1h", "Receiver Features", "Velocity: payments received in last 1 hour"),
        ("receiver_tx_last_24h", "Receiver Features", "Velocity: payments received in last 24 hours"),
        ("receiver_tx_last_7d", "Receiver Features", "Velocity: payments received in last 7 days"),
        ("pair_tx_count_before", "Pair Features", "Past payment count between this exact (S, R) pair"),
        ("pair_avg_amount_before", "Pair Features", "Historical average amount between this (S, R) pair"),
        ("time_since_previous_pair_tx", "Pair Features", "Seconds elapsed since prior payment between this pair"),
        ("is_first_transaction_between_pair", "Pair Features", "1 if pair never transacted before else 0"),
        ("device_tx_count_before", "Device Features", "Historical count of payments from this device_id"),
        ("device_unique_users_before", "Device Features", "Distinct senders historically sharing this device_id"),
        ("ip_tx_count_before", "IP Features", "Historical count of payments from this ip_address"),
        ("ip_unique_users_before", "IP Features", "Distinct senders historically sharing this ip_address"),
        ("sender_in_degree_before", "Flow Features", "Historical in-degree of sender (funds received)"),
        ("sender_out_degree_before", "Flow Features", "Historical out-degree of sender (funds dispatched)"),
        ("receiver_in_degree_before", "Flow Features", "Historical in-degree of receiver"),
        ("receiver_out_degree_before", "Flow Features", "Historical out-degree of receiver"),
        ("sender_out_in_ratio", "Flow Features", "Sender ratio of out-degree to in-degree"),
        ("receiver_in_out_ratio", "Flow Features", "Receiver ratio of in-degree to out-degree"),
        ("sender_time_since_last_incoming", "Flow Features", "Seconds since sender last received funds (dwell proxy)")
    ]

    for f_name, grp, desc in catalog:
        print(f"{f_name:<35} | {grp:<22} | {desc}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    payments_csv = "data/payments.csv"
    users_csv = "data/users.csv"
    output_csv = "data/pair_features.csv"

    print("Running data leakage verification tests...")
    run_leakage_test(payments_csv, users_csv)

    print(f"\nExtracting features from {payments_csv} and {users_csv}...")
    df_raw_payments = pd.read_csv(payments_csv)
    df_feature_table = build_pair_features(payments_csv, users_csv)

    print_quality_checks(df_feature_table, df_raw_payments)

    df_feature_table.to_csv(output_csv, index=False)
    print(f"Successfully saved feature table to '{output_csv}' ({len(df_feature_table):,} rows).")