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

import pandas as pd

from ringbreaker.features.online import FeaturePayment, OnlineFeatureState




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


def build_pair_features(payments_path="data/payments.csv", users_path="data/users.csv"):
    """
    Single-pass causal feature extractor.

    Delegates to :class:`ringbreaker.features.online.OnlineFeatureState`, the same
    state object the live API uses, so offline training rows and online scoring
    rows are computed by identical code. For row t, state reflects rows 0..t-1.
    """
    df_payments, df_users = load_and_prepare_data(payments_path, users_path)
    return build_feature_frame(df_payments, df_users)


def build_feature_frame(df_payments: pd.DataFrame, df_users: pd.DataFrame, include_behaviour: bool = False) -> pd.DataFrame:
    """Build the pair-model (and optionally EIF behaviour) features for sorted payments."""
    signups = {
        str(u): pd.Timestamp(ts).to_pydatetime()
        for u, ts in zip(df_users["user_id"], pd.to_datetime(df_users["signup_timestamp"]))
    }
    state = OnlineFeatureState(signups)
    rows = []
    for row in df_payments.itertuples(index=False):
        p = FeaturePayment(
            sender=str(row.sender),
            receiver=str(row.receiver),
            amount=float(row.amount),
            timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
            device=str(row.device_id) if pd.notnull(row.device_id) else None,
            ip=str(row.ip_address) if pd.notnull(row.ip_address) else None,
        )
        record = {
            "transaction_id": row.transaction_id,
            "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": p.sender,
            "receiver": p.receiver,
            "is_fraud": int(row.is_fraud),
            "ring_id": row.ring_id if pd.notnull(row.ring_id) else "",
        }
        if hasattr(row, "label_observed"):
            record["label_observed"] = int(row.label_observed)
        record.update(state.pair_features(p))
        if include_behaviour:
            record.update(state.behaviour_features(p))
        state.update(p)
        rows.append(record)
    return pd.DataFrame(rows)


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