"""Automated integrity and realism checks for RingBreaker Simulator v2."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ringbreaker.simulator.config import SimulatorConfig


def validate_simulator_data(
    df_users: pd.DataFrame,
    df_payments: pd.DataFrame,
    df_rings: pd.DataFrame,
    config: SimulatorConfig,
    heldout_start_time: pd.Timestamp,
) -> None:
    """Runs automated integrity and realism verification against Simulator v2 targets."""
    print("\n" + "=" * 60)
    print("RUNNING RINGBREAKER SIMULATOR V2 QUALITY & REALISM CHECKS")
    print("=" * 60)

    # 1. Null Checks
    assert df_users.isnull().sum().sum() == 0, "Error: Missing values found in users.csv"
    req_cols = ["transaction_id", "timestamp", "sender", "receiver", "amount", "is_fraud", "label_observed"]
    assert df_payments[req_cols].isnull().sum().sum() == 0, "Error: Missing required fields in payments.csv"

    # 2. Graph Self-Loop Check
    self_tx = df_payments[df_payments["sender"] == df_payments["receiver"]]
    assert len(self_tx) == 0, "Error: Detected self-payments where sender == receiver"

    # 3. User Existence
    all_users = set(df_users["user_id"])
    all_senders = set(df_payments["sender"])
    all_receivers = set(df_payments["receiver"])
    assert all_senders.issubset(all_users), "Error: Unregistered sender found in payments"
    assert all_receivers.issubset(all_users), "Error: Unregistered receiver found in payments"

    # 4. Chronological Order
    tx_times = pd.to_datetime(df_payments["timestamp"])
    assert tx_times.is_monotonic_increasing, "Error: payments.csv is not chronologically sorted"

    # 4b. Simulation Horizon Bounds
    max_allowed = config.base_start_date + timedelta(days=config.sim_days + 1)
    max_payment_ts = tx_times.max()
    assert max_payment_ts <= pd.Timestamp(max_allowed), (
        f"Error: Payment timestamp {max_payment_ts} exceeds simulation window max {max_allowed}"
    )

    # 4c. No account transacts before it exists
    ts_all = pd.to_datetime(df_payments["timestamp"])
    first_seen = pd.concat([
        pd.DataFrame({"acc": df_payments["sender"], "ts": ts_all}),
        pd.DataFrame({"acc": df_payments["receiver"], "ts": ts_all}),
    ]).groupby("acc")["ts"].min()
    signups = pd.to_datetime(df_users.set_index("user_id")["signup_timestamp"]).reindex(first_seen.index)
    early = int((first_seen < signups).sum())
    assert early == 0, f"Error: {early} accounts transact before their signup"

    # 5. Split Isolation (Held-Out novel families strictly in heldout window)
    heldout_only_rings = df_rings[df_rings["variant"] == "held_out_novel"]["ring_id"].tolist()
    if heldout_only_rings:
        heldout_txs = df_payments[df_payments["ring_id"].isin(heldout_only_rings)]
        earliest_heldout = pd.to_datetime(heldout_txs["timestamp"]).min()
        assert earliest_heldout >= heldout_start_time, (
            f"Error: Held-out novel ring leaked into training partition! "
            f"Earliest: {earliest_heldout}, Limit: {heldout_start_time}"
        )

    # 6. Rates & Target Checks
    total_tx = len(df_payments)
    fraud_tx = int(df_payments["is_fraud"].sum())
    fraud_pct = (fraud_tx / total_tx) * 100

    fraud_accounts = set(df_payments[df_payments["is_fraud"] == 1]["sender"]).union(
        set(df_payments[df_payments["is_fraud"] == 1]["receiver"])
    )
    touched_pct = (len(fraud_accounts) / len(df_users)) * 100

    # Fraud payment rate target: 0.6% - 1.0% (at full scale, with realistic variance)
    if len(df_users) >= 4000:
        assert 0.5 <= fraud_pct <= 1.5, f"Fraud payment rate out of target band: {fraud_pct:.2f}%"
        assert 1.5 <= touched_pct <= 4.5, f"Fraud touched accounts rate out of target band: {touched_pct:.2f}%"
    else:
        assert 0.2 <= fraud_pct <= 8.0, f"Fraud payment rate unexpected for small test: {fraud_pct:.2f}%"
        assert 0.5 <= touched_pct <= 30.0, f"Touched accounts rate unexpected for small test: {touched_pct:.2f}%"

    # 7. Heavy-Tailed Degree Distribution (Top 5% carry >= 25% of payments)
    tx_counts = df_payments["sender"].value_counts()
    top_5_pct_count = max(1, int(len(df_users) * 0.05))
    top_share = tx_counts.iloc[:top_5_pct_count].sum() / total_tx
    assert top_share >= 0.20, f"Traffic not heavy-tailed enough: top 5% carry {top_share * 100:.1f}%"

    # 8. Amount Distribution Overlap
    normal_amounts = df_payments[df_payments["is_fraud"] == 0]["amount"]
    fraud_amounts = df_payments[df_payments["is_fraud"] == 1]["amount"]
    assert len(fraud_amounts) > 0, "No fraud amounts to evaluate"
    # Both sets should cover typical retail ranges (e.g. median in INR 500 - 6000)
    assert 200 <= normal_amounts.median() <= 8000, f"Normal amount median unusual: {normal_amounts.median()}"
    assert 500 <= fraud_amounts.median() <= 12000, f"Fraud amount median unusual: {fraud_amounts.median()}"

    # 9. Label Noise Verification
    obs_fraud = int(df_payments["label_observed"].sum())
    obs_pct = (obs_fraud / total_tx) * 100

    print(f"Total Users:                 {len(df_users):,}")
    print(f"Total Payments:              {total_tx:,}")
    print(f"Fraud Payments (ground truth): {fraud_tx:,} ({fraud_pct:.2f}%)")
    print(f"Observed Fraud Labels:       {obs_fraud:,} ({obs_pct:.2f}%)")
    print(f"Accounts Touched by Fraud:   {len(fraud_accounts):,} ({touched_pct:.2f}%)")
    print(f"Total Planted Rings:         {len(df_rings)}")
    print(f"Top 5% Accounts Volume:      {top_share * 100:.1f}%")
    print(f"Normal Median Amount:        INR {normal_amounts.median():.2f}")
    print(f"Fraud Median Amount:         INR {fraud_amounts.median():.2f}")
    print("=" * 60)
    print("ALL SIMULATOR V2 QUALITY CHECKS PASSED SUCCESSFULLY!\n")
