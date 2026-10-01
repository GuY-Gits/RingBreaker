"""Tests for Simulator v2: determinism, rates, temporal isolation, distributions, and leakage guard."""

import pandas as pd
import pytest

from ringbreaker.simulator.config import SimulatorConfig
from ringbreaker.simulator.generate import generate_dataset


@pytest.fixture(scope="module")
def sim_data():
    # Use a slightly smaller config for fast unit testing in pytest
    cfg = SimulatorConfig(
        num_users=1000,
        target_payments=8000,
        random_seed=42,
    )
    users, payments, rings = generate_dataset(cfg)
    return users, payments, rings, cfg


def test_simulator_determinism():
    cfg1 = SimulatorConfig(num_users=300, target_payments=1500, random_seed=99)
    u1, p1, r1 = generate_dataset(cfg1)

    cfg2 = SimulatorConfig(num_users=300, target_payments=1500, random_seed=99)
    u2, p2, r2 = generate_dataset(cfg2)

    pd.testing.assert_frame_equal(u1, u2)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(r1, r2)


def test_fraud_rates_and_counts(sim_data):
    users, payments, rings, cfg = sim_data
    assert len(users) == 1000
    assert len(payments) >= 6000

    fraud_tx = int(payments["is_fraud"].sum())
    fraud_pct = (fraud_tx / len(payments)) * 100
    # Target 0.6% - 1.2%
    assert 0.5 <= fraud_pct <= 1.5, f"Fraud payment rate unexpected: {fraud_pct:.2f}%"

    fraud_accounts = set(payments[payments["is_fraud"] == 1]["sender"]).union(
        set(payments[payments["is_fraud"] == 1]["receiver"])
    )
    touched_pct = (len(fraud_accounts) / len(users)) * 100
    assert 1.5 <= touched_pct <= 8.0, f"Touched accounts rate unexpected: {touched_pct:.2f}%"


def test_split_isolation(sim_data):
    users, payments, rings, cfg = sim_data
    heldout_rings = rings[rings["split"] == "heldout"]["ring_id"].tolist()
    assert len(heldout_rings) > 0, "No held-out rings found"

    heldout_txs = payments[payments["ring_id"].isin(heldout_rings)]
    assert len(heldout_txs) > 0, "No payments in heldout rings"

    heldout_start = pd.Timestamp(cfg.base_start_date) + pd.Timedelta(days=cfg.held_out_start_day)
    earliest_heldout = pd.to_datetime(heldout_txs["timestamp"]).min()
    assert earliest_heldout >= heldout_start, (
        f"Temporal leakage: earliest held-out payment at {earliest_heldout} before cutoff {heldout_start}"
    )


def test_label_noise_rates(sim_data):
    _, payments, _, cfg = sim_data
    assert "label_observed" in payments.columns

    fraud_true = payments["is_fraud"] == 1
    normal_true = payments["is_fraud"] == 0

    # False negatives: fraud marked as 0 in label_observed
    fn_rate = (payments.loc[fraud_true, "label_observed"] == 0).mean()
    # Target ~10% (allow tolerance for small sample)
    assert 0.04 <= fn_rate <= 0.20, f"False negative rate: {fn_rate}"

    # False positives: normal marked as 1 in label_observed
    fp_rate = (payments.loc[normal_true, "label_observed"] == 1).mean()
    # Target ~0.2% (allow tolerance)
    assert 0.0005 <= fp_rate <= 0.01, f"False positive rate: {fp_rate}"


def test_feature_leakage_guard(sim_data):
    users, payments, _, _ = sim_data
    # Verify that metadata columns exist in raw tables
    assert "persona" in users.columns
    assert "label_observed" in payments.columns

    # Ensure prohibited metadata columns are NEVER present in online feature definitions
    from ringbreaker.features.online import BEHAVIOUR_FEATURE_NAMES, PAIR_FEATURE_NAMES
    prohibited = {"persona", "family", "split", "variant", "params", "label_observed", "is_fraud", "ring_id"}
    assert not (set(PAIR_FEATURE_NAMES) & prohibited)
    assert not (set(BEHAVIOUR_FEATURE_NAMES) & prohibited)


def test_no_account_transacts_before_signup(sim_data):
    users, payments, _, _ = sim_data
    ts = pd.to_datetime(payments["timestamp"])
    first = pd.concat([
        pd.DataFrame({"acc": payments["sender"], "ts": ts}),
        pd.DataFrame({"acc": payments["receiver"], "ts": ts}),
    ]).groupby("acc")["ts"].min()
    signup = pd.to_datetime(users.set_index("user_id")["signup_timestamp"]).reindex(first.index)
    assert int((first < signup).sum()) == 0


def test_novel_families_use_fresh_accounts(sim_data):
    _, _, rings, _ = sim_data
    novel = rings[rings["variant"] == "held_out_novel"]
    earlier = rings[~rings["split"].str.contains("heldout")]
    earlier_accounts = {m for ms in earlier["members"] for m in ms.split(";")}
    assert len(novel) > 0
    for ms in novel["members"]:
        assert not set(ms.split(";")) & earlier_accounts


def test_benign_groups_tagged_and_genuine(sim_data):
    _, payments, _, _ = sim_data
    tagged = payments[payments["group_id"].notna()]
    assert len(tagged) > 0
    assert int(tagged["is_fraud"].sum()) == 0
    assert "benign_groups" in payments.attrs


def test_synthetic_identities_share_fragments(sim_data):
    users, _, rings, _ = sim_data
    meta = users.set_index("user_id")
    for ms in rings[rings["family"] == "synthetic_sleeper"]["members"]:
        members = ms.split(";")
        devices = meta.loc[members, "device_id"]
        assert devices.nunique() < len(members)
