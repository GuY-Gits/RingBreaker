from datetime import datetime, timedelta

from ringbreaker.features.flow import account_flow_features
from ringbreaker.graphs.build import TransactionGraph


def test_dwell_and_pass_through():
    g = TransactionGraph()
    t = datetime(2024, 1, 1, 10, 0, 0)
    g.add_payment("S", "A", 100, t, transaction_id="in1")
    g.add_payment("A", "B", 80, t + timedelta(hours=2), transaction_id="out1")
    feats = account_flow_features(g, "A", as_of=t + timedelta(hours=2))
    assert feats["incoming_count"] == 1
    assert feats["outgoing_count"] == 1
    assert feats["incoming_amount"] == 100
    assert feats["outgoing_amount"] == 80
    assert abs(feats["pass_through_ratio"] - 0.8) < 1e-9
    assert abs(feats["mean_dwell_seconds"] - 7200) < 1e-6


def test_new_account_zeros():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    feats = account_flow_features(g, "new", as_of=t)
    assert feats["incoming_count"] == 0
    assert feats["pass_through_ratio"] == 0.0
    assert feats["mean_dwell_seconds"] == 0.0


def test_in_only_and_out_only():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    g.add_payment("S", "IN", 10, t, transaction_id="a")
    g.add_payment("OUT", "R", 10, t, transaction_id="b")
    inn = account_flow_features(g, "IN", as_of=t)
    out = account_flow_features(g, "OUT", as_of=t)
    assert inn["pass_through_ratio"] == 0.0
    assert inn["outgoing_count"] == 0
    assert out["incoming_count"] == 0
    assert out["pass_through_ratio"] == 0.0
    assert out["mean_dwell_seconds"] == 0.0


def test_zero_amount_does_not_nan():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    g.add_payment("S", "A", 0, t, transaction_id="z")
    feats = account_flow_features(g, "A", as_of=t)
    assert feats["pass_through_ratio"] == 0.0
    assert feats["mean_dwell_seconds"] == 0.0
