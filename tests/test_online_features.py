"""Temporal correctness of the shared online feature store and engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from ringbreaker.engine import Engine, EngineError
from ringbreaker.features.online import FeaturePayment, OnlineFeatureState
from ringbreaker.features.pair_features import build_feature_frame

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _payments():
    return [
        FeaturePayment("A", "B", 100, T0, "d1", "ip1"),
        FeaturePayment("B", "C", 95, T0 + timedelta(minutes=5), "d2", "ip2"),
        FeaturePayment("A", "B", 50, T0 + timedelta(hours=2), "d1", "ip1"),
        FeaturePayment("C", "A", 30, T0 + timedelta(days=1), "d1", "ip1"),
    ]


def test_features_ignore_later_payments():
    """Features for payment k are identical whether or not later payments exist."""
    pays = _payments()
    full = OnlineFeatureState()
    rows_full = []
    for p in pays:
        rows_full.append(full.pair_features(p))
        full.update(p)
    for k in range(len(pays)):
        partial = OnlineFeatureState()
        for p in pays[:k]:
            partial.update(p)
        assert partial.pair_features(pays[k]) == rows_full[k]


def test_candidate_not_counted_in_its_own_features():
    s = OnlineFeatureState()
    p = _payments()[0]
    f = s.pair_features(p)
    assert f["sender_tx_count_before"] == 0
    assert f["is_first_transaction_between_pair"] == 1.0
    assert f["device_unique_users_before"] == 0


def test_signup_after_payment_is_not_visible():
    s = OnlineFeatureState({"B": T0 + timedelta(days=3)})
    assert s.pair_features(_payments()[0])["receiver_account_age_days"] == 0.0


def test_batch_and_online_features_match():
    pays = _payments()
    df = pd.DataFrame([{
        "transaction_id": f"t{i}", "timestamp": p.timestamp, "sender": p.sender, "receiver": p.receiver,
        "amount": p.amount, "device_id": p.device, "ip_address": p.ip, "is_fraud": 0, "ring_id": None,
    } for i, p in enumerate(pays)])
    users = pd.DataFrame({"user_id": ["A", "B", "C"], "signup_timestamp": [T0 - timedelta(days=9)] * 3})
    batch = build_feature_frame(df, users)
    online = OnlineFeatureState({u: T0 - timedelta(days=9) for u in "ABC"})
    for i, p in enumerate(pays):
        f = online.pair_features(p)
        online.update(p)
        for name, value in f.items():
            assert batch.iloc[i][name] == pytest.approx(value), name


def test_engine_rejects_out_of_order_and_duplicates():
    e = Engine(load_history=False)
    e.score({"sender": "A", "receiver": "B", "amount": 10, "timestamp": T0.isoformat(), "transaction_id": "x1"})
    with pytest.raises(EngineError) as exc:
        e.score({"sender": "A", "receiver": "B", "amount": 10, "timestamp": (T0 - timedelta(seconds=1)).isoformat()})
    assert exc.value.status == 409
    with pytest.raises(EngineError):
        e.score({"sender": "A", "receiver": "B", "amount": 10, "timestamp": T0.isoformat(), "transaction_id": "x1"})


def test_engine_scores_on_history_before_insert():
    e = Engine(load_history=False)
    out1 = e.score({"sender": "A", "receiver": "B", "amount": 10, "timestamp": T0.isoformat()})
    rec1 = e.payment_index[out1["transaction_id"]]
    assert rec1["_features"]["sender_tx_count_before"] == 0
    out2 = e.score({"sender": "A", "receiver": "B", "amount": 10,
                    "timestamp": (T0 + timedelta(minutes=1)).isoformat()})
    assert e.payment_index[out2["transaction_id"]]["_features"]["sender_tx_count_before"] == 1


def test_engine_handles_empty_state_and_bad_input():
    e = Engine(load_history=False)
    assert e.overview()["scored_payments"] == 0
    assert e.graph_snapshot()["nodes"] == []
    for bad in ({"sender": "A", "receiver": "B", "amount": float("nan"), "timestamp": T0.isoformat()},
                {"sender": "A", "receiver": "B", "amount": float("inf"), "timestamp": T0.isoformat()},
                {"sender": "", "receiver": "B", "amount": 1, "timestamp": T0.isoformat()}):
        with pytest.raises(EngineError):
            e.score(bad)
    with pytest.raises(EngineError):
        e.account_profile("missing")
