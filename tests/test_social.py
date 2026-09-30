from datetime import datetime, timedelta

from ringbreaker.features.social import pair_social_features
from ringbreaker.graphs.build import TransactionGraph

from tests.conftest import build_fixture


def test_reciprocity_and_first_time_payee():
    g, _, t0 = build_fixture()
    before_return = pair_social_features(g, "B", "A", as_of=t0 + timedelta(minutes=1))
    assert before_return["reciprocity"] == 1
    assert before_return["first_time_payee"] == 1
    after = pair_social_features(g, "A", "B", as_of=t0 + timedelta(hours=3))
    assert after["first_time_payee"] == 0
    assert after["prior_pair_count"] == 1
    assert after["reciprocity"] == 1


def test_shared_neighbours():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    g.add_payment("A", "C", 1, t, transaction_id="1")
    g.add_payment("B", "C", 1, t, transaction_id="2")
    feats = pair_social_features(g, "A", "B", as_of=t)
    assert feats["shared_neighbour_count"] == 1
    assert feats["first_time_payee"] == 1
    assert feats["reciprocity"] == 0


def test_seconds_since_last_pair():
    g = TransactionGraph()
    t = datetime(2024, 1, 1, 12, 0, 0)
    g.add_payment("A", "B", 1, t, transaction_id="1")
    later = t + timedelta(hours=5)
    feats = pair_social_features(g, "A", "B", as_of=later)
    assert abs(feats["seconds_since_last_pair"] - 5 * 3600) < 1e-6
