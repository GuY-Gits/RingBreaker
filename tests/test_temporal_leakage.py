from datetime import datetime, timedelta

from ringbreaker.features.flow import account_flow_features
from ringbreaker.features.lifelike import account_lifelikeness
from ringbreaker.features.social import pair_social_features
from ringbreaker.graph_metrics import account_graph_metrics
from ringbreaker.graphs.build import TransactionGraph
from ringbreaker.patterns.fan_in import detect_fan_in_collectors


def _base_graph():
    g = TransactionGraph()
    t = datetime(2024, 3, 1, 12, 0, 0)
    g.register_account("A", signup_at=t - timedelta(days=30))
    g.register_account("B", signup_at=t - timedelta(days=30))
    g.add_payment("A", "B", 25, t, transaction_id="t0")
    g.add_payment("C", "A", 40, t + timedelta(hours=1), transaction_id="t1")
    g.add_payment("A", "D", 10, t + timedelta(hours=3), transaction_id="t2")
    return g, t


def test_future_payment_does_not_change_flow_at_t():
    g, t = _base_graph()
    as_of = t + timedelta(hours=3)
    before = account_flow_features(g, "A", as_of=as_of)
    g.add_payment("E", "A", 999, as_of + timedelta(days=1), transaction_id="future")
    after = account_flow_features(g, "A", as_of=as_of)
    assert before == after


def test_future_payment_does_not_change_social_at_t():
    g, t = _base_graph()
    as_of = t + timedelta(hours=3)
    before = pair_social_features(g, "A", "B", as_of=as_of)
    g.add_payment("B", "A", 1, as_of + timedelta(days=1), transaction_id="future_rec")
    after = pair_social_features(g, "A", "B", as_of=as_of)
    assert before == after


def test_future_payment_does_not_change_lifelike_or_metrics():
    g, t = _base_graph()
    as_of = t + timedelta(hours=3)
    life_before = account_lifelikeness(g, "A", as_of=as_of)
    met_before = account_graph_metrics(g, "A", as_of=as_of)
    g.add_payment("A", "Z", 50, as_of + timedelta(days=1), transaction_id="future_out")
    assert account_lifelikeness(g, "A", as_of=as_of) == life_before
    assert account_graph_metrics(g, "A", as_of=as_of) == met_before


def test_future_payment_does_not_change_fan_in_at_t():
    g = TransactionGraph()
    t = datetime(2024, 3, 1, 12, 0, 0)
    for i, sender in enumerate(["A", "B", "C", "D"]):
        g.add_payment(sender, "M", 10, t + timedelta(minutes=i), transaction_id=f"in{i}")
    as_of = t + timedelta(hours=1)
    before = [r.to_dict() for r in detect_fan_in_collectors(g, as_of=as_of, min_senders=4)]
    g.add_payment("E", "M", 10, as_of + timedelta(days=1), transaction_id="later")
    after = [r.to_dict() for r in detect_fan_in_collectors(g, as_of=as_of, min_senders=4)]
    assert before == after
