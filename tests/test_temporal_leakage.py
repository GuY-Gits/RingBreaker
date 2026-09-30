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


def test_future_payment_does_not_change_chain_at_t():
    from ringbreaker.patterns.chain import detect_pass_through_chains
    g = TransactionGraph()
    t = datetime(2024, 3, 1, 12, 0, 0)
    g.add_payment("A", "B", 100, t, transaction_id="c1")
    g.add_payment("B", "C", 95, t + timedelta(minutes=2), transaction_id="c2")
    g.add_payment("C", "D", 90, t + timedelta(minutes=4), transaction_id="c3")
    as_of = t + timedelta(minutes=10)
    before = [r.to_dict() for r in detect_pass_through_chains(g, as_of=as_of)]
    # Add future extension of the chain
    g.add_payment("D", "E", 85, as_of + timedelta(days=1), transaction_id="future_hop")
    after = [r.to_dict() for r in detect_pass_through_chains(g, as_of=as_of)]
    assert before == after


def test_future_payment_does_not_change_loop_at_t():
    from ringbreaker.patterns.loop import detect_closed_loops
    g = TransactionGraph()
    t = datetime(2024, 3, 1, 12, 0, 0)
    g.add_payment("L1", "L2", 50, t, transaction_id="l1")
    g.add_payment("L2", "L3", 50, t + timedelta(minutes=1), transaction_id="l2")
    g.add_payment("L3", "L1", 50, t + timedelta(minutes=2), transaction_id="l3")
    as_of = t + timedelta(hours=1)
    before = [r.to_dict() for r in detect_closed_loops(g, as_of=as_of)]
    # Add future payment forming another loop
    g.add_payment("L3", "L4", 50, as_of + timedelta(days=1), transaction_id="future_l4")
    g.add_payment("L4", "L1", 50, as_of + timedelta(days=1, minutes=1), transaction_id="future_l1")
    after = [r.to_dict() for r in detect_closed_loops(g, as_of=as_of)]
    assert before == after


def test_future_account_registration_does_not_leak_into_as_of_t():
    from ringbreaker.features.lifelike import account_lifelikeness
    from ringbreaker.features.lockstep import detect_lockstep
    g = TransactionGraph()
    t = datetime(2024, 3, 1, 12, 0, 0)
    g.register_account("EXISTING", signup_at=t - timedelta(days=10))
    g.add_payment("EXISTING", "OTHER", 10, t, transaction_id="ex1")

    life_before = account_lifelikeness(g, "EXISTING", as_of=t)
    lockstep_before = detect_lockstep(g, as_of=t)

    # Register an account in the future
    g.register_account("FUTURE_ACC", signup_at=t + timedelta(days=30))
    g.add_payment("FUTURE_ACC", "EXISTING", 100, t + timedelta(days=31), transaction_id="fut_tx")

    life_after = account_lifelikeness(g, "EXISTING", as_of=t)
    lockstep_after = detect_lockstep(g, as_of=t)
    assert life_before == life_after
    assert lockstep_before == lockstep_after
    assert "FUTURE_ACC" not in g.account_ids(as_of=t)


def test_future_identity_link_does_not_leak_into_as_of_t():
    from ringbreaker.graphs.build import IdentityGraph
    from ringbreaker.patterns.star import detect_shared_device_stars
    ident = IdentityGraph()
    t = datetime(2024, 3, 1, 12, 0, 0)
    ident.add_account_identities("U1", device="dev_common", observed_at=t - timedelta(days=5))
    ident.add_account_identities("U2", device="dev_common", observed_at=t - timedelta(days=5))
    # As of t, only 2 accounts share dev_common (so min_accounts=3 finds 0 stars)
    before_stars = detect_shared_device_stars(ident, as_of=t, min_accounts=3)
    assert before_stars == []
    assert len(ident.get_shared_device_cluster("dev_common", as_of=t)) == 2

    # A 3rd account links the device in the FUTURE
    ident.add_account_identities("U3", device="dev_common", observed_at=t + timedelta(days=10))

    # As of t, still 2 accounts and 0 stars
    after_stars = detect_shared_device_stars(ident, as_of=t, min_accounts=3)
    assert after_stars == []
    assert len(ident.get_shared_device_cluster("dev_common", as_of=t)) == 2

    # But as of future, all 3 accounts are seen and star is detected
    future_stars = detect_shared_device_stars(ident, as_of=t + timedelta(days=15), min_accounts=3)
    assert len(future_stars) == 1
    assert set(future_stars[0].members) == {"U1", "U2", "U3"}
