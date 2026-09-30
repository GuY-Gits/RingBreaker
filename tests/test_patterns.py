from datetime import timedelta

from ringbreaker.patterns.chain import detect_pass_through_chains
from ringbreaker.patterns.fan_in import detect_fan_in_collectors
from ringbreaker.patterns.loop import detect_closed_loops
from ringbreaker.patterns.star import detect_shared_device_stars

from tests.conftest import build_fixture


def _names(results):
    return {r.pattern_name for r in results}


def test_shared_device_star():
    _, ident, t0 = build_fixture()
    stars = detect_shared_device_stars(ident, as_of=t0, min_accounts=3)
    assert stars
    star = stars[0]
    payload = star.to_dict()
    assert payload["pattern_name"] == "shared_device_star"
    assert set(payload["members"]) == {"R1", "R2", "R3"}
    assert payload["roles"]["identity_value"] == "device_1"
    assert "nodes" in payload["subgraph"] and "edges" in payload["subgraph"]


def test_pass_through_chain():
    g, _, t0 = build_fixture()
    chains = detect_pass_through_chains(g, as_of=t0 + timedelta(days=2))
    members = [tuple(c.members) for c in chains]
    assert any(m == ("X", "Y", "Z", "Q") or set(["X", "Y", "Z", "Q"]).issubset(m) for m in members)
    long = [c for c in chains if set(c.members) >= {"X", "Y", "Z", "Q"}]
    assert long
    assert long[0].roles["source"] == "X"
    assert long[0].roles["destination"] == "Q"
    assert long[0].evidence["transactions"]


def test_fan_in_collector():
    g, _, t0 = build_fixture()
    fans = detect_fan_in_collectors(g, as_of=t0 + timedelta(days=1), min_senders=4)
    assert fans
    fan = next(f for f in fans if f.roles["collector"] == "M")
    assert set(fan.roles["senders"]) >= {"A", "B", "C", "D"}
    assert fan.evidence["sender_count"] >= 4


def test_closed_loop():
    g, _, t0 = build_fixture()
    loops = detect_closed_loops(g, as_of=t0 + timedelta(days=3))
    assert loops
    found = False
    for loop in loops:
        if set(loop.members) == {"L1", "L2", "L3"}:
            found = True
            assert loop.pattern_name == "closed_loop"
            assert loop.evidence["cycle_length"] == 3
    assert found


def test_pattern_output_schema():
    g, ident, t0 = build_fixture()
    as_of = t0 + timedelta(days=3)
    samples = [
        detect_shared_device_stars(ident, as_of=as_of)[0].to_dict(),
        detect_pass_through_chains(g, as_of=as_of)[0].to_dict(),
        detect_fan_in_collectors(g, as_of=as_of)[0].to_dict(),
        detect_closed_loops(g, as_of=as_of)[0].to_dict(),
    ]
    for payload in samples:
        assert set(payload) == {"pattern_name", "members", "roles", "subgraph", "evidence"}
        assert isinstance(payload["members"], list)
        assert isinstance(payload["roles"], dict)
        assert "nodes" in payload["subgraph"]
        assert "edges" in payload["subgraph"]


def test_shared_device_star_negative():
    from ringbreaker.graphs.build import IdentityGraph
    ident = IdentityGraph()
    # Negative 1: Each account has a unique device
    ident.add_account_identities("U1", device="dev_1")
    ident.add_account_identities("U2", device="dev_2")
    ident.add_account_identities("U3", device="dev_3")
    assert detect_shared_device_stars(ident, min_accounts=2) == []

    # Negative 2: 2 accounts share a device, but min_accounts=3
    ident.add_account_identities("U4", device="dev_shared")
    ident.add_account_identities("U5", device="dev_shared")
    assert detect_shared_device_stars(ident, min_accounts=3) == []


def test_pass_through_chain_negative():
    from datetime import datetime
    from ringbreaker.graphs.build import TransactionGraph
    t = datetime(2024, 1, 1, 12, 0, 0)

    # Negative 1: decreasing timestamps (money cannot flow backwards in time)
    g1 = TransactionGraph()
    g1.add_payment("A", "B", 100, t, transaction_id="tx1")
    g1.add_payment("B", "C", 95, t - timedelta(minutes=5), transaction_id="tx2")
    assert detect_pass_through_chains(g1, as_of=t + timedelta(hours=1)) == []

    # Negative 2: duration exceeds window (15 minutes)
    g2 = TransactionGraph()
    g2.add_payment("A", "B", 100, t, transaction_id="tx3")
    g2.add_payment("B", "C", 95, t + timedelta(hours=2), transaction_id="tx4")
    assert detect_pass_through_chains(g2, as_of=t + timedelta(hours=3), window=timedelta(minutes=15)) == []

    # Negative 3: amount mismatch (retention too high, not pass-through)
    g3 = TransactionGraph()
    g3.add_payment("A", "B", 1000, t, transaction_id="tx5")
    g3.add_payment("B", "C", 10, t + timedelta(minutes=2), transaction_id="tx6")
    assert detect_pass_through_chains(g3, as_of=t + timedelta(hours=1), amount_slack=0.15) == []


def test_fan_in_collector_negative():
    from datetime import datetime
    from ringbreaker.graphs.build import TransactionGraph
    t = datetime(2024, 1, 1, 12, 0, 0)

    # Negative 1: only 2 senders when min_senders=4
    g1 = TransactionGraph()
    g1.add_payment("S1", "M", 50, t, transaction_id="f1")
    g1.add_payment("S2", "M", 50, t + timedelta(minutes=10), transaction_id="f2")
    assert detect_fan_in_collectors(g1, as_of=t + timedelta(hours=1), min_senders=4) == []

    # Negative 2: same sender paying 5 times (only 1 unique sender)
    g2 = TransactionGraph()
    for i in range(5):
        g2.add_payment("S1", "M", 50, t + timedelta(minutes=i), transaction_id=f"rep{i}")
    assert detect_fan_in_collectors(g2, as_of=t + timedelta(hours=1), min_senders=4) == []

    # Negative 3: 4 senders spread out across 10 days (window 24 hours)
    g3 = TransactionGraph()
    for i, s in enumerate(["S1", "S2", "S3", "S4"]):
        g3.add_payment(s, "M", 50, t + timedelta(days=i * 2), transaction_id=f"spread{i}")
    assert detect_fan_in_collectors(g3, as_of=t + timedelta(days=10), min_senders=4, window=timedelta(hours=24)) == []


def test_closed_loop_negative():
    from datetime import datetime
    from ringbreaker.graphs.build import TransactionGraph
    t = datetime(2024, 1, 1, 12, 0, 0)

    # Negative: Directed Acyclic Graph (DAG) with no return edge
    g = TransactionGraph()
    g.add_payment("A", "B", 10, t, transaction_id="d1")
    g.add_payment("B", "C", 10, t + timedelta(minutes=1), transaction_id="d2")
    g.add_payment("C", "D", 10, t + timedelta(minutes=2), transaction_id="d3")
    assert detect_closed_loops(g, as_of=t + timedelta(hours=1)) == []


def test_detect_all_patterns_comprehensive():
    from ringbreaker.patterns import detect_all_patterns
    g, ident, _ = build_fixture()
    # Call without as_of (uses full graph)
    all_res = detect_all_patterns(g, ident)
    names = {r.pattern_name for r in all_res}
    assert "shared_device_star" in names
    assert "pass_through_chain" in names
    assert "fan_in_collector" in names
    assert "closed_loop" in names

    # Call with identity_graph=None
    tx_only = detect_all_patterns(g, identity_graph=None)
    tx_names = {r.pattern_name for r in tx_only}
    assert "shared_device_star" not in tx_names
    assert "pass_through_chain" in tx_names
