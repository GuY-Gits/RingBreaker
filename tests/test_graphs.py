from datetime import datetime, timedelta

from ringbreaker.graphs.build import IdentityGraph, TransactionGraph

from tests.conftest import build_fixture


def test_add_payment_and_neighbors():
    g, _, t0 = build_fixture()
    assert "B" in g.get_out_neighbors("A", as_of=t0 + timedelta(days=3))
    assert "A" in g.get_in_neighbors("B", as_of=t0 + timedelta(days=3))


def test_graph_updates_on_new_payment():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    g.add_payment("A", "B", 10, t, transaction_id="1")
    assert g.get_out_neighbors("A", as_of=t) == ["B"]
    g.add_payment("A", "C", 5, t + timedelta(hours=1), transaction_id="2")
    assert set(g.get_out_neighbors("A", as_of=t + timedelta(hours=1))) == {"B", "C"}
    assert g.get_out_neighbors("A", as_of=t) == ["B"]


def test_pair_and_account_history():
    g, _, t0 = build_fixture()
    as_of = t0 + timedelta(days=3)
    pair = g.get_pair_history("A", "B", as_of=as_of)
    assert len(pair) == 1
    between = g.get_transactions_between("A", "B", as_of=as_of)
    assert len(between) == 2
    hist = g.get_account_history("A", as_of=as_of)
    assert hist


def test_bounded_subgraph():
    g, _, t0 = build_fixture()
    sub = g.subgraph("Y", hops=1, as_of=t0 + timedelta(days=3))
    nodes = set(sub.nodes())
    assert "X" in nodes and "Z" in nodes
    assert "Q" not in nodes
    two = g.subgraph("Y", hops=2, as_of=t0 + timedelta(days=3))
    assert "Q" in two.nodes()


def test_identity_shared_device_cluster():
    _, ident, t0 = build_fixture()
    cluster = ident.get_shared_device_cluster("device_1", as_of=t0)
    assert set(cluster) == {"R1", "R2", "R3"}
    assert ident.get_identity_neighbours("R1", as_of=t0)
    shared = ident.get_shared_identities("R1", "R2", as_of=t0)
    assert any(item["identity_value"] == "device_1" for item in shared)


def test_optional_identity_fields():
    ident = IdentityGraph()
    ident.add_account_identities("U1", phone="555", ip=None, address="")
    assert ident.get_identities("U1")[0]["identity_type"] == "phone"
    assert ident.get_accounts_for_identity("ip", "x") == []


def test_snapshot_serializable():
    g, ident, t0 = build_fixture()
    snap = g.snapshot(as_of=t0 + timedelta(days=3))
    assert "nodes" in snap and "edges" in snap
    id_snap = ident.snapshot(as_of=t0)
    assert id_snap["nodes"]


def test_identity_graph_helpers_and_as_of_graph():
    ident = IdentityGraph()
    t = datetime(2024, 1, 1)
    ident.add_account_identities(
        "A1", phone="+12345", ip="192.168.1.1", address="123 Main St", observed_at=t
    )
    ident.add_account_identities(
        "A2", phone="+12345", ip="192.168.1.1", address="123 Main St", observed_at=t
    )

    assert set(ident.get_shared_phone_cluster("+12345", as_of=t)) == {"A1", "A2"}
    assert set(ident.get_shared_ip_cluster("192.168.1.1", as_of=t)) == {"A1", "A2"}
    assert set(ident.get_shared_address_cluster("123 Main St", as_of=t)) == {"A1", "A2"}

    nx_graph = ident.as_of_graph(as_of=t)
    assert "A1" in nx_graph
    assert "A2" in nx_graph
    assert nx_graph.has_edge("A1", "phone:+12345")


def test_account_ids_temporal_filtering():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    g.register_account("ACC_EARLY", signup_at=t)
    g.register_account("ACC_LATE", signup_at=t + timedelta(days=10))

    early_ids = g.account_ids(as_of=t)
    assert "ACC_EARLY" in early_ids
    assert "ACC_LATE" not in early_ids

    all_ids = g.account_ids()
    assert "ACC_EARLY" in all_ids and "ACC_LATE" in all_ids
