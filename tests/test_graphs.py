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
