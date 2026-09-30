from datetime import timedelta

from ringbreaker.graph_metrics import account_graph_metrics, pair_graph_metrics

from tests.conftest import build_fixture


def test_account_metrics_degrees():
    g, _, t0 = build_fixture()
    as_of = t0 + timedelta(days=3)
    m_metrics = account_graph_metrics(g, "M", as_of=as_of)
    assert m_metrics["in_degree"] >= 4
    assert m_metrics["out_degree"] == 0
    assert m_metrics["in_out_ratio"] == 1.0
    a_metrics = account_graph_metrics(g, "A", as_of=as_of)
    assert a_metrics["reciprocal_pair_count"] >= 1
    assert a_metrics["neighbor_count"] >= 1


def test_pair_shared_neighbors():
    g, _, t0 = build_fixture()
    as_of = t0 + timedelta(days=3)
    pair = pair_graph_metrics(g, "A", "B", as_of=as_of)
    assert pair["reverse_edge_count"] >= 1
    assert pair["undirected_connected"] == 1


def test_no_label_features_present():
    g, _, t0 = build_fixture()
    metrics = account_graph_metrics(g, "A", as_of=t0 + timedelta(days=3))
    assert "community_fraud_rate" not in metrics
    assert "second_hop_fraud_rate" not in metrics
