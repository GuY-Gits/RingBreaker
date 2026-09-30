"""As-of graph metrics for RingBreaker.

Rewritten from scratch. Useful neighbourhood statistics only — no label-derived
features (no community_fraud_rate, no second_hop_fraud_rate).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import networkx as nx

from ringbreaker.graphs.build import TransactionGraph


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def account_graph_metrics(
    graph: TransactionGraph,
    account_id: str,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
    hops: int = 2,
) -> dict[str, Any]:
    """Degree, in/out ratio, reciprocity, and bounded neighbourhood size.

    All values use payments with timestamp <= ``as_of`` (and optionally skip one
    transaction id so a candidate payment is not scored against itself).
    Includes localized PageRank and clustering coefficient without label leakage.
    """
    account_id = str(account_id)
    incoming = graph.incoming_payments(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    outgoing = graph.outgoing_payments(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    in_degree = len({p.sender for p in incoming})
    out_degree = len({p.receiver for p in outgoing})
    in_count = len(incoming)
    out_count = len(outgoing)
    in_amount = sum(p.amount for p in incoming)
    out_amount = sum(p.amount for p in outgoing)

    neighbors = graph.get_neighbors(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    reciprocal_pairs = 0
    for neighbor in neighbors:
        a_to_b = graph.get_pair_history(
            account_id, neighbor, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
        b_to_a = graph.get_pair_history(
            neighbor, account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
        if a_to_b and b_to_a:
            reciprocal_pairs += 1
    reciprocity = _safe_div(reciprocal_pairs, len(neighbors))
    hop1 = set(neighbors)
    hop2: set[str] = set()
    for neighbor in hop1:
        for extra in graph.get_neighbors(
            neighbor, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        ):
            if extra != account_id and extra not in hop1:
                hop2.add(extra)
    neighbourhood = graph.subgraph(
        account_id, hops=hops, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )

    pagerank = 0.0
    clustering = 0.0
    if neighbourhood.number_of_nodes() > 0:
        simple_di = nx.DiGraph()
        for u, v, data in neighbourhood.edges(data=True):
            w = float(data.get("amount", 1.0))
            if simple_di.has_edge(u, v):
                simple_di[u][v]["weight"] += w
            else:
                simple_di.add_edge(u, v, weight=w)
        for node in neighbourhood.nodes():
            if node not in simple_di:
                simple_di.add_node(node)
        try:
            pr_dict = nx.pagerank(simple_di, weight="weight")
            pagerank = float(pr_dict.get(account_id, 0.0))
        except Exception:
            pagerank = 1.0 / max(1, neighbourhood.number_of_nodes())
        try:
            simple_undirected = nx.Graph(simple_di)
            clustering = float(nx.clustering(simple_undirected, account_id))
        except Exception:
            clustering = 0.0

    return {
        "account_id": account_id,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": in_degree + out_degree,
        "in_count": in_count,
        "out_count": out_count,
        "total_count": in_count + out_count,
        "in_amount": float(in_amount),
        "out_amount": float(out_amount),
        "in_out_ratio": _safe_div(in_degree, out_degree) if out_degree else (1.0 if in_degree else 0.0),
        "in_out_amount_ratio": _safe_div(in_amount, out_amount) if out_amount else (1.0 if in_amount else 0.0),
        "reciprocity": reciprocity,
        "reciprocal_pair_count": reciprocal_pairs,
        "neighbor_count": len(neighbors),
        "second_hop_neighbor_count": len(hop2),
        "bounded_neighbourhood_node_count": neighbourhood.number_of_nodes(),
        "bounded_neighbourhood_edge_count": neighbourhood.number_of_edges(),
        "pagerank": pagerank,
        "clustering_coefficient": clustering,
    }


def pair_graph_metrics(
    graph: TransactionGraph,
    sender: str,
    receiver: str,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
) -> dict[str, Any]:
    """Pair-level graph statistics Person 1 can concatenate onto social features."""
    sender, receiver = str(sender), str(receiver)
    a_neighbors = set(
        graph.get_neighbors(sender, as_of=as_of, exclude_transaction_id=exclude_transaction_id)
    )
    b_neighbors = set(
        graph.get_neighbors(receiver, as_of=as_of, exclude_transaction_id=exclude_transaction_id)
    )
    a_neighbors.discard(receiver)
    b_neighbors.discard(sender)
    shared = a_neighbors & b_neighbors
    union = a_neighbors | b_neighbors
    directed = graph.get_pair_history(
        sender, receiver, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    reverse = graph.get_pair_history(
        receiver, sender, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    return {
        "graph_shared_neighbour_count": len(shared),
        "graph_shared_neighbour_jaccard": len(shared) / len(union) if union else 0.0,
        "pair_edge_count": len(directed),
        "pair_total_amount": float(sum(p.amount for p in directed)),
        "reverse_edge_count": len(reverse),
        "reverse_total_amount": float(sum(p.amount for p in reverse)),
        "undirected_connected": int(bool(directed or reverse)),
    }


def neighbourhood_connectivity(
    graph: TransactionGraph,
    account_id: str,
    as_of: datetime | str,
    hops: int = 2,
    exclude_transaction_id: str | None = None,
) -> dict[str, Any]:
    """Density of the bounded as-of neighbourhood (pattern-support statistic)."""
    sub = graph.subgraph(
        account_id, hops=hops, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    n = sub.number_of_nodes()
    e = sub.number_of_edges()
    simple = sub.to_undirected()
    possible = n * (n - 1) / 2 if n > 1 else 0.0
    density = _safe_div(simple.number_of_edges(), possible) if possible else 0.0
    return {
        "account_id": str(account_id),
        "hops": hops,
        "node_count": n,
        "directed_edge_count": e,
        "undirected_density": density,
    }
