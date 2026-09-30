"""As-of graph metrics for RingBreaker.

Rewritten from scratch. Useful neighbourhood statistics only — no label-derived
features (no community_fraud_rate, no second_hop_fraud_rate).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
    return {
        "account_id": account_id,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "in_count": in_count,
        "out_count": out_count,
        "in_out_ratio": _safe_div(in_degree, out_degree) if out_degree else (1.0 if in_degree else 0.0),
        "reciprocity": reciprocity,
        "reciprocal_pair_count": reciprocal_pairs,
        "neighbor_count": len(neighbors),
        "second_hop_neighbor_count": len(hop2),
        "bounded_neighbourhood_node_count": neighbourhood.number_of_nodes(),
        "bounded_neighbourhood_edge_count": neighbourhood.number_of_edges(),
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
    directed = graph.get_pair_history(
        sender, receiver, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    reverse = graph.get_pair_history(
        receiver, sender, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    return {
        "graph_shared_neighbour_count": len(shared),
        "pair_edge_count": len(directed),
        "reverse_edge_count": len(reverse),
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
