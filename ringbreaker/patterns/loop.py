"""F8: closed-loop detector.

Bounded directed cycles of length 3–4 (accounts), using DFS from each node on
the as-of payment graph. Timestamp evidence is attached; cycle membership does
not require a perfect time order around the loop.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

import networkx as nx

from ringbreaker.graphs.build import Payment, TransactionGraph
from ringbreaker.patterns.result import PatternResult, graph_to_subgraph

DEFAULT_MIN_LENGTH = 3
DEFAULT_MAX_LENGTH = 4
MAX_CYCLES = 200


def detect_closed_loops(
    graph: TransactionGraph,
    as_of: datetime | str | None = None,
    exclude_transaction_id: str | None = None,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[PatternResult]:
    payments = graph.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
    adjacency: dict[str, list[Payment]] = {}
    for payment in payments:
        adjacency.setdefault(payment.sender, []).append(payment)

    results: list[PatternResult] = []
    seen: set[tuple[str, ...]] = set()
    for start in sorted(adjacency):
        for cycle_payments in _cycles_from(
            start, adjacency, min_length=min_length, max_length=max_length
        ):
            members = _canonical(cycle_payments)
            if members in seen:
                continue
            seen.add(members)
            results.append(_to_result(list(members), cycle_payments))
            if len(results) >= MAX_CYCLES:
                return results
    return results


def _cycles_from(
    start: str,
    adjacency: dict[str, list[Payment]],
    min_length: int,
    max_length: int,
) -> Iterator[list[Payment]]:
    stack: list[tuple[str, list[Payment], set[str]]] = [(start, [], {start})]
    while stack:
        node, path, used = stack.pop()
        for payment in adjacency.get(node, []):
            nxt = payment.receiver
            new_path = path + [payment]
            hops = len(new_path)
            if nxt == start and min_length <= hops <= max_length:
                yield new_path
                continue
            if hops >= max_length:
                continue
            if nxt in used:
                continue
            # Only expand cycles whose smallest node is the start, to cut duplicates.
            if nxt < start:
                continue
            stack.append((nxt, new_path, used | {nxt}))


def _canonical(path: list[Payment]) -> tuple[str, ...]:
    nodes = [path[0].sender] + [p.receiver for p in path[:-1]]
    rotated = min(tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes)))
    return rotated


def _to_result(members: list[str], path: list[Payment]) -> PatternResult:
    sub = nx.MultiDiGraph()
    for account in members:
        sub.add_node(account, kind="account", role="loop_member")
    evidence_tx = []
    for payment in path:
        sub.add_edge(
            payment.sender,
            payment.receiver,
            transaction_id=payment.transaction_id,
            timestamp=payment.timestamp,
            amount=payment.amount,
        )
        evidence_tx.append(payment.to_dict())
    return PatternResult(
        pattern_name="closed_loop",
        members=members,
        roles={"loop_members": members},
        subgraph=graph_to_subgraph(sub),
        evidence={
            "cycle_length": len(members),
            "transactions": evidence_tx,
        },
    )
