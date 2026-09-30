"""F8: fan-in collector detector.

Many distinct senders pay one account inside a time window. The sender-count
threshold is configurable (default 4, matching a typical planted collector).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import networkx as nx

from ringbreaker.graphs.build import Payment, TransactionGraph
from ringbreaker.patterns.result import PatternResult, graph_to_subgraph

DEFAULT_MIN_SENDERS = 4
DEFAULT_WINDOW = timedelta(hours=24)


def detect_fan_in_collectors(
    graph: TransactionGraph,
    as_of: datetime | str | None = None,
    exclude_transaction_id: str | None = None,
    min_senders: int = DEFAULT_MIN_SENDERS,
    window: timedelta = DEFAULT_WINDOW,
) -> list[PatternResult]:
    """Flag accounts that received payments from >= ``min_senders`` in ``window``.

    For each candidate collector, the tightest window that still covers
    ``min_senders`` distinct senders is reported (sliding over incoming txs).
    """
    payments = graph.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
    incoming: dict[str, list[Payment]] = {}
    for payment in payments:
        incoming.setdefault(payment.receiver, []).append(payment)
    results: list[PatternResult] = []
    for collector, txs in incoming.items():
        txs = sorted(txs, key=lambda p: p.timestamp)
        hit = _best_window(txs, min_senders, window)
        if hit is None:
            continue
        window_txs, senders = hit
        results.append(_to_result(collector, window_txs, senders, min_senders, window))
    return results


def _best_window(
    txs: list[Payment],
    min_senders: int,
    window: timedelta,
) -> tuple[list[Payment], list[str]] | None:
    best: tuple[list[Payment], list[str]] | None = None
    best_span: float | None = None
    left = 0
    for right, payment in enumerate(txs):
        while txs[right].timestamp - txs[left].timestamp > window:
            left += 1
        slice_txs = txs[left : right + 1]
        senders: list[str] = []
        seen: set[str] = set()
        for item in slice_txs:
            if item.sender not in seen:
                seen.add(item.sender)
                senders.append(item.sender)
        if len(senders) >= min_senders:
            span = (slice_txs[-1].timestamp - slice_txs[0].timestamp).total_seconds()
            if best_span is None or span < best_span:
                best_span = span
                best = (slice_txs, senders)
    return best


def _to_result(
    collector: str,
    txs: list[Payment],
    senders: list[str],
    min_senders: int,
    window: timedelta,
) -> PatternResult:
    members = [collector] + [s for s in senders if s != collector]
    sub = nx.MultiDiGraph()
    sub.add_node(collector, kind="account", role="collector")
    for sender in senders:
        sub.add_node(sender, kind="account", role="sender")
    evidence_tx = []
    for payment in txs:
        if payment.sender in senders:
            sub.add_edge(
                payment.sender,
                payment.receiver,
                transaction_id=payment.transaction_id,
                timestamp=payment.timestamp,
                amount=payment.amount,
            )
            evidence_tx.append(payment.to_dict())
    span = (txs[-1].timestamp - txs[0].timestamp).total_seconds() if txs else 0.0
    roles: dict[str, Any] = {
        "collector": collector,
        "senders": senders,
    }
    return PatternResult(
        pattern_name="fan_in_collector",
        members=members,
        roles=roles,
        subgraph=graph_to_subgraph(sub),
        evidence={
            "sender_count": len(senders),
            "min_senders": min_senders,
            "window_seconds": window.total_seconds(),
            "duration_seconds": span,
            "incoming_amount": sum(p.amount for p in txs),
            "transactions": evidence_tx,
        },
    )
