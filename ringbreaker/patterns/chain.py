"""F8: pass-through chain detector.

Looks for money hopping through 3–6 accounts with strictly increasing timestamps
inside a bounded window (default 6 hours), each hop forwarding 85–100% of the
previous amount. Wider windows mostly match ordinary traffic. Search is a
bounded DFS from each account; it does not enumerate the whole graph.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from typing import Iterator

import networkx as nx

from ringbreaker.graphs.build import Payment, TransactionGraph
from ringbreaker.patterns.result import PatternResult, graph_to_subgraph

DEFAULT_MIN_ACCOUNTS = 3
DEFAULT_MAX_ACCOUNTS = 6
DEFAULT_WINDOW = timedelta(hours=6)
DEFAULT_MIN_HOP_RATIO = 0.85  # each hop forwards 85-100% of what it received
MAX_PATHS_PER_SOURCE = 50


def detect_pass_through_chains(
    graph: TransactionGraph,
    as_of: datetime | str | None = None,
    exclude_transaction_id: str | None = None,
    min_accounts: int = DEFAULT_MIN_ACCOUNTS,
    max_accounts: int = DEFAULT_MAX_ACCOUNTS,
    window: timedelta = DEFAULT_WINDOW,
    amount_slack: float | None = 1.0 - DEFAULT_MIN_HOP_RATIO,
    allow_split: bool = False,
    maximal_only: bool = False,
) -> list[PatternResult]:
    """Detect ordered payment paths A -> B -> ... of length min..max accounts."""
    payments = graph.payments(as_of=as_of, exclude_transaction_id=exclude_transaction_id)
    by_sender: dict[str, list[Payment]] = {}
    for payment in payments:
        by_sender.setdefault(payment.sender, []).append(payment)
    for bucket in by_sender.values():
        bucket.sort(key=lambda p: p.timestamp)

    results: list[PatternResult] = []
    seen: set[tuple[str, ...]] = set()
    sources = sorted({p.sender for p in payments})
    for source in sources:
        found = 0
        for path_payments in _walk(
            source,
            by_sender,
            min_hops=min_accounts - 1,
            max_hops=max_accounts - 1,
            window=window,
            amount_slack=amount_slack,
            allow_split=allow_split,
        ):
            members = _members(path_payments)
            key = tuple(members)
            if key in seen:
                continue
            seen.add(key)
            results.append(_to_result(path_payments, window, as_of))
            found += 1
            if found >= MAX_PATHS_PER_SOURCE:
                break
    if maximal_only and results:
        all_members = [tuple(r.members) for r in results]
        filtered = []
        for r in results:
            m = tuple(r.members)
            # Check if m is a strict sub-slice of any other chain
            is_sub = False
            for other in all_members:
                if len(other) > len(m):
                    for i in range(len(other) - len(m) + 1):
                        if other[i : i + len(m)] == m:
                            is_sub = True
                            break
                if is_sub:
                    break
            if not is_sub:
                filtered.append(r)
        return filtered
    return results


def _walk(
    source: str,
    by_sender: dict[str, list[Payment]],
    min_hops: int,
    max_hops: int,
    window: timedelta,
    amount_slack: float | None,
    allow_split: bool = True,
) -> Iterator[list[Payment]]:
    stack: list[tuple[list[Payment], set[str]]] = []
    for first in by_sender.get(source, []):
        stack.append(([first], {source, first.receiver}))
    while stack:
        path, used = stack.pop()
        if min_hops <= len(path) <= max_hops:
            yield path
        if len(path) >= max_hops:
            continue
        last = path[-1]
        start_ts = path[0].timestamp
        for nxt in by_sender.get(last.receiver, []):
            if nxt.timestamp < last.timestamp:
                continue
            if nxt.timestamp - start_ts > window:
                continue
            if nxt.receiver in used:
                continue
            if not _amount_ok(last, nxt, amount_slack, allow_split=allow_split):
                continue
            stack.append((path + [nxt], used | {nxt.receiver}))


def _amount_ok(prev: Payment, nxt: Payment, slack: float | None, allow_split: bool = False) -> bool:
    """Laundered money only shrinks along a chain (mules skim a cut), so the
    next hop must forward between (1 - slack) and 100% of the previous amount.
    Symmetric tolerance matched far too much ordinary traffic. ``allow_split``
    also accepts a hop forwarding a part (>= 35%) of the amount."""
    if prev.amount <= 0 or nxt.amount <= 0:
        return False
    if slack is None:
        return True
    min_factor = min(0.35, 1.0 - slack) if allow_split else (1.0 - slack)
    return min_factor * prev.amount - 1e-9 <= nxt.amount <= prev.amount + 1e-9


def _members(path: list[Payment]) -> list[str]:
    accounts = [path[0].sender]
    for payment in path:
        accounts.append(payment.receiver)
    return accounts


def _to_result(
    path: list[Payment], window: timedelta, as_of: Any = None
) -> PatternResult:
    members = _members(path)
    roles = {
        "source": members[0],
        "intermediary": members[1:-1],
        "destination": members[-1],
    }
    sub = nx.MultiDiGraph()
    for account in members:
        role = "source" if account == members[0] else ("destination" if account == members[-1] else "intermediary")
        sub.add_node(account, kind="account", role=role)
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
    span = (path[-1].timestamp - path[0].timestamp).total_seconds()
    return PatternResult(
        pattern_name="pass_through_chain",
        members=members,
        roles=roles,
        subgraph=graph_to_subgraph(sub),
        evidence={
            "hop_count": len(path),
            "account_count": len(members),
            "window_seconds": window.total_seconds(),
            "duration_seconds": span,
            "transactions": evidence_tx,
            "as_of": as_of,
        },
    )
