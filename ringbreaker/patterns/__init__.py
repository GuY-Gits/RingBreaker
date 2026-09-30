"""F8 named pattern detectors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ringbreaker.patterns.chain import detect_pass_through_chains
from ringbreaker.patterns.fan_in import detect_fan_in_collectors
from ringbreaker.patterns.loop import detect_closed_loops
from ringbreaker.patterns.result import PatternResult
from ringbreaker.patterns.star import detect_shared_device_stars


def detect_all_patterns(
    transaction_graph,
    identity_graph=None,
    as_of=None,
    exclude_transaction_id=None,
) -> list[PatternResult]:
    """Run all four detectors. Person 4 can serialize each result with ``to_dict()``."""
    found: list[PatternResult] = []
    if identity_graph is not None:
        found.extend(
            detect_shared_device_stars(identity_graph, as_of=as_of)
        )
    if transaction_graph is not None:
        found.extend(
            detect_pass_through_chains(
                transaction_graph, as_of=as_of, exclude_transaction_id=exclude_transaction_id
            )
        )
        found.extend(
            detect_fan_in_collectors(
                transaction_graph, as_of=as_of, exclude_transaction_id=exclude_transaction_id
            )
        )
        found.extend(
            detect_closed_loops(
                transaction_graph, as_of=as_of, exclude_transaction_id=exclude_transaction_id
            )
        )
    return found


def detect_for_payment(
    transaction_graph,
    sender: str,
    receiver: str,
    identity_graph=None,
    as_of=None,
    exclude_transaction_id=None,
) -> Optional[Dict[str, Any]]:
    """Detect if either sender or receiver is part of a named pattern."""
    all_patterns = detect_all_patterns(
        transaction_graph=transaction_graph,
        identity_graph=identity_graph,
        as_of=as_of,
        exclude_transaction_id=exclude_transaction_id,
    )
    s, r = str(sender), str(receiver)
    for p in all_patterns:
        if s in p.members or r in p.members:
            return p.to_dict()
    return None


__all__ = [
    "PatternResult",
    "detect_all_patterns",
    "detect_for_payment",
    "detect_shared_device_stars",
    "detect_pass_through_chains",
    "detect_fan_in_collectors",
    "detect_closed_loops",
]
