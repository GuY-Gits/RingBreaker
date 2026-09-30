"""F6: social plausibility for a sender-receiver pair.

For a candidate payment A -> B at time T (use ``as_of=T`` and optionally
``exclude_transaction_id`` for that payment):

- reciprocity: 1 if B has previously sent money to A, else 0.
- shared_neighbour_count: accounts connected to both A and B (undirected),
  excluding A and B themselves.
- first_time_payee: 1 if A has never paid B before T.
- prior_pair_count: number of prior A -> B payments.
- seconds_since_last_pair: time since the previous A -> B payment, or -1 if none.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ringbreaker.graphs.build import TransactionGraph, parse_timestamp


def pair_social_features(
    graph: TransactionGraph,
    sender: str,
    receiver: str,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
) -> dict[str, Any]:
    sender, receiver = str(sender), str(receiver)
    as_of_ts = parse_timestamp(as_of)
    prior = graph.get_pair_history(
        sender, receiver, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    reverse = graph.get_pair_history(
        receiver, sender, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    a_neighbors = set(
        graph.get_neighbors(sender, as_of=as_of, exclude_transaction_id=exclude_transaction_id)
    )
    b_neighbors = set(
        graph.get_neighbors(receiver, as_of=as_of, exclude_transaction_id=exclude_transaction_id)
    )
    a_neighbors.discard(receiver)
    b_neighbors.discard(sender)
    shared = a_neighbors & b_neighbors
    if prior:
        seconds_since = (as_of_ts - prior[-1].timestamp).total_seconds()
        if seconds_since < 0:
            seconds_since = 0.0
    else:
        seconds_since = -1.0
    return {
        "sender": sender,
        "receiver": receiver,
        "reciprocity": int(bool(reverse)),
        "shared_neighbour_count": len(shared),
        "first_time_payee": int(len(prior) == 0),
        "prior_pair_count": len(prior),
        "seconds_since_last_pair": float(seconds_since),
        "reverse_pair_count": len(reverse),
    }
