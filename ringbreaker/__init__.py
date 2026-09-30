from ringbreaker.features.flow import account_flow_features
from ringbreaker.features.graph_metrics import (
    account_graph_metrics,
    neighbourhood_connectivity,
    pair_graph_metrics,
)
from ringbreaker.features.lifelike import account_lifelikeness
from ringbreaker.features.lockstep import account_lockstep_features, detect_lockstep
from ringbreaker.features.social import pair_social_features
from ringbreaker.graphs.build import IdentityGraph, Payment, TransactionGraph, parse_timestamp
from ringbreaker.patterns import (
    PatternResult,
    detect_all_patterns,
    detect_closed_loops,
    detect_fan_in_collectors,
    detect_pass_through_chains,
    detect_shared_device_stars,
)

__all__ = [
    "IdentityGraph",
    "TransactionGraph",
    "Payment",
    "parse_timestamp",
    "account_graph_metrics",
    "pair_graph_metrics",
    "neighbourhood_connectivity",
    "account_flow_features",
    "pair_social_features",
    "account_lifelikeness",
    "detect_lockstep",
    "account_lockstep_features",
    "PatternResult",
    "detect_all_patterns",
    "detect_shared_device_stars",
    "detect_pass_through_chains",
    "detect_fan_in_collectors",
    "detect_closed_loops",
    "sender_features",
    "receiver_features",
    "pair_features",
    "graph_features",
]


def sender_features(graph: TransactionGraph, sender: str, as_of, exclude_transaction_id=None) -> dict:
    """Person 1: account-level features for the sender as of ``as_of``."""
    return graph_features(graph, sender, as_of, exclude_transaction_id=exclude_transaction_id)


def receiver_features(graph: TransactionGraph, receiver: str, as_of, exclude_transaction_id=None) -> dict:
    """Person 1: account-level features for the receiver as of ``as_of``."""
    return graph_features(graph, receiver, as_of, exclude_transaction_id=exclude_transaction_id)


def graph_features(graph: TransactionGraph, account_id: str, as_of, exclude_transaction_id=None) -> dict:
    """Combine flow, lifelikeness, and graph metrics for one account."""
    payload = {}
    payload.update(
        account_flow_features(
            graph, account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
    )
    payload.update(
        account_lifelikeness(
            graph, account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
    )
    payload.update(
        account_graph_metrics(
            graph, account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
        )
    )
    return payload


def pair_features(
    graph: TransactionGraph,
    sender: str,
    receiver: str,
    as_of,
    exclude_transaction_id=None,
) -> dict:
    """Person 1: sender-receiver pair features (social + pair graph metrics)."""
    payload = {}
    payload.update(
        pair_social_features(
            graph,
            sender,
            receiver,
            as_of=as_of,
            exclude_transaction_id=exclude_transaction_id,
        )
    )
    payload.update(
        pair_graph_metrics(
            graph,
            sender,
            receiver,
            as_of=as_of,
            exclude_transaction_id=exclude_transaction_id,
        )
    )
    return payload
