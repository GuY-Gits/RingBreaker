"""F8: shared-device (identity-fragment) star detector.

A star here is many accounts linked to one identity fragment, especially a
device. Detection is structural only — a shared device is not a fraud ring.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import networkx as nx

from ringbreaker.graphs.build import IdentityGraph, identity_node_id
from ringbreaker.patterns.result import PatternResult, graph_to_subgraph

DEFAULT_MIN_ACCOUNTS = 3


def detect_shared_device_stars(
    identity_graph: IdentityGraph,
    as_of: datetime | str | None = None,
    min_accounts: int = DEFAULT_MIN_ACCOUNTS,
    identity_type: str = "device",
) -> list[PatternResult]:
    """Return one result per identity value with >= ``min_accounts`` accounts."""
    results: list[PatternResult] = []
    for cluster in identity_graph.identity_clusters(
        identity_type=identity_type, min_accounts=min_accounts, as_of=as_of
    ):
        results.append(_star_from_cluster(cluster, as_of))
    return results


def _star_from_cluster(
    cluster: dict[str, Any],
    as_of: datetime | str | None,
) -> PatternResult:
    itype = cluster["identity_type"]
    ivalue = cluster["identity_value"]
    members = sorted(cluster["members"])
    ident_id = identity_node_id(itype, ivalue)
    sub = nx.Graph()
    sub.add_node(
        ident_id,
        kind="identity",
        identity_type=itype,
        identity_value=ivalue,
        role="hub",
    )
    for account in members:
        sub.add_node(account, kind="account", role="member")
        sub.add_edge(account, ident_id, identity_type=itype)
    roles: dict[str, Any] = {
        "hub": ident_id,
        "identity_type": itype,
        "identity_value": ivalue,
        "member_accounts": members,
    }
    return PatternResult(
        pattern_name="shared_device_star" if itype == "device" else f"shared_{itype}_star",
        members=members,
        roles=roles,
        subgraph=graph_to_subgraph(sub),
        evidence={
            "account_count": len(members),
            "min_accounts_threshold": True,
            "as_of": as_of,
        },
    )
