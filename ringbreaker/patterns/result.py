"""Serializable named-pattern result for Person 4 case files (F8 / F12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx

from ringbreaker.graphs.build import serialize_digraph


def _jsonify(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    return value


@dataclass
class PatternResult:
    pattern_name: str
    members: list[str]
    roles: dict[str, Any]
    subgraph: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "members": list(self.members),
            "roles": _jsonify(self.roles),
            "subgraph": _jsonify(self.subgraph),
            "evidence": _jsonify(self.evidence),
        }


def graph_to_subgraph(graph: nx.MultiDiGraph | nx.Graph) -> dict[str, Any]:
    if isinstance(graph, nx.MultiDiGraph):
        return serialize_digraph(graph)
    snapshot = nx.MultiDiGraph()
    snapshot.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        snapshot.add_edge(u, v, **data)
    return serialize_digraph(snapshot)
