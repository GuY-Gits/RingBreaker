"""
F12 / F16 — Case-file alert JSON.

Assembles pattern, evidence subgraph, timeline, top factors (SHAP-style),
counterfactual, and optional LLM summary placeholder for GET /alerts/{id}.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from explain.counterfactual import compute_counterfactual


class PatternHit(TypedDict, total=False):
    pattern_name: str
    members: List[str]
    roles: Dict[str, str]
    subgraph: Dict[str, Any]


class PaymentEvent(TypedDict, total=False):
    sender: str
    receiver: str
    amount: float
    timestamp: str
    device: Optional[str]
    payment_id: Optional[str]


class Factor(TypedDict):
    name: str
    value: float
    contribution: float


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def build_timeline(
    events: Sequence[PaymentEvent],
    member_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Chronological payment list for case file (F12)."""
    members = set(member_ids or [])
    rows: List[PaymentEvent] = list(events)
    if members:
        rows = [
            e
            for e in rows
            if e.get("sender") in members or e.get("receiver") in members
        ]
    rows.sort(key=lambda e: _parse_ts(str(e.get("timestamp", ""))))
    timeline: List[Dict[str, Any]] = []
    for e in rows:
        timeline.append(
            {
                "timestamp": e.get("timestamp"),
                "sender": e.get("sender"),
                "receiver": e.get("receiver"),
                "amount": e.get("amount"),
                "device": e.get("device"),
                "payment_id": e.get("payment_id"),
            }
        )
    return timeline


def build_subgraph(
    pattern: Optional[PatternHit],
    payment_events: Sequence[PaymentEvent],
) -> Dict[str, Any]:
    """Evidence subgraph: nodes, edges, and member roles (F12)."""
    if pattern and pattern.get("subgraph"):
        return pattern["subgraph"]

    members: List[str] = list(pattern.get("members", [])) if pattern else []
    roles: Dict[str, str] = dict(pattern.get("roles", {})) if pattern else {}
    node_set: set[str] = set(members)

    edges: List[Dict[str, Any]] = []
    for e in payment_events:
        s, r = e.get("sender"), e.get("receiver")
        if not s or not r:
            continue
        if members and s not in node_set and r not in node_set:
            continue
        node_set.add(str(s))
        node_set.add(str(r))
        edges.append(
            {
                "source": str(s),
                "target": str(r),
                "amount": e.get("amount"),
                "timestamp": e.get("timestamp"),
            }
        )

    nodes = [
        {"id": nid, "role": roles.get(nid, "unknown")}
        for nid in sorted(node_set)
    ]
    return {"nodes": nodes, "edges": edges}


def top_factors_from_shap(
    shap_values: Optional[Dict[str, float]],
    *,
    limit: int = 8,
) -> List[Factor]:
    """Rank top contributing features for the alert (F12)."""
    if not shap_values:
        return []
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    factors: List[Factor] = []
    for name, contribution in ranked[:limit]:
        factors.append(
            {
                "name": name,
                "value": 0.0,
                "contribution": round(float(contribution), 4),
            }
        )
    return factors


def assemble_case_file(
    *,
    alert_id: str,
    risk_score: float,
    action: str,
    sub_scores: Dict[str, float],
    pattern: Optional[PatternHit] = None,
    payment_events: Optional[Sequence[PaymentEvent]] = None,
    shap_values: Optional[Dict[str, float]] = None,
    summary: Optional[str] = None,
    feature_values: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Single JSON document returned by GET /alerts/{id}.
    """
    events = list(payment_events or [])
    members = pattern.get("members", []) if pattern else []
    subgraph = build_subgraph(pattern, events)
    timeline = build_timeline(events, members or None)

    factors = top_factors_from_shap(shap_values)
    if feature_values and factors:
        for f in factors:
            if f["name"] in feature_values:
                f["value"] = round(float(feature_values[f["name"]]), 4)

    counterfactual = compute_counterfactual(sub_scores, risk_score, action)

    pattern_name = pattern.get("pattern_name") if pattern else None
    roles = pattern.get("roles", {}) if pattern else {}

    case: Dict[str, Any] = {
        "alert_id": alert_id,
        "risk_score": round(float(risk_score), 4),
        "action": action,
        "sub_scores": {k: round(float(v), 4) for k, v in sub_scores.items()},
        "pattern": pattern_name,
        "members": list(members),
        "roles": dict(roles),
        "subgraph": subgraph,
        "timeline": timeline,
        "top_factors": factors,
        "counterfactual": counterfactual,
        "summary": summary,
    }
    return case
