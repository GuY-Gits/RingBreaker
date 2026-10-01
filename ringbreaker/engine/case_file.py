"""F12 / F16 / F17 case file, assembled from engine state as of the alert time.

Everything in a case file is traceable to stored facts: the scored payment
record (features, SHAP values, signals), the graph and identity graph as of the
payment timestamp, detected patterns, and analyst actions.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set

from ringbreaker.engine.risk import FEATURE_LABELS, fuse
from ringbreaker.explain.counterfactual import compute_counterfactual
from ringbreaker.graphs.build import parse_timestamp
from ringbreaker.scoring.action import ALLOW_THRESHOLD

if TYPE_CHECKING:  # pragma: no cover
    from ringbreaker.engine.engine import Engine

MAX_CASE_NODES = 45
MAX_TIMELINE = 80


def build(engine: "Engine", alert: Dict[str, Any]) -> Dict[str, Any]:
    rec = alert["_record"]
    ts = parse_timestamp(rec["timestamp"])
    s, r = rec["sender"], rec["receiver"]
    order = {pid: i for i, pid in enumerate(alert["pattern_ids"])}  # already most-specific first
    patterns = sorted((p for p in engine.patterns.values() if p["id"] in order), key=lambda p: order[p["id"]])

    nodes: Set[str] = {s, r}
    for p in patterns:
        if len(nodes) + len(p["members"]) <= MAX_CASE_NODES or p["type"] != "shared_device_star":
            nodes.update(p["members"])
    for party in (s, r):
        for n in engine.graph.get_neighbors(party, as_of=ts)[:10]:
            if len(nodes) < MAX_CASE_NODES:
                nodes.add(n)
    subgraph = subgraph_payload(engine, nodes, ts, highlight={s, r}, include_identity=True,
                                patterns=patterns, trigger=rec["transaction_id"])

    ctx = rec["_context"]
    return {
        "alert_id": alert["alert_id"],
        "status": alert["status"],
        "created_at": alert["created_at"],
        "verdict": alert["verdict"],
        "payment": engine.public_payment(rec),
        "risk_score": rec["overall_risk"],
        "overall_risk": rec["overall_risk"],
        "risk_percent": rec["risk_percent"],
        "action": rec["action"],
        "sub_scores": rec["sub_scores"],
        "signals": rec["signals"],
        "reasons": rec["reasons"],
        "pattern": alert["pattern"],
        "patterns": [_pattern_view(engine, p) for p in patterns],
        "members": sorted({m for p in patterns for m in p["members"]} | {s, r}),
        "roles": _roles(patterns, s, r),
        "suspects": suspects(engine, alert),
        "parties": {
            "sender": engine.account_profile(s, as_of=ts),
            "receiver": engine.account_profile(r, as_of=ts),
        },
        "relationship": ctx["social"],
        "subgraph": subgraph,
        "timeline": timeline(engine, alert, nodes, ts),
        "top_factors": top_factors(rec),
        "counterfactual": counterfactual(engine, rec),
        "summary": summary(rec, patterns),
        "model_version": engine.models.version,
    }


def suspects(engine: "Engine", alert: Dict[str, Any]) -> List[str]:
    """Accounts a confirmation will mark as fraud by default.

    The receiver and detected ring members are suspects. The sender is left
    out when the alert looks like a scam-victim payment (sender warned because
    the relationship, not the sender's behaviour, drove the risk).
    """
    rec = alert["_record"]
    out = [rec["receiver"]]
    sub = rec["sub_scores"]
    victim_like = (
        rec["action"] == "WARN_SENDER"
        and sub["relationship_plausibility"] >= sub["sender_anomaly"]
    )
    if not victim_like:
        out.append(rec["sender"])
    for p in engine.patterns.values():
        if p["id"] in alert["pattern_ids"] and p["type"] != "shared_device_star":
            out.extend(m for m in p["members"] if m != rec["sender"] or not victim_like)
    seen: Set[str] = set()
    return [a for a in out if not (a in seen or seen.add(a))]


def _roles(patterns: List[Dict[str, Any]], s: str, r: str) -> Dict[str, List[str]]:
    roles: Dict[str, List[str]] = {s: ["sender"], r: ["receiver"]}
    for p in patterns:
        for acc, role in _member_roles(p).items():
            roles.setdefault(acc, []).append(role)
    return roles


def _member_roles(p: Dict[str, Any]) -> Dict[str, str]:
    t, roles, members = p["type"], p.get("roles") or {}, p["members"]
    if t == "pass_through_chain":
        out = {m: "intermediary" for m in members}
        out[members[0]] = "source"
        out[members[-1]] = "destination"
        return out
    if t == "fan_in_collector":
        out = {m: "feeder" for m in members}
        out[roles.get("collector", members[0])] = "collector"
        return out
    if t == "closed_loop":
        return {m: "loop member" for m in members}
    if t == "lockstep_cluster":
        return {m: "lockstep member" for m in members}
    return {m: "shares device" for m in members}


def _pattern_view(engine: "Engine", p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **{k: v for k, v in p.items() if k != "subgraph"},
        "member_roles": _member_roles(p),
        "member_risk": {m: engine.account_risk(m) for m in p["members"]},
    }


def subgraph_payload(engine: "Engine", nodes: Iterable[str], as_of: Optional[datetime],
                     highlight: Optional[Set[str]] = None, include_identity: bool = True,
                     patterns: Optional[List[Dict[str, Any]]] = None,
                     trigger: Optional[str] = None) -> Dict[str, Any]:
    """Nodes, aggregated payment edges and shared-identity edges among ``nodes``."""
    nodes = set(nodes)
    highlight = highlight or set()
    pattern_members: Dict[str, List[str]] = {}
    for p in (patterns if patterns is not None else engine.patterns_for(nodes)):
        for m in p["members"]:
            if m in nodes:
                pattern_members.setdefault(m, []).append(p["id"])

    edges: Dict[tuple, Dict[str, Any]] = {}
    for acc in nodes:
        for p in engine.graph.outgoing_payments(acc, as_of=as_of):
            if p.receiver not in nodes:
                continue
            key = (p.sender, p.receiver)
            e = edges.setdefault(key, {
                "id": f"{p.sender}->{p.receiver}", "source": p.sender, "target": p.receiver,
                "kind": "payment", "count": 0, "amount": 0.0, "first_ts": p.timestamp.isoformat(),
                "last_ts": None, "max_risk": None, "worst_action": None, "transactions": [],
                "is_trigger": False,
            })
            e["count"] += 1
            e["amount"] = round(e["amount"] + p.amount, 2)
            e["last_ts"] = p.timestamp.isoformat()
            rec = engine.payment_index.get(p.transaction_id or "")
            if rec is not None and (e["max_risk"] is None or rec["overall_risk"] > e["max_risk"]):
                e["max_risk"] = rec["overall_risk"]
                e["worst_action"] = rec["action"]
            if p.transaction_id == trigger:
                e["is_trigger"] = True
            e["transactions"].append({"id": p.transaction_id, "ts": p.timestamp.isoformat(),
                                      "amount": round(p.amount, 2)})
    for e in edges.values():
        e["transactions"] = e["transactions"][-6:]

    out_nodes = []
    for n in sorted(nodes):
        out_nodes.append({
            "id": n, "kind": "account", "label": n,
            "risk": engine.account_risk(n),
            "status": engine.account_status(n),
            "propagated_risk": round(engine.propagated.get(n, 0.0), 4),
            "patterns": pattern_members.get(n, []),
            "lockstep": n in engine.lockstep,
            "highlight": n in highlight,
            "degree": engine.state.s_count.get(n, 0) + engine.state.r_count.get(n, 0),
        })
    out_edges = list(edges.values())
    if include_identity:
        seen_ident: Dict[tuple, List[str]] = {}
        for n in nodes:
            for ident in engine.identity.get_identities(n, as_of=as_of):
                seen_ident.setdefault((ident["identity_type"], ident["identity_value"]), []).append(n)
        for (itype, value), accs in seen_ident.items():
            if len(accs) < 2:
                continue
            ident_id = f"{itype}:{value}"
            out_nodes.append({"id": ident_id, "kind": "identity", "identity_type": itype,
                              "label": value, "risk": 0.0, "patterns": [], "highlight": False,
                              "shared_count": len(accs)})
            for a in accs:
                out_edges.append({"id": f"{a}~{ident_id}", "source": a, "target": ident_id,
                                  "kind": "identity", "identity_type": itype})
    return {"nodes": out_nodes, "edges": out_edges, "as_of": as_of.isoformat() if as_of else None,
            "highlight": sorted(highlight)}


def timeline(engine: "Engine", alert: Dict[str, Any], nodes: Set[str], ts: datetime) -> List[Dict[str, Any]]:
    rec = alert["_record"]
    parties = {rec["sender"], rec["receiver"]}
    events: List[Dict[str, Any]] = []
    horizon = ts - timedelta(days=14)
    for acc in nodes:
        signup = engine.accounts.get(acc, {}).get("signup_at")
        if signup and signup <= ts and (acc in parties or signup >= horizon):
            events.append({"ts": signup.isoformat(), "kind": "signup", "accounts": [acc],
                           "title": f"{acc} account created"})
    payments = []
    for acc in nodes:
        for p in engine.graph.outgoing_payments(acc, as_of=ts):
            if p.receiver in nodes and p.timestamp >= horizon:
                payments.append(p)
    payments.sort(key=lambda p: p.timestamp)
    for p in payments[-MAX_TIMELINE:]:
        scored = engine.payment_index.get(p.transaction_id or "")
        events.append({
            "ts": p.timestamp.isoformat(), "kind": "payment", "accounts": [p.sender, p.receiver],
            "title": f"{p.sender} → {p.receiver}", "amount": round(p.amount, 2),
            "transaction_id": p.transaction_id,
            "risk": scored["overall_risk"] if scored else None,
            "action": scored["action"] if scored else None,
            "is_trigger": p.transaction_id == rec["transaction_id"],
        })
    for other in engine.alerts.values():
        if other["alert_id"] != alert["alert_id"] and {other["sender"], other["receiver"]} & nodes:
            if parse_timestamp(other["created_at"]) <= ts:
                events.append({"ts": other["created_at"], "kind": "alert",
                               "accounts": [other["sender"], other["receiver"]],
                               "title": f"Alert {other['alert_id']} · {other['action']}",
                               "alert_id": other["alert_id"], "status": other["status"]})
    for v in engine.verdicts:
        if set(v["seeds"]) & nodes or v["alert_id"] == alert["alert_id"]:
            events.append({"ts": v["at"], "kind": "verdict", "accounts": v["seeds"],
                           "title": f"Analyst {'confirmed fraud' if v['verdict'] == 'confirm' else 'cleared'}"
                                    f" on {v['alert_id']}", "alert_id": v["alert_id"]})
    for p in engine.patterns.values():
        if p["id"] in alert["pattern_ids"] and p.get("first_detected_at"):
            events.append({"ts": p["first_detected_at"], "kind": "pattern", "accounts": p["members"][:8],
                           "title": f"{p['type'].replace('_', ' ').capitalize()} detected "
                                    f"({len(p['members'])} accounts)", "pattern_id": p["id"]})
    events.sort(key=lambda e: e["ts"] or "")
    return events


def top_factors(rec: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    """Pair-model SHAP contributions (log-odds), strongest first."""
    feats, shap = rec["_features"], rec["_shap"]
    ranked = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]
    return [{
        "name": name,
        "label": FEATURE_LABELS.get(name, name),
        "value": round(float(feats.get(name, 0.0)), 4),
        "contribution": round(float(c), 4),
        "direction": "increases" if c > 0 else "decreases",
    } for name, c in ranked if abs(c) > 1e-4]


def counterfactual(engine: "Engine", rec: Dict[str, Any]) -> Dict[str, Any]:
    """F16: smallest change that flips the decision, and what evasion would cost.

    Amount search: re-run the pair model with only the amount changed (other
    signals held fixed) to find the largest amount that would be allowed.
    """
    sig, feats = rec["signals"], dict(rec["_features"])
    base = compute_counterfactual(rec["sub_scores"], rec["overall_risk"], rec["action"])

    def risk_at(amount: float) -> float:
        f = dict(feats)
        f["amount"] = amount
        f["log_amount"] = math.log1p(amount)
        pair, _ = engine.models.pair_predict(f)
        return fuse(pair, sig["anomaly"], sig["coordination"], sig["network_risk"], engine.models.calibration)

    amount = feats["amount"]
    floor = risk_at(1.0)
    result: Dict[str, Any] = {
        "sub_score": base,
        "risk_without_amount": round(floor, 4),
        "allow_threshold": ALLOW_THRESHOLD,
    }
    if floor >= ALLOW_THRESHOLD:
        drivers = [k for k, v in (("network risk from confirmed fraud", sig["network_risk"]),
                                  ("lockstep coordination", sig["coordination"]),
                                  ("pair-model graph features", sig["pair_risk"])) if v >= 0.3]
        result["max_allowed_amount"] = None
        result["counterfactual_line"] = (
            f"No amount flips this to ALLOW: even ₹1 scores {floor:.0%}. "
            f"Risk is driven by {', '.join(drivers) or 'account history'}, not the amount."
        )
        result["evasion_cost"] = "high — requires fresh identities or new counterparties, not smaller payments"
        return result
    lo, hi = 1.0, amount
    for _ in range(30):
        mid = (lo + hi) / 2
        if risk_at(mid) < ALLOW_THRESHOLD:
            lo = mid
        else:
            hi = mid
    result["max_allowed_amount"] = round(lo, 2)
    splits = max(2, int(-(-amount // max(lo, 1.0))))
    result["counterfactual_line"] = (
        f"At ₹{lo:,.0f} or less (instead of ₹{amount:,.0f}) this payment would be allowed; "
        f"everything else unchanged."
    )
    result["evasion_cost"] = (
        f"low — splitting into ~{splits} payments could evade; velocity features raise risk on repeats"
    )
    return result


def summary(rec: Dict[str, Any], patterns: List[Dict[str, Any]]) -> str:
    """F17: short narrative that cites only fields present in the case file."""
    parts = [
        f"{rec['sender']} → {rec['receiver']} for ₹{rec['amount']:,.2f} scored "
        f"{rec['overall_risk']:.0%} ({rec['action'].replace('_', ' ').lower()})."
    ]
    if patterns:
        p = patterns[0]
        parts.append(f"Both accounts sit in a detected {p['type'].replace('_', ' ')} of "
                     f"{len(p['members'])} accounts." if {rec['sender'], rec['receiver']} <= set(p['members'])
                     else f"A party belongs to a detected {p['type'].replace('_', ' ')} of "
                          f"{len(p['members'])} accounts.")
    top = [r["text"] for r in rec["reasons"] if r["code"] != "pattern"][:2]
    if top:
        parts.append("Key evidence: " + "; ".join(t[0].lower() + t[1:] for t in top) + ".")
    return " ".join(parts)
