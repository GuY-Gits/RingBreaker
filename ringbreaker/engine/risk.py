"""Sub-scores (F10), risk fusion and fact-grounded reasons.

Every number here is derived from as-of-T features, the two trained models,
slow-path lockstep clusters and analyst-propagated risk. Nothing reads labels.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ringbreaker.scoring.action import determine_action
from ringbreaker.scoring.risk_engine import ANOMALY_WEIGHT, LOCKSTEP_WEIGHT, PAIR_WEIGHT

FEATURE_LABELS = {
    "amount": "Payment amount",
    "log_amount": "Payment amount (log)",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "day_of_month": "Day of month",
    "is_weekend": "Weekend payment",
    "is_night": "Night-time payment (00–06h)",
    "sender_account_age_days": "Sender account age (days)",
    "receiver_account_age_days": "Receiver account age (days)",
    "sender_tx_count_before": "Sender payments sent before",
    "sender_unique_receivers_before": "Sender distinct payees",
    "sender_avg_amount_before": "Sender average amount",
    "sender_max_amount_before": "Sender largest amount",
    "sender_avg_time_gap": "Sender average gap (s)",
    "sender_tx_last_1h": "Sender payments, last 1h",
    "sender_tx_last_24h": "Sender payments, last 24h",
    "sender_tx_last_7d": "Sender payments, last 7d",
    "receiver_tx_count_before": "Receiver payments received before",
    "receiver_unique_senders_before": "Receiver distinct payers",
    "receiver_avg_amount_before": "Receiver average inflow",
    "receiver_max_amount_before": "Receiver largest inflow",
    "receiver_tx_last_1h": "Receiver inflows, last 1h",
    "receiver_tx_last_24h": "Receiver inflows, last 24h",
    "receiver_tx_last_7d": "Receiver inflows, last 7d",
    "pair_tx_count_before": "Prior payments on this pair",
    "pair_avg_amount_before": "Pair average amount",
    "time_since_previous_pair_tx": "Seconds since last pair payment",
    "is_first_transaction_between_pair": "First payment to this payee",
    "device_tx_count_before": "Payments from this device",
    "device_unique_users_before": "Accounts using this device",
    "ip_tx_count_before": "Payments from this IP",
    "ip_unique_users_before": "Accounts using this IP",
    "sender_in_degree_before": "Sender inflow count",
    "sender_out_degree_before": "Sender outflow count",
    "receiver_in_degree_before": "Receiver inflow count",
    "receiver_out_degree_before": "Receiver outflow count",
    "sender_out_in_ratio": "Sender out/in ratio",
    "receiver_in_out_ratio": "Receiver in/out ratio",
    "sender_time_since_last_incoming": "Seconds since sender last received money",
}


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_sub_scores(f: Dict[str, float], ctx: Dict[str, Any]) -> Dict[str, float]:
    """Three PRD sub-scores in [0, 1]; higher always means riskier.

    ``relationship_plausibility`` keeps the PRD field name but is reported as
    risk: 1.0 = an implausible relationship (stranger, no reciprocity, no
    shared contacts, unusual amount).
    """
    avg = f["sender_avg_amount_before"]
    spike = _clip((f["amount"] - avg) / (avg * 3.0)) if avg > 0 else 0.0
    velocity = _clip(f["sender_tx_last_1h"] / 3.0 + f["sender_tx_last_24h"] / 10.0)
    sender = _clip(
        0.50 * ctx["anomaly"] + 0.20 * velocity + 0.20 * spike + 0.10 * f["is_night"]
        + 0.5 * ctx["sender_propagated"]
    )

    age = f["receiver_account_age_days"]
    newness = 1.0 if age < 7 else max(0.0, (30.0 - age) / 30.0)
    burst = _clip(f["receiver_tx_last_1h"] / 3.0 + f["receiver_tx_last_24h"] / 8.0)
    shared = _clip((ctx["receiver_identity_share"] - 1) / 4.0)
    thin = _clip(1.0 - ctx["receiver_lifelikeness"])
    receiver = _clip(
        0.25 * newness + 0.20 * burst + 0.20 * shared + 0.20 * thin
        + 0.15 * ctx["receiver_pass_through"] + 0.5 * ctx["receiver_propagated"]
    )

    social = ctx["social"]
    rel = _clip(
        0.45 * f["is_first_transaction_between_pair"]
        + 0.20 * (1 - social["reciprocity"])
        + 0.15 * (1 if social["shared_neighbour_count"] == 0 else 0)
        + 0.20 * spike
    )
    return {
        "sender_anomaly": round(sender, 4),
        "receiver_mule_propensity": round(receiver, 4),
        "relationship_plausibility": round(rel, 4),
    }


def fuse(pair: float, anomaly: float, coordination: float, network: float) -> float:
    """PRD risk engine weights, then noisy-OR with analyst-propagated network risk."""
    fused = PAIR_WEIGHT * pair + ANOMALY_WEIGHT * anomaly + LOCKSTEP_WEIGHT * coordination
    return round(_clip(1.0 - (1.0 - _clip(fused)) * (1.0 - _clip(network))), 4)


def action_for(overall: float, sub: Dict[str, float]) -> str:
    return determine_action(overall, sub)


def build_reasons(f: Dict[str, float], ctx: Dict[str, Any], signals: Dict[str, float]) -> List[Dict[str, Any]]:
    """Plain-language reasons, each tied to the field that produced it."""
    out: List[Dict[str, Any]] = []

    def add(code: str, text: str, severity: str, source: str) -> None:
        out.append({"code": code, "text": text, "severity": severity, "source": source})

    if signals["pair_risk"] >= 0.5:
        add("pair_model", f"Pair model gives {signals['pair_risk']:.0%} fraud probability", "high", "pair_risk")
    if signals["network_risk"] >= 0.2:
        add("propagated", f"Linked to analyst-confirmed fraud (propagated risk {signals['network_risk']:.0%})",
            "high", "network_risk")
    if signals["coordination"] >= 0.5:
        add("lockstep", f"Account is in a lockstep cluster of {ctx['lockstep_size']} accounts "
            f"(score {signals['coordination']:.2f})", "high", "coordination")
    if signals["anomaly"] >= 0.95:
        add("behaviour", f"Sender behaviour is more unusual than {signals['anomaly']:.0%} of training payments",
            "medium", "anomaly")
    if f["receiver_account_age_days"] < 7:
        add("new_receiver", f"Receiver account is {f['receiver_account_age_days']:.1f} days old", "medium",
            "receiver_account_age_days")
    if f["receiver_tx_last_1h"] >= 2:
        add("inflow_burst", f"Receiver got {int(f['receiver_tx_last_1h'])} payments in the last hour", "medium",
            "receiver_tx_last_1h")
    if ctx["receiver_identity_share"] >= 3:
        add("shared_identity", f"Receiver shares an identity fragment with {ctx['receiver_identity_share'] - 1} "
            "other accounts", "medium", "identity_graph")
    if f["device_unique_users_before"] >= 3:
        add("shared_device", f"Sender's device has been used by {int(f['device_unique_users_before'])} accounts",
            "medium", "device_unique_users_before")
    since_in = f["sender_time_since_last_incoming"]
    if 0 <= since_in <= 3600:
        add("pass_through", f"Sender received money {since_in / 60:.0f} min before forwarding it", "medium",
            "sender_time_since_last_incoming")
    social = ctx["social"]
    if f["is_first_transaction_between_pair"] and not social["reciprocity"] and social["shared_neighbour_count"] == 0:
        add("stranger", "First payment to a payee with no shared contacts and no history back", "low",
            "social")
    if ctx["receiver_lifelikeness"] < 0.3:
        add("thin_identity", f"Receiver identity is thin (lifelikeness {ctx['receiver_lifelikeness']:.2f})", "low",
            "lifelikeness")
    for p in ctx.get("patterns", [])[:3]:
        add("pattern", f"Party to detected {p['type'].replace('_', ' ')} ({len(p['members'])} accounts)",
            "high", "patterns")
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda r: order[r["severity"]])
    return out
