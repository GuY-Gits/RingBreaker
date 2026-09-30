"""
F16 — Counterfactual and evasion cost.

Smallest change to sub-scores that would flip the adaptive action (F11).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Matches scoring.action dominant dimensions
SUB_SCORE_KEYS = ("sender_anomaly", "receiver_mule_propensity", "relationship_plausibility")

# Index in the ladder at or above which we flag (aligned with default action thresholds)
DEFAULT_THRESHOLDS = {
    "allow": 0.0,
    "warn_sender": 0.35,
    "hold_receiver": 0.55,
    "block": 0.75,
}


def _dominant_sub_score(sub_scores: Dict[str, float]) -> str:
    return max(SUB_SCORE_KEYS, key=lambda k: float(sub_scores.get(k, 0.0)))


def _action_from_sub_scores(
    sub_scores: Dict[str, float],
    combined_risk: float,
    thresholds: Dict[str, float],
) -> str:
    """Mirror F11: dominant sub-score drives the action ladder."""
    dominant = _dominant_sub_score(sub_scores)
    score = float(sub_scores.get(dominant, combined_risk))

    if score >= thresholds["block"]:
        return "block"
    if score >= thresholds["hold_receiver"]:
        return "hold_receiver" if dominant == "receiver_mule_propensity" else "hold_receiver"
    if score >= thresholds["warn_sender"]:
        return "warn_sender" if dominant in ("sender_anomaly", "relationship_plausibility") else "hold_receiver"
    return "allow"


def _action_ladder_index(action: str) -> int:
    order = ["allow", "warn_sender", "hold_receiver", "block"]
    try:
        return order.index(action)
    except ValueError:
        return 0


def compute_counterfactual(
    sub_scores: Dict[str, float],
    combined_risk: float,
    action: str,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Find the smallest uniform reduction on the dominant sub-score that flips
    the decision to the next lower action (or to allow).
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    dominant = _dominant_sub_score(sub_scores)
    current = float(sub_scores.get(dominant, combined_risk))
    current_action = action or _action_from_sub_scores(sub_scores, combined_risk, thresholds)
    current_idx = _action_ladder_index(current_action)

    if current_idx == 0:
        return {
            "counterfactual_line": "No change needed — payment would already be allowed.",
            "dominant_factor": dominant,
            "delta_required": 0.0,
            "suggested_change": None,
            "evasion_cost": 0.0,
            "flipped_action": "allow",
        }

    target_idx = current_idx - 1
    order = ["allow", "warn_sender", "hold_receiver", "block"]
    target_action = order[target_idx]

    # Threshold on dominant dimension for target action
    if target_action == "allow":
        target_threshold = thresholds["allow"]
    elif target_action == "warn_sender":
        target_threshold = thresholds["warn_sender"]
    elif target_action == "hold_receiver":
        target_threshold = thresholds["hold_receiver"]
    else:
        target_threshold = thresholds["block"]

    delta = max(0.0, current - target_threshold + 1e-6)
    suggested: Dict[str, Any] = {
        "adjust": dominant,
        "from": round(current, 4),
        "to": round(max(0.0, current - delta), 4),
    }

    # Evasion cost: weighted sum of deltas across sub-scores if attacker only moves one knob
    evasion_cost = round(delta, 4)

    human_labels = {
        "sender_anomaly": "sender behavioural anomaly",
        "receiver_mule_propensity": "receiver mule propensity",
        "relationship_plausibility": "relationship plausibility",
    }
    label = human_labels.get(dominant, dominant)
    counterfactual_line = (
        f"Lower {label} by {delta:.2f} (from {current:.2f} to {suggested['to']:.2f}) "
        f"to change action from {current_action} to {target_action}."
    )

    return {
        "counterfactual_line": counterfactual_line,
        "dominant_factor": dominant,
        "delta_required": round(delta, 4),
        "suggested_change": suggested,
        "evasion_cost": evasion_cost,
        "flipped_action": target_action,
    }


def batch_counterfactuals(
    alerts: List[Dict[str, Any]],
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Attach counterfactual fields to alert dicts (for tests or batch export)."""
    out: List[Dict[str, Any]] = []
    for alert in alerts:
        sub = alert.get("sub_scores") or {}
        cf = compute_counterfactual(
            sub,
            float(alert.get("risk_score", 0.0)),
            str(alert.get("action", "allow")),
            thresholds=thresholds,
        )
        merged = {**alert, "counterfactual": cf}
        out.append(merged)
    return out
