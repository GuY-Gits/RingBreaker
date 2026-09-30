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
    "ALLOW": 0.0,
    "REVIEW": 0.30,
    "BLOCK": 0.70,
    "allow": 0.0,
    "review": 0.30,
    "block": 0.70,
}


def _dominant_sub_score(sub_scores: Dict[str, float]) -> str:
    return max(SUB_SCORE_KEYS, key=lambda k: float(sub_scores.get(k, 0.0)))


def _action_from_sub_scores(
    sub_scores: Dict[str, float],
    combined_risk: float,
    thresholds: Dict[str, float],
) -> str:
    """Derive action from risk using canonical thresholds."""
    if combined_risk >= thresholds.get("BLOCK", 0.70):
        return "BLOCK"
    if combined_risk >= thresholds.get("REVIEW", 0.30):
        return "REVIEW"
    return "ALLOW"


def _action_ladder_index(action: str) -> int:
    act = str(action).strip().upper()
    if act == "BLOCK":
        return 2
    if act in ("REVIEW", "WARN_SENDER", "HOLD_RECEIVER"):
        return 1
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
    current_action = str(action).upper() if action else _action_from_sub_scores(sub_scores, combined_risk, thresholds)
    current_idx = _action_ladder_index(current_action)

    if current_idx == 0:
        return {
            "counterfactual_line": "No change needed — payment would already be allowed.",
            "dominant_factor": dominant,
            "delta_required": 0.0,
            "suggested_change": None,
            "evasion_cost": 0.0,
            "flipped_action": "ALLOW",
        }

    target_idx = current_idx - 1
    order = ["ALLOW", "REVIEW", "BLOCK"]
    target_action = order[target_idx]

    # Threshold on dominant dimension for target action
    if target_action == "ALLOW":
        target_threshold = thresholds.get("ALLOW", thresholds.get("allow", 0.0))
    elif target_action == "REVIEW":
        target_threshold = thresholds.get("REVIEW", thresholds.get("review", 0.30))
    else:
        target_threshold = thresholds.get("BLOCK", thresholds.get("block", 0.70))

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
