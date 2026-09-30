"""RingBreaker - Action Policy Layer (Canonical Risk Engine Thresholds).

Action thresholds:
- ALLOW:  overall_risk < 0.30
- REVIEW: 0.30 <= overall_risk < 0.70
- BLOCK:  overall_risk >= 0.70

Single source of truth:
All actions, percentages, and case files derive strictly from overall_risk.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

SUB_SCORE_KEYS = ("sender_anomaly", "receiver_mule_propensity", "relationship_plausibility")

ALLOW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.70

DEFAULT_THRESHOLDS = {
    "ALLOW": 0.0,
    "REVIEW": ALLOW_THRESHOLD,
    "BLOCK": BLOCK_THRESHOLD,
    # Lowercase aliases for backward compatibility
    "allow": 0.0,
    "review": ALLOW_THRESHOLD,
    "block": BLOCK_THRESHOLD,
}


def determine_action(
    overall_risk: float,
    sub_scores: Optional[Dict[str, float]] = None,
) -> str:
    """PRD F11 Adaptive action policy.

    - overall_risk < 0.30 -> ALLOW
    - overall_risk >= 0.70 -> BLOCK
    - 0.30 <= overall_risk < 0.70, routed by the dominant sub-score:
        - receiver_mule_propensity dominates -> HOLD_RECEIVER
        - sender_anomaly or relationship (implausibility) dominates -> WARN_SENDER
        - no sub-scores at all -> REVIEW
    """
    risk = float(overall_risk)
    if risk >= BLOCK_THRESHOLD:
        return "BLOCK"
    if risk < ALLOW_THRESHOLD:
        return "ALLOW"

    if not sub_scores:
        return "REVIEW"

    s_anom = float(sub_scores.get("sender_anomaly", 0.0))
    r_mule = float(sub_scores.get("receiver_mule_propensity", 0.0))
    rel = float(sub_scores.get("relationship_plausibility", 0.0))

    # Receiver looks like a mule -> hold the money at the receiver.
    if r_mule > s_anom and r_mule >= rel:
        return "HOLD_RECEIVER"
    # Sender behaves oddly, or is paying an implausible stranger (scam victim
    # pattern) -> warn the sender before the money leaves.
    if s_anom > 0.0 or rel > 0.0 or r_mule > 0.0:
        return "WARN_SENDER" if (s_anom >= r_mule or rel > r_mule) else "HOLD_RECEIVER"
    return "REVIEW"


def format_canonical_risk(
    overall_risk: float,
    sub_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Single canonical risk representation with decimal, percentage, and action."""
    risk = round(float(np.clip(overall_risk, 0.0, 1.0)), 4)
    return {
        "overall_risk": risk,
        "risk_percent": round(risk * 100.0, 2),
        "action": determine_action(risk, sub_scores),
    }


def dominant_sub_score(sub_scores: Dict[str, float]) -> str:
    """Returns the sub-score key with the highest value."""
    if not sub_scores:
        return "relationship_plausibility"
    return max(SUB_SCORE_KEYS, key=lambda k: float(sub_scores.get(k, 0.0)))


def decide_action(
    sub_scores: Dict[str, float],
    risk_score: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Backward compatibility wrapper mapping sub-scores and risk to adaptive action."""
    return determine_action(risk_score, sub_scores)


def format_action_payload(
    transaction_id: str,
    overall_risk: float,
    pair_risk: float,
    behavioural_anomaly: float,
    sub_scores: Optional[Dict[str, float]] = None,
    components: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Builds a structured dictionary matching the RingBreaker PRD API interface."""
    sub = sub_scores or {
        "sender_anomaly": round(float(behavioural_anomaly), 4),
        "receiver_mule_propensity": round(float(pair_risk), 4),
        "relationship_plausibility": 0.25,
    }
    canonical = format_canonical_risk(overall_risk, sub)
    payload = {
        "transaction_id": transaction_id,
        "risk_score": canonical["overall_risk"],
        "overall_risk": canonical["overall_risk"],
        "risk_percent": canonical["risk_percent"],
        "pair_risk": round(float(pair_risk), 4),
        "behavioural_anomaly": round(float(behavioural_anomaly), 4),
        "sub_scores": sub,
        "action": canonical["action"],
    }
    if components:
        payload["components"] = components
    return payload