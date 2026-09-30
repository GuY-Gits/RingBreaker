"""
RingBreaker - Action Policy Layer
Module: scoring/action.py

Maps continuous risk scores [0.0, 1.0] to operational decisions:
- ALLOW:  risk < 0.30
- REVIEW: 0.30 <= risk < 0.70
- BLOCK:  risk >= 0.70
"""

from typing import Dict, Any

ALLOW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.70


def determine_action(risk_score: float) -> str:
    """Classifies risk score into ALLOW, REVIEW, or BLOCK."""
    if risk_score < ALLOW_THRESHOLD:
        return "ALLOW"
    elif risk_score < BLOCK_THRESHOLD:
        return "REVIEW"
    else:
        return "BLOCK"


def format_action_payload(
    transaction_id: str,
    overall_risk: float,
    pair_risk: float,
    behavioural_anomaly: float,
    components: Dict[str, float] = None
) -> Dict[str, Any]:
    """Builds a structured dictionary matching the RingBreaker PRD API interface."""
    action = determine_action(overall_risk)
    payload = {
        "transaction_id": transaction_id,
        "risk_score": round(float(overall_risk), 4),
        "pair_risk": round(float(pair_risk), 4),
        "behavioural_anomaly": round(float(behavioural_anomaly), 4),
        "action": action
    }
    if components:
        payload["components"] = components
    return payload