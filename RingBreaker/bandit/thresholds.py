"""
RingBreaker - P2 Feature F23: Adaptive Action Threshold Policies
Module: bandit/thresholds.py

Defines the action threshold policies and reward matrix for the bandit experiment.
"""

from typing import Dict, Any

POLICIES: Dict[str, Dict[str, float]] = {
    "conservative": {
        "review_threshold": 0.20,
        "block_threshold": 0.60
    },
    "current": {
        "review_threshold": 0.30,
        "block_threshold": 0.70
    },
    "aggressive": {
        "review_threshold": 0.40,
        "block_threshold": 0.80
    }
}

REWARD_MATRIX: Dict[str, Dict[str, float]] = {
    "fraud": {
        "BLOCK": 3.0,
        "REVIEW": 2.0,
        "ALLOW": -5.0
    },
    "normal": {
        "ALLOW": 2.0,
        "REVIEW": -1.0,
        "BLOCK": -3.0
    }
}


def action_from_risk(risk_score: float, policy_name: str) -> str:
    """Maps a risk score [0.0, 1.0] to ALLOW, REVIEW, or BLOCK based on the chosen policy."""
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown threshold policy: {policy_name}")

    p = POLICIES[policy_name]
    rev_t = p["review_threshold"]
    blk_t = p["block_threshold"]

    if risk_score >= blk_t:
        return "BLOCK"
    elif risk_score >= rev_t:
        return "REVIEW"
    return "ALLOW"


def calculate_reward(action: str, is_fraud: int) -> float:
    """Calculates reward based on operational action and ground-truth confirmation."""
    label_key = "fraud" if is_fraud == 1 else "normal"
    return REWARD_MATRIX[label_key][action]