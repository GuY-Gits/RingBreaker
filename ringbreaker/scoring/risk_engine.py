"""
RingBreaker - Integrated Risk Engine (3-Signal Fusion with Model Selection)
Module: scoring/risk_engine.py

Combines:
1. Transaction-level Pair Risk (0.60) - supports 'original' vs 'retrained' XGBoost models
2. Account-level Behavioural Anomaly (Extended Isolation Forest) (0.25)
3. Account-level Lockstep Coordination (DBSCAN) (0.15)

The live engine (``ringbreaker.engine``) uses these weights; see
``engine/risk.py`` for how propagated analyst risk is combined on top.
"""

from typing import Dict, Any, Optional

import numpy as np

from ringbreaker.scoring.action import determine_action

# Configurable Weights (Unchanged)
PAIR_WEIGHT = 0.60
ANOMALY_WEIGHT = 0.25
LOCKSTEP_WEIGHT = 0.15


class RiskEngine:
    def __init__(
        self,
        pair_weight: float = PAIR_WEIGHT,
        anomaly_weight: float = ANOMALY_WEIGHT,
        lockstep_weight: float = LOCKSTEP_WEIGHT
    ):
        self.pair_weight = pair_weight
        self.anomaly_weight = anomaly_weight
        self.lockstep_weight = lockstep_weight

    def score(
        self,
        payment: Dict[str, Any],
        pair_risk: float,
        behavioural_anomaly: float,
        coordination_score: float = 0.0,
        graph_risk: Optional[float] = None,
        sub_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        overall = (
            self.pair_weight * pair_risk +
            self.anomaly_weight * behavioural_anomaly +
            self.lockstep_weight * coordination_score
        )
        overall = float(np.clip(overall, 0.0, 1.0))
        if sub_scores is None:
            sub_scores = {
                "sender_anomaly": float(behavioural_anomaly),
                "receiver_mule_propensity": float(pair_risk),
                "relationship_plausibility": float(coordination_score),
            }
        action = determine_action(overall, sub_scores)

        return {
            "transaction_id": payment.get("transaction_id", "UNKNOWN"),
            "risk_score": round(overall, 4),
            "overall_risk": round(overall, 4),
            "risk_percent": round(overall * 100.0, 2),
            "pair_risk": round(float(pair_risk), 4),
            "behavioural_anomaly": round(float(behavioural_anomaly), 4),
            "coordination_score": round(float(coordination_score), 4),
            "sub_scores": sub_scores,
            "action": action
        }
