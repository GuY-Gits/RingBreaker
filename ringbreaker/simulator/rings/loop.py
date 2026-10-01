"""Closed loop fraud ring family (A -> B -> C -> ... -> A)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from ringbreaker.simulator.rings.base import RingSpec, make_payment


def generate_loop_instance(
    ring_id: str,
    members: List[str],
    user_meta: Dict[str, Dict[str, Any]],
    start_time: datetime,
    split: str,
    tx_counter_start: int,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates a closed loop ring instance with randomized cycle time, hop delay and amount decay."""
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start
    current_time = start_time
    base_amount = float(round(random.uniform(3200, 7500), 2))

    # Span between 1 hour and 3 days total
    total_span_hours = random.uniform(1.0, 72.0)
    hop_minutes = max(5, int((total_span_hours * 60) / max(len(members), 1)))

    decay_range = (0.95, 0.99)
    cycle = members + [members[0]]
    for i in range(len(cycle) - 1):
        u_from = cycle[i]
        u_to = cycle[i + 1]
        delay_m = random.randint(max(2, hop_minutes // 2), max(5, int(hop_minutes * 1.5)))
        current_time += timedelta(minutes=delay_m, seconds=random.randint(0, 59))
        amt = round(base_amount * random.uniform(decay_range[0], decay_range[1]), 2)
        base_amount = amt

        payments.append(make_payment(
            tx_id=tx_id,
            sender=u_from,
            receiver=u_to,
            amount=amt,
            when=current_time,
            user_meta=user_meta,
            ring_id=ring_id,
            is_fraud=1,
        ))
        tx_id += 1

    spec = RingSpec(
        ring_id=ring_id,
        family="closed_loop",
        split=split,
        variant="standard",
        members=members,
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "member_count": len(members),
            "start_amount": round(base_amount, 2),
            "span_hours": round(total_span_hours, 2),
            "decay_range": decay_range,
        },
    )
    return payments, spec, tx_id
