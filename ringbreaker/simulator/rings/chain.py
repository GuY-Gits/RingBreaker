"""Mule chain fraud ring family (A -> B -> C -> D -> ...)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from ringbreaker.simulator.rings.base import RingSpec, make_payment


def generate_chain_instance(
    ring_id: str,
    members: List[str],
    user_meta: Dict[str, Dict[str, Any]],
    start_time: datetime,
    split: str,
    tx_counter_start: int,
    waves: int = 1,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates a mule pass-through chain instance with 1-3 waves, 3-6 hops,

    1 min - 3 h hop delay, and 1-6% skim per hop.
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start

    current_wave_start = start_time
    total_waves = max(1, waves)

    for wave_idx in range(total_waves):
        current_time = current_wave_start
        amount = float(round(random.uniform(4500, 9500), 2))
        skim_rate = random.uniform(0.01, 0.06)

        # In wave 2 or 3, maybe use a subset or permuted sequence of the same mules
        chain_nodes = members if wave_idx == 0 else random.sample(members, max(3, len(members) - 1))

        for i in range(len(chain_nodes) - 1):
            u_from = chain_nodes[i]
            u_to = chain_nodes[i + 1]

            # Hop delay between 1 minute and 3 hours
            delay_minutes = random.randint(1, 180)
            current_time += timedelta(minutes=delay_minutes, seconds=random.randint(0, 59))
            amount = round(amount * (1.0 - skim_rate), 2)

            payments.append(make_payment(
                tx_id=tx_id,
                sender=u_from,
                receiver=u_to,
                amount=amount,
                when=current_time,
                user_meta=user_meta,
                ring_id=ring_id,
                is_fraud=1,
            ))
            tx_id += 1

        # Next wave occurs 2-6 days later
        current_wave_start = current_time + timedelta(days=random.randint(2, 6))

    spec = RingSpec(
        ring_id=ring_id,
        family="mule_chain",
        split=split,
        variant="multi_wave" if total_waves > 1 else "standard",
        members=members,
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "member_count": len(members),
            "waves": total_waves,
            "hop_count": len(members) - 1,
            "start_amount": round(payments[0]["amount"], 2),
        },
    )
    return payments, spec, tx_id
