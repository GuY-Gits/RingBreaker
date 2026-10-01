"""Fan-in collector and cash-out fraud ring family."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from ringbreaker.simulator.rings.base import RingSpec, make_payment


def generate_fan_in_instance(
    ring_id: str,
    feeders: List[str],
    collector: str,
    exit_nodes: List[str],
    user_meta: Dict[str, Dict[str, Any]],
    start_time: datetime,
    split: str,
    tx_counter_start: int,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates a fan-in collector ring where 4-15 feeders send funds to a central mule within

    a 1-48 hour window, followed by cash-out to 1-3 exit nodes.
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start

    window_hours = random.uniform(1.0, 48.0)
    window_seconds = int(window_hours * 3600)

    total_collected = 0.0
    latest_feed_time = start_time

    for f in feeders:
        offset_sec = random.randint(0, max(60, window_seconds))
        tx_time = start_time + timedelta(seconds=offset_sec)
        latest_feed_time = max(latest_feed_time, tx_time)

        amt = float(round(random.uniform(2200, 5800), 2))
        total_collected += amt

        payments.append(make_payment(
            tx_id=tx_id,
            sender=f,
            receiver=collector,
            amount=amt,
            when=tx_time,
            user_meta=user_meta,
            ring_id=ring_id,
            is_fraud=1,
        ))
        tx_id += 1

    # Cash-out to exits: happens 15 to 90 minutes after latest feed
    cashout_time = latest_feed_time + timedelta(minutes=random.randint(15, 90))
    retained_cut = total_collected * random.uniform(0.93, 0.98)
    exit_share = retained_cut / len(exit_nodes)

    for ex in exit_nodes:
        cashout_time += timedelta(minutes=random.randint(2, 10))
        payments.append(make_payment(
            tx_id=tx_id,
            sender=collector,
            receiver=ex,
            amount=round(exit_share, 2),
            when=cashout_time,
            user_meta=user_meta,
            ring_id=ring_id,
            is_fraud=1,
        ))
        tx_id += 1

    spec = RingSpec(
        ring_id=ring_id,
        family="fan_in_collector",
        split=split,
        variant="standard",
        members=feeders + [collector] + exit_nodes,
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "feeders_count": len(feeders),
            "exit_count": len(exit_nodes),
            "window_hours": round(window_hours, 2),
            "total_collected": round(total_collected, 2),
        },
    )
    return payments, spec, tx_id
