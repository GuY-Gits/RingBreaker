"""Device farm (star topology) fraud ring family."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ringbreaker.simulator.rings.base import RingSpec, make_payment


def generate_star_instance(
    ring_id: str,
    spoke_accounts: List[str],
    hub_account: str,
    user_meta: Dict[str, Dict[str, Any]],
    start_time: datetime,
    split: str,
    tx_counter_start: int,
    shared_device_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates a device farm / star ring where 5-15 accounts on 1-2 shared devices

    make multiple small payments and periodically sweep funds to a central hub/collector.
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start

    dev = shared_device_id or f"DEV_FARM_{random.randint(100, 999)}"

    # Multiple small payments per spoke
    total_swept = 0.0
    latest_time = start_time
    for spoke in spoke_accounts:
        num_payments = random.randint(1, 3)
        for _ in range(num_payments):
            offset_h = random.uniform(0.5, 36.0)
            tx_time = start_time + timedelta(hours=offset_h)
            latest_time = max(latest_time, tx_time)

            amt = float(round(random.uniform(800, 2800), 2))
            total_swept += amt

            payments.append(make_payment(
                tx_id=tx_id,
                sender=spoke,
                receiver=hub_account,
                amount=amt,
                when=tx_time,
                user_meta=user_meta,
                ring_id=ring_id,
                is_fraud=1,
                override_device=dev,
            ))
            tx_id += 1

    # Optional final sweep from hub to off-ramp
    sweep_time = latest_time + timedelta(minutes=random.randint(20, 60))
    payments.append(make_payment(
        tx_id=tx_id,
        sender=hub_account,
        receiver=spoke_accounts[0],
        amount=round(total_swept * 0.95, 2),
        when=sweep_time,
        user_meta=user_meta,
        ring_id=ring_id,
        is_fraud=1,
    ))
    tx_id += 1

    spec = RingSpec(
        ring_id=ring_id,
        family="device_farm_star",
        split=split,
        variant="standard",
        members=spoke_accounts + [hub_account],
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "spokes_count": len(spoke_accounts),
            "shared_device": dev,
            "total_swept": round(total_swept, 2),
        },
    )
    return payments, spec, tx_id
