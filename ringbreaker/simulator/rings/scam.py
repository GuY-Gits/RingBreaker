"""Scam victim fraud injection (social engineering / phishing payments to mules)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from ringbreaker.simulator.rings.base import RingSpec, make_payment


def generate_scam_instance(
    ring_id: str,
    victim_accounts: List[str],
    mule_account: str,
    exit_account: str,
    user_meta: Dict[str, Dict[str, Any]],
    start_time: datetime,
    split: str,
    tx_counter_start: int,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates an instance where 1-4 established victims make first-time large out-of-pattern

    payments to a mule, followed by the mule cashing out.
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start

    total_scammed = 0.0
    latest_time = start_time

    for victim in victim_accounts:
        offset_m = random.randint(5, 360)
        tx_time = start_time + timedelta(minutes=offset_m)
        latest_time = max(latest_time, tx_time)

        # High amount out-of-pattern
        amt = float(round(random.uniform(5000, 18000), 2))
        total_scammed += amt

        payments.append(make_payment(
            tx_id=tx_id,
            sender=victim,
            receiver=mule_account,
            amount=amt,
            when=tx_time,
            user_meta=user_meta,
            ring_id=ring_id,
            is_fraud=1,
        ))
        tx_id += 1

    # Mule forwards the funds to an exit account within 10 to 60 minutes
    drain_time = latest_time + timedelta(minutes=random.randint(10, 60))
    payments.append(make_payment(
        tx_id=tx_id,
        sender=mule_account,
        receiver=exit_account,
        amount=round(total_scammed * 0.94, 2),
        when=drain_time,
        user_meta=user_meta,
        ring_id=ring_id,
        is_fraud=1,
    ))
    tx_id += 1

    spec = RingSpec(
        ring_id=ring_id,
        family="scam_victim_mule",
        split=split,
        variant="standard",
        members=victim_accounts + [mule_account, exit_account],
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "victim_count": len(victim_accounts),
            "mule": mule_account,
            "total_scammed": round(total_scammed, 2),
        },
    )
    return payments, spec, tx_id
