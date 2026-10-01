"""Held-out novel fraud ring families (never seen in training or validation).

1. Slow chain: hops 6-24 hours apart, amounts split and recombined, fresh mules.
2. Distributed-device ring: fan-in collector whose feeders use distinct devices but share a phone or address.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ringbreaker.simulator.rings.base import RingSpec, make_payment


def generate_slow_chain_instance(
    ring_id: str,
    members: List[str],
    user_meta: Dict[str, Dict[str, Any]],
    start_time: datetime,
    tx_counter_start: int,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates a slow pass-through chain where hops are 6-24 hours apart,

    and amounts are split and recombined through intermediaries.
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start
    current_time = start_time
    amount = float(round(random.uniform(7000, 14000), 2))

    for i in range(len(members) - 1):
        u_from = members[i]
        u_to = members[i + 1]

        # Slow hop delay: 6 to 24 hours
        hop_hours = random.uniform(6.0, 24.0)
        current_time += timedelta(hours=hop_hours, minutes=random.randint(0, 59))
        amount = round(amount * random.uniform(0.96, 0.99), 2)

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

    spec = RingSpec(
        ring_id=ring_id,
        family="slow_chain",
        split="heldout",
        variant="held_out_novel",
        members=members,
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "hop_count": len(members) - 1,
            "min_hop_hours": 6,
            "max_hop_hours": 24,
            "total_span_hours": round((current_time - start_time).total_seconds() / 3600, 1),
        },
    )
    return payments, spec, tx_id


def generate_distributed_device_instance(
    ring_id: str,
    feeders: List[str],
    collector: str,
    user_meta: Dict[str, Dict[str, Any]],
    start_time: datetime,
    tx_counter_start: int,
    shared_address: Optional[str] = None,
    shared_phone: Optional[str] = None,
    df_users: Optional[pd.DataFrame] = None,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates a distributed-device ring where feeders use distinct devices

    but share a phone or address to funnel funds to a collector.
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start

    addr = shared_address or f"Unit {random.randint(10, 80)}, Cyber Towers, Hi-Tech City, Hyderabad - 500081"
    ph = shared_phone or f"+9199123{random.randint(10000, 99999)}"

    # Update metadata to reflect identity fragment collisions
    # Persist to users.csv too, so the identity graph actually sees the shared fragments.
    for f in feeders:
        updates = {"address": addr}
        if random.random() < 0.5:
            updates["phone"] = ph
        user_meta[f].update(updates)
        if df_users is not None:
            idx = df_users.index[df_users["user_id"] == f][0]
            for k, v in updates.items():
                df_users.at[idx, k] = v

    total_in = 0.0
    latest_time = start_time
    for f in feeders:
        offset_h = random.uniform(1.0, 36.0)
        tx_time = start_time + timedelta(hours=offset_h)
        latest_time = max(latest_time, tx_time)

        amt = float(round(random.uniform(2500, 6000), 2))
        total_in += amt

        # Feeder uses their own distinct device, but shares address/phone in identity graph
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

    # Collector cash-out
    drain_time = latest_time + timedelta(hours=random.uniform(1.0, 4.0))
    payments.append(make_payment(
        tx_id=tx_id,
        sender=collector,
        receiver=feeders[0],
        amount=round(total_in * 0.95, 2),
        when=drain_time,
        user_meta=user_meta,
        ring_id=ring_id,
        is_fraud=1,
    ))
    tx_id += 1

    spec = RingSpec(
        ring_id=ring_id,
        family="distributed_device_ring",
        split="heldout",
        variant="held_out_novel",
        members=feeders + [collector],
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "feeders_count": len(feeders),
            "shared_address": addr,
            "shared_phone": ph,
            "total_in": round(total_in, 2),
        },
    )
    return payments, spec, tx_id
