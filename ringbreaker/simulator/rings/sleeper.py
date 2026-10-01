"""Synthetic-identity sleeper batch fraud ring family."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd

from ringbreaker.simulator.rings.base import RingSpec, make_payment


def generate_sleeper_instance(
    ring_id: str,
    members: List[str],
    cashout_accounts: List[str],
    df_users: pd.DataFrame,
    user_meta: Dict[str, Dict[str, Any]],
    start_date: datetime,
    split: str,
    tx_counter_start: int,
    signup_day_start: int = 30,
) -> Tuple[List[Dict[str, Any]], RingSpec, int]:
    """Generates a synthetic-identity sleeper ring.

    - 8-20 accounts created within 1-7 days of each other, sharing devices/addresses
    - Regular intra-cluster payment rhythm every 1-4 days
    - Sleep period of 2-6 weeks
    - Synchronized bust-out where members send large sums to a collector who exits
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start

    num_devices = max(1, min(4, len(members) // 4))
    devices = [f"DEV_SYN_{ring_id}_{i+1}" for i in range(num_devices)]
    phones = [f"+9190000{random.randint(10000, 99999)}" for _ in range(max(1, num_devices // 2 + 1))]
    addresses = [
        f"Flat {random.randint(10, 50)}, Syndicate Enclave, Ring Road, Pune - 411001",
        f"Flat {random.randint(51, 99)}, Syndicate Enclave, Ring Road, Pune - 411001",
    ]
    ips = [f"10.8.{random.randint(1, 250)}.{random.randint(2, 250)}" for _ in range(2)]

    signup_base = start_date + timedelta(days=signup_day_start)
    signup_spread_days = random.randint(1, 7)

    all_ring_users = members + cashout_accounts
    for i, uid in enumerate(all_ring_users):
        idx_list = df_users.index[df_users["user_id"] == uid].tolist()
        if not idx_list:
            continue
        idx = idx_list[0]
        signup = signup_base + timedelta(hours=random.uniform(0, signup_spread_days * 24))
        df_users.at[idx, "signup_timestamp"] = signup.strftime("%Y-%m-%d %H:%M:%S")
        df_users.at[idx, "device_id"] = devices[i % len(devices)]
        df_users.at[idx, "phone"] = phones[i % len(phones)]
        df_users.at[idx, "address"] = addresses[i % len(addresses)]
        df_users.at[idx, "ip_address"] = ips[i % len(ips)]
        df_users.at[idx, "user_role"] = "mule" if (uid in cashout_accounts or uid == members[0]) else "ring_member"
        # Update user_meta in place
        user_meta[uid]["signup_timestamp"] = signup.strftime("%Y-%m-%d %H:%M:%S")
        user_meta[uid]["device_id"] = devices[i % len(devices)]
        user_meta[uid]["phone"] = phones[i % len(phones)]
        user_meta[uid]["address"] = addresses[i % len(addresses)]
        user_meta[uid]["ip_address"] = ips[i % len(ips)]

    # 1. Warm-up rhythm: small regular payments every 2-5 days
    rhythm_step = random.randint(2, 5)
    rhythm_start_day = signup_day_start + signup_spread_days + 3
    sleep_weeks = random.randint(2, 4)
    bust_out_day = min(88, rhythm_start_day + sleep_weeks * 7)
    if ring_id == "RING_SLEEPER":
        # Ensure the demo/eval sleeper bust out is placed in the held-out live stream (day 79-84)
        bust_out_day = random.randint(79, 83)
    if bust_out_day <= rhythm_start_day + 10:
        bust_out_day = min(88, rhythm_start_day + 14)

    for d in range(rhythm_start_day, min(bust_out_day - 1, 86), rhythm_step):
        base_t = start_date + timedelta(days=d, hours=random.randint(19, 22))
        num_tx = random.choice([1, 1, 2])
        for _ in range(num_tx):
            s, r = random.sample(members, 2)
            tx_time = base_t + timedelta(minutes=random.randint(0, 30), seconds=random.randint(0, 59))
            amt = float(round(random.uniform(100, 450), 2))
            payments.append(make_payment(
                tx_id=tx_id,
                sender=s,
                receiver=r,
                amount=amt,
                when=tx_time,
                user_meta=user_meta,
                ring_id=ring_id,
                is_fraud=1,
            ))
            tx_id += 1

    # 2. Bust-out: coordinated rush to the collector, then rapid cash-out
    collector = members[0]
    bust_time = start_date + timedelta(days=bust_out_day, hours=random.randint(11, 16))
    total_bust = 0.0
    latest_feed = bust_time

    for m in members[1:]:
        t_feed = bust_time + timedelta(minutes=random.randint(1, 90), seconds=random.randint(0, 59))
        latest_feed = max(latest_feed, t_feed)
        amt = float(round(random.uniform(3500, 6800), 2))
        total_bust += amt
        payments.append(make_payment(
            tx_id=tx_id,
            sender=m,
            receiver=collector,
            amount=amt,
            when=t_feed,
            user_meta=user_meta,
            ring_id=ring_id,
            is_fraud=1,
        ))
        tx_id += 1

    # Collector pays cash-out accounts shortly after
    c_time = latest_feed
    for j, c in enumerate(cashout_accounts):
        c_time += timedelta(minutes=random.randint(2, 8))
        share = total_bust * (0.48 if j == 0 else 0.46)
        payments.append(make_payment(
            tx_id=tx_id,
            sender=collector,
            receiver=c,
            amount=round(share, 2),
            when=c_time,
            user_meta=user_meta,
            ring_id=ring_id,
            is_fraud=1,
        ))
        tx_id += 1

    spec = RingSpec(
        ring_id=ring_id,
        family="synthetic_sleeper",
        split=split,
        variant="sleeper_bustout",
        members=all_ring_users,
        start_time=min(p["timestamp"] for p in payments),
        end_time=max(p["timestamp"] for p in payments),
        params={
            "member_count": len(members),
            "cashout_count": len(cashout_accounts),
            "sleep_weeks": sleep_weeks,
            "bust_out_day": bust_out_day,
            "total_bust": round(total_bust, 2),
        },
    )
    return payments, spec, tx_id
