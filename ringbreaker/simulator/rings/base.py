"""Base dataclass and shared helpers for fraud ring generators."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class RingSpec:
    ring_id: str
    family: str
    split: str          # "train", "validation", "heldout"
    variant: str        # e.g., "standard", "held_out_novel", "two_wave"
    members: List[str]
    start_time: str
    end_time: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ring_id": self.ring_id,
            "ring_type": self.family,
            "family": self.family,
            "split": self.split,
            "variant": self.variant,
            "members": ";".join(self.members),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "params": json.dumps(self.params),
        }


def make_payment(
    tx_id: int,
    sender: str,
    receiver: str,
    amount: float,
    when: datetime,
    user_meta: Dict[str, Dict[str, Any]],
    ring_id: Optional[str] = None,
    is_fraud: int = 1,
    override_device: Optional[str] = None,
    override_ip: Optional[str] = None,
) -> Dict[str, Any]:
    sender_meta = user_meta[sender]
    dev = override_device if override_device is not None else sender_meta["device_id"]
    ip = override_ip if override_ip is not None else sender_meta["ip_address"]
    return {
        "transaction_id": f"TX_{tx_id:07d}",
        "timestamp": when.strftime("%Y-%m-%d %H:%M:%S"),
        "sender": sender,
        "receiver": receiver,
        "amount": round(float(amount), 2),
        "device_id": dev,
        "ip_address": ip,
        "is_fraud": is_fraud,
        "ring_id": ring_id,
    }


def add_camouflage_payments(
    ring_accounts: List[str],
    user_meta: Dict[str, Dict[str, Any]],
    contacts_map: Dict[str, List[str]],
    all_users: List[str],
    base_start_date: datetime,
    sim_days: int,
    tx_counter_start: int,
    camouflage_ratio: float = 0.60,
    tx_count_range: Tuple[int, int] = (2, 10),
) -> Tuple[List[Dict[str, Any]], int]:
    """60% of ring accounts make 2-10 ordinary payments to normal contacts over weeks,

    so ring members are not 'only fraud'.
    """
    payments: List[Dict[str, Any]] = []
    tx_id = tx_counter_start

    for acc in ring_accounts:
        if random.random() > camouflage_ratio:
            continue
        meta = user_meta[acc]
        signup_dt = datetime.strptime(meta["signup_timestamp"], "%Y-%m-%d %H:%M:%S")
        valid_start = max(base_start_date, signup_dt)
        time_span = (base_start_date + timedelta(days=sim_days)) - valid_start
        if time_span.total_seconds() <= 7200:
            continue

        num_camo = random.randint(tx_count_range[0], tx_count_range[1])
        acc_contacts = contacts_map.get(acc, [])
        for _ in range(num_camo):
            rec = random.choice(acc_contacts) if acc_contacts else random.choice(all_users)
            while rec == acc:
                rec = random.choice(all_users)
            offset_sec = random.randint(0, int(time_span.total_seconds()))
            tx_time = valid_start + timedelta(seconds=offset_sec)
            amt = float(round(random.uniform(50, 2500), 2))
            payments.append(make_payment(
                tx_id=tx_id,
                sender=acc,
                receiver=rec,
                amount=amt,
                when=tx_time,
                user_meta=user_meta,
                ring_id=None,
                is_fraud=0,
            ))
            tx_id += 1

    return payments, tx_id
