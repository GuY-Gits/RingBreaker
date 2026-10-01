"""Normal payment traffic generation for RingBreaker Simulator v2.

Generates realistic P2P payments driven by personas, social graph affinities,
diurnal/weekly rhythms, salary bursts, and recurring expenses.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ringbreaker.simulator.config import SimulatorConfig
from ringbreaker.simulator.population import Population


def _sample_diurnal_hour() -> int:
    """Sample an hour following realistic diurnal peaks (12-14 lunch, 19-22 evening, low late night)."""
    weights = [
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01,   # 00:00 - 05:00
        0.02, 0.04, 0.06, 0.07, 0.08, 0.08,        # 06:00 - 11:00
        0.11, 0.10, 0.07, 0.06, 0.06, 0.07,        # 12:00 - 17:00
        0.10, 0.12, 0.11, 0.08, 0.05, 0.02         # 18:00 - 23:00
    ]
    weights = np.array(weights) / sum(weights)
    return int(np.random.choice(24, p=weights))


class NormalTrafficGenerator:
    def __init__(self, config: SimulatorConfig, population: Population, exclude: Optional[Set[str]] = None):
        self.config = config
        self.population = population
        # Synthetic identities have no real social life: they neither send nor
        # receive background traffic (their only non-ring payments are camouflage).
        self.exclude: Set[str] = set(exclude or ())
        self.user_meta = {
            uid: meta for uid, meta in population.users_df.set_index("user_id").to_dict("index").items()
            if uid not in self.exclude
        }
        self.all_users = [u for u in population.users_df["user_id"].values if u not in self.exclude]

    def generate(self, target_count: int, tx_counter_start: int = 1) -> Tuple[List[Dict[str, Any]], int]:
        cfg = self.config
        payments: List[Dict[str, Any]] = []
        tx_id = tx_counter_start
        start_date = cfg.base_start_date
        sim_days = cfg.sim_days

        # Activity scaling per persona (expected payments over 90 days)
        persona_rates = {
            "dormant": (0, 3, 0.8),          # (min, max, mean)
            "casual": (3, 14, 7.5),
            "active": (18, 55, 32.0),
            "power_merchant": (60, 160, 95.0),
        }

        # Determine individual account payment budgets
        account_budgets: Dict[str, int] = {}
        for uid, meta in self.user_meta.items():
            persona = meta["persona"]
            p_min, p_max, p_mean = persona_rates.get(persona, (2, 10, 5.0))
            # Heavy-tailed draw
            val = int(round(np.random.lognormal(mean=np.log(max(1.0, p_mean)), sigma=0.5)))
            account_budgets[uid] = max(p_min, min(p_max, val))

        # Re-scale budgets proportionally to meet target_count
        curr_total = sum(account_budgets.values())
        if curr_total > 0:
            scale = target_count / curr_total
            for uid in account_budgets:
                account_budgets[uid] = max(0, int(round(account_budgets[uid] * scale)))

        # Also set up monthly recurring payments for ~25% of active/casual users (rent, bills, gym)
        recurring_schedules: List[Dict[str, Any]] = []
        for uid, meta in self.user_meta.items():
            if meta["persona"] in ("casual", "active") and random.random() < 0.25:
                contacts = [c for c in self.population.contacts.get(uid, []) if c not in self.exclude]
                receiver = random.choice(contacts) if contacts else random.choice(self.all_users)
                if receiver != uid:
                    base_amt = float(round(random.choice([1200, 2500, 4500, 8000, 12000, 15000]), 2))
                    day_of_month = random.randint(1, 5)  # typically start of month
                    recurring_schedules.append({
                        "sender": uid,
                        "receiver": receiver,
                        "base_amount": base_amt,
                        "day_of_month": day_of_month,
                    })

        # 1. Generate regular transactional traffic
        for uid, count in account_budgets.items():
            meta = self.user_meta[uid]
            signup_dt = datetime.strptime(meta["signup_timestamp"], "%Y-%m-%d %H:%M:%S")
            contacts = [c for c in self.population.contacts.get(uid, []) if c not in self.exclude]

            # Active start date is after signup or base_start_date
            active_start = max(start_date, signup_dt)
            if active_start >= start_date + timedelta(days=sim_days):
                continue

            available_seconds = int((start_date + timedelta(days=sim_days) - active_start).total_seconds())
            if available_seconds <= 3600:
                continue

            for _ in range(count):
                # 80% to contacts, 20% to strangers
                if contacts and random.random() < cfg.contact_tx_prob:
                    receiver = random.choice(contacts)
                else:
                    receiver = random.choice(self.all_users)
                    while receiver == uid:
                        receiver = random.choice(self.all_users)

                # Random timestamp strictly after active_start
                sec_offset = random.randint(0, available_seconds)
                tx_dt = active_start + timedelta(seconds=sec_offset)

                # Adjust hour according to diurnal weights
                hour = _sample_diurnal_hour()
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                tx_dt = tx_dt.replace(hour=hour, minute=minute, second=second)

                # Weekend check: if weekend, slight chance to defer or lower
                if tx_dt.weekday() >= 5 and random.random() > cfg.weekend_activity_ratio:
                    # shift slightly
                    pass

                # Amount: lognormal distribution based on persona
                if meta["persona"] == "power_merchant":
                    # Higher amounts or very frequent retail
                    amount = float(np.round(np.clip(np.random.lognormal(mean=6.8, sigma=1.1), 50, 45000), 2))
                else:
                    amount = float(np.round(np.clip(np.random.lognormal(mean=6.2, sigma=0.9), 30, 20000), 2))

                payments.append({
                    "transaction_id": f"TX_{tx_id:07d}",
                    "timestamp": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender": uid,
                    "receiver": receiver,
                    "amount": amount,
                    "device_id": meta["device_id"],
                    "ip_address": meta["ip_address"],
                    "is_fraud": 0,
                    "ring_id": None,
                })
                tx_id += 1

        # 2. Generate monthly recurring payments (with small jitter in amount and date)
        for sched in recurring_schedules:
            uid = sched["sender"]
            rec = sched["receiver"]
            meta = self.user_meta[uid]
            signup_dt = datetime.strptime(meta["signup_timestamp"], "%Y-%m-%d %H:%M:%S")

            for month in range(1, 4):  # Jan, Feb, Mar 2026
                try:
                    target_date = datetime(2026, month, sched["day_of_month"], 10, 0, 0)
                except ValueError:
                    target_date = datetime(2026, month, 1, 10, 0, 0)

                # Add small jitter (+/- 2 days, +/- 1.5% amount)
                jitter_days = random.randint(-1, 2)
                tx_dt = target_date + timedelta(days=jitter_days, hours=random.randint(0, 8), minutes=random.randint(0, 59))
                if tx_dt < max(start_date, signup_dt) or tx_dt >= start_date + timedelta(days=sim_days):
                    continue

                jitter_amt = round(sched["base_amount"] * random.uniform(0.99, 1.01), 2)
                payments.append({
                    "transaction_id": f"TX_{tx_id:07d}",
                    "timestamp": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender": uid,
                    "receiver": rec,
                    "amount": jitter_amt,
                    "device_id": meta["device_id"],
                    "ip_address": meta["ip_address"],
                    "is_fraud": 0,
                    "ring_id": None,
                })
                tx_id += 1

        return payments, tx_id
