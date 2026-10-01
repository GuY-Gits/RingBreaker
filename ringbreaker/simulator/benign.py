"""Hard negative (benign look-alike groups) generator for RingBreaker Simulator v2.

Creates legitimate groups whose payment and graph patterns mirror detectors
(shared devices, fan-in collectors, lockstep rhythms, closed loops, pass-throughs).
Every benign payment is legitimate (is_fraud = 0, ring_id = None).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from ringbreaker.simulator.config import SimulatorConfig
from ringbreaker.simulator.rings.base import make_payment


class BenignGroupGenerator:
    def __init__(
        self,
        config: SimulatorConfig,
        df_users: pd.DataFrame,
        user_meta: Dict[str, Dict[str, Any]],
        available_users: Set[str],
    ):
        self.config = config
        self.df_users = df_users
        self.user_meta = user_meta
        self.available_users = available_users
        self.benign_groups_meta: List[Dict[str, Any]] = []

    def _reserve(self, n: int) -> List[str]:
        chosen = random.sample(list(self.available_users), min(n, len(self.available_users)))
        for u in chosen:
            self.available_users.discard(u)
        return chosen

    def generate_all(self, tx_counter_start: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        cfg = self.config
        payments: List[Dict[str, Any]] = []
        tx_id = tx_counter_start
        start_date = cfg.base_start_date
        sim_days = cfg.sim_days

        scale = max(0.1, min(1.0, cfg.num_users / 5000.0)) if cfg.num_users < 5000 else 1.0

        # 1. Household sharing a device/address (40-60 groups)
        # 2-5 accounts, shared device/address/IP; normal payments among themselves
        n_households = max(2, int(round(random.randint(cfg.benign_counts["household_sharing"][0], cfg.benign_counts["household_sharing"][1]) * scale)))
        for h_idx in range(n_households):
            size = random.randint(2, 5)
            members = self._reserve(size)
            if len(members) < 2:
                continue

            shared_dev = f"DEV_HOME_{h_idx + 1:03d}"
            shared_addr = f"Flat {100 + h_idx}, Green Acres, Indiranagar, Bengaluru - 560038"
            shared_ip = f"192.168.1.{random.randint(2, 250)}"

            for m in members:
                idx = self.df_users.index[self.df_users["user_id"] == m][0]
                self.df_users.at[idx, "device_id"] = shared_dev
                self.df_users.at[idx, "address"] = shared_addr
                self.df_users.at[idx, "ip_address"] = shared_ip
                self.user_meta[m]["device_id"] = shared_dev
                self.user_meta[m]["address"] = shared_addr
                self.user_meta[m]["ip_address"] = shared_ip

            # Multiple intra-household payments over the timeline
            num_tx = random.randint(3, 8)
            for _ in range(num_tx):
                s, r = random.sample(members, 2)
                day = random.uniform(1, sim_days - 2)
                tx_time = start_date + timedelta(days=day, hours=random.randint(8, 22), minutes=random.randint(0, 59))
                amt = float(round(random.uniform(100, 1500), 2))
                payments.append(make_payment(
                    tx_id=tx_id, sender=s, receiver=r, amount=amt, when=tx_time,
                    user_meta=self.user_meta, is_fraud=0, ring_id=None
                ))
                tx_id += 1

            self.benign_groups_meta.append({
                "group_id": f"BENIGN_HOUSEHOLD_{h_idx+1:03d}",
                "type": "household_sharing",
                "members": members,
            })

        # 2. Landlord / tutor / club collector (10-15 groups)
        # 4-15 payers on fixed dates, recurring amounts
        n_collectors = max(1, int(round(random.randint(cfg.benign_counts["recurring_collector"][0], cfg.benign_counts["recurring_collector"][1]) * scale)))
        for c_idx in range(n_collectors):
            collector = self._reserve(1)
            if not collector:
                continue
            col_id = collector[0]
            n_payers = random.randint(4, 12)
            payers = self._reserve(n_payers)
            if len(payers) < 3:
                continue

            fixed_day = random.randint(1, 5)
            base_amt = float(round(random.choice([1500, 2500, 5000, 12000, 18000]), 2))

            for month in (1, 2, 3):
                for p in payers:
                    tx_time = datetime(2026, month, fixed_day, random.randint(9, 18), random.randint(0, 59))
                    amt = round(base_amt + random.uniform(-50, 50), 2)
                    payments.append(make_payment(
                        tx_id=tx_id, sender=p, receiver=col_id, amount=amt, when=tx_time,
                        user_meta=self.user_meta, is_fraud=0, ring_id=None
                    ))
                    tx_id += 1

            self.benign_groups_meta.append({
                "group_id": f"BENIGN_COLLECTOR_{c_idx+1:03d}",
                "type": "recurring_collector",
                "members": [col_id] + payers,
            })

        # 3. Event or trip split (10-15 groups)
        # One organizer collects, later refunds a few
        n_trips = max(1, int(round(random.randint(cfg.benign_counts["trip_split"][0], cfg.benign_counts["trip_split"][1]) * scale)))
        for t_idx in range(n_trips):
            organizer = self._reserve(1)
            if not organizer:
                continue
            org_id = organizer[0]
            friends = self._reserve(random.randint(4, 8))
            if len(friends) < 3:
                continue

            trip_day = random.uniform(5, sim_days - 10)
            base_t = start_date + timedelta(days=trip_day, hours=10)

            # Collection phase
            for f in friends:
                tx_time = base_t + timedelta(hours=random.uniform(0.5, 8.0))
                amt = float(round(random.uniform(2000, 6000), 2))
                payments.append(make_payment(
                    tx_id=tx_id, sender=f, receiver=org_id, amount=amt, when=tx_time,
                    user_meta=self.user_meta, is_fraud=0, ring_id=None
                ))
                tx_id += 1

            # Partial refund phase (a few days later)
            refund_t = base_t + timedelta(days=random.randint(2, 5))
            for f in random.sample(friends, random.randint(1, 3)):
                tx_time = refund_t + timedelta(hours=random.uniform(0.5, 4.0))
                refund_amt = float(round(random.uniform(300, 900), 2))
                payments.append(make_payment(
                    tx_id=tx_id, sender=org_id, receiver=f, amount=refund_amt, when=tx_time,
                    user_meta=self.user_meta, is_fraud=0, ring_id=None
                ))
                tx_id += 1

            self.benign_groups_meta.append({
                "group_id": f"BENIGN_TRIP_{t_idx+1:03d}",
                "type": "trip_split",
                "members": [org_id] + friends,
            })

        # 4. Salary pass-through (15-25 groups)
        # Receives salary, pays rent/bills within hours
        n_salary = max(1, int(round(random.randint(cfg.benign_counts["salary_passthrough"][0], cfg.benign_counts["salary_passthrough"][1]) * scale)))
        for s_idx in range(n_salary):
            emp = self._reserve(1)
            employer = self._reserve(1)
            billee = self._reserve(1)
            if not (emp and employer and billee):
                continue
            e_id, corp_id, b_id = emp[0], employer[0], billee[0]

            for month in (1, 2, 3):
                sal_day = 1 if month > 1 else 31
                try:
                    sal_time = datetime(2026, month, sal_day, 10, 0, 0)
                except ValueError:
                    sal_time = datetime(2026, month, 1, 10, 0, 0)

                sal_amt = float(round(random.uniform(45000, 95000), 2))
                payments.append(make_payment(
                    tx_id=tx_id, sender=corp_id, receiver=e_id, amount=sal_amt, when=sal_time,
                    user_meta=self.user_meta, is_fraud=0, ring_id=None
                ))
                tx_id += 1

                # Rapid pass-through: pays rent / loan within 1-6 hours
                pass_time = sal_time + timedelta(hours=random.uniform(1.0, 6.0))
                rent_amt = float(round(sal_amt * random.uniform(0.30, 0.45), 2))
                payments.append(make_payment(
                    tx_id=tx_id, sender=e_id, receiver=b_id, amount=rent_amt, when=pass_time,
                    user_meta=self.user_meta, is_fraud=0, ring_id=None
                ))
                tx_id += 1

            self.benign_groups_meta.append({
                "group_id": f"BENIGN_SALARY_{s_idx+1:03d}",
                "type": "salary_passthrough",
                "members": [corp_id, e_id, b_id],
            })

        # 5. Office lunch / savings circle (5-8 groups)
        # Colleagues joined within days, pay each other on weekly rhythm
        n_circles = max(1, int(round(random.randint(cfg.benign_counts["office_savings_circle"][0], cfg.benign_counts["office_savings_circle"][1]) * scale)))
        for o_idx in range(n_circles):
            colleagues = self._reserve(random.randint(4, 7))
            if len(colleagues) < 3:
                continue

            # Align signups within 3 days
            base_signup = start_date + timedelta(days=random.randint(5, 20))
            for c in colleagues:
                idx = self.df_users.index[self.df_users["user_id"] == c][0]
                new_s = base_signup + timedelta(hours=random.uniform(0, 72))
                self.df_users.at[idx, "signup_timestamp"] = new_s.strftime("%Y-%m-%d %H:%M:%S")
                self.user_meta[c]["signup_timestamp"] = new_s.strftime("%Y-%m-%d %H:%M:%S")

            # Weekly lunch round on Fridays at 13:00
            for week in range(2, 11):
                friday = start_date + timedelta(days=week * 7 + 4, hours=13)
                payer = colleagues[week % len(colleagues)]
                for c in colleagues:
                    if c != payer:
                        tx_time = friday + timedelta(minutes=random.randint(5, 45))
                        amt = float(round(random.uniform(250, 650), 2))
                        payments.append(make_payment(
                            tx_id=tx_id, sender=c, receiver=payer, amount=amt, when=tx_time,
                            user_meta=self.user_meta, is_fraud=0, ring_id=None
                        ))
                        tx_id += 1

            self.benign_groups_meta.append({
                "group_id": f"BENIGN_SAVINGS_{o_idx+1:03d}",
                "type": "office_savings_circle",
                "members": colleagues,
            })

        # 6. Friend group settling up (20-30 groups)
        # A -> B -> C -> A within a day, small amounts
        n_friends = max(1, int(round(random.randint(cfg.benign_counts["settling_up_loop"][0], cfg.benign_counts["settling_up_loop"][1]) * scale)))
        for f_idx in range(n_friends):
            group = self._reserve(random.randint(3, 4))
            if len(group) < 3:
                continue

            day = random.uniform(2, sim_days - 3)
            base_t = start_date + timedelta(days=day, hours=random.randint(18, 21))

            cycle = group + [group[0]]
            cur_t = base_t
            for i in range(len(cycle) - 1):
                cur_t += timedelta(minutes=random.randint(15, 90))
                amt = float(round(random.uniform(150, 850), 2))
                payments.append(make_payment(
                    tx_id=tx_id, sender=cycle[i], receiver=cycle[i + 1], amount=amt, when=cur_t,
                    user_meta=self.user_meta, is_fraud=0, ring_id=None
                ))
                tx_id += 1

            self.benign_groups_meta.append({
                "group_id": f"BENIGN_LOOP_{f_idx+1:03d}",
                "type": "settling_up_loop",
                "members": group,
            })

        # 7. Legit high-velocity sender (10 groups)
        # Small shop owner paying suppliers
        n_merchants = max(1, int(round(cfg.benign_counts["high_velocity_sender"][0] * scale)))
        for m_idx in range(n_merchants):
            merchant = self._reserve(1)
            suppliers = self._reserve(random.randint(5, 10))
            if not merchant or len(suppliers) < 3:
                continue
            m_id = merchant[0]

            for day in range(10, 85, 3):
                burst_t = start_date + timedelta(days=day, hours=11)
                for sup in random.sample(suppliers, min(len(suppliers), random.randint(3, 6))):
                    tx_time = burst_t + timedelta(minutes=random.randint(1, 30))
                    amt = float(round(random.uniform(1200, 8000), 2))
                    payments.append(make_payment(
                        tx_id=tx_id, sender=m_id, receiver=sup, amount=amt, when=tx_time,
                        user_meta=self.user_meta, is_fraud=0, ring_id=None
                    ))
                    tx_id += 1

            self.benign_groups_meta.append({
                "group_id": f"BENIGN_MERCHANT_{m_idx+1:03d}",
                "type": "high_velocity_sender",
                "members": [m_id] + suppliers,
            })

        return payments, self.benign_groups_meta, tx_id
