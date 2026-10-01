"""Synthetic population generation for RingBreaker Simulator v2.

Generates accounts, personas, identity fragments, and community-based social graph.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ringbreaker.simulator.config import SimulatorConfig


class Population:
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.users_df: pd.DataFrame = pd.DataFrame()
        self.contacts: Dict[str, List[str]] = {}
        self.family_clusters: List[List[str]] = []
        self.workplace_clusters: List[List[str]] = []

    def generate(self) -> pd.DataFrame:
        cfg = self.config
        num_users = cfg.num_users
        start_date = cfg.base_start_date
        sim_days = cfg.sim_days

        cities = ["Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
        streets = [
            "MG Road", "Station Road", "Park Street", "Ring Road", "Nehru Nagar",
            "Indira Nagar", "Civil Lines", "Brigade Road", "FC Road", "Bandra West"
        ]

        # Partition personas
        personas: List[str] = []
        for persona, share in cfg.persona_shares.items():
            count = int(round(num_users * share))
            personas.extend([persona] * count)
        # Pad or trim to exactly num_users
        while len(personas) < num_users:
            personas.append("casual")
        personas = personas[:num_users]
        random.shuffle(personas)

        users = []
        for i in range(1, num_users + 1):
            u_id = f"U{i:05d}"
            persona = personas[i - 1]

            # Realistic signups: 60% joined before start_date (-365 to 0 days),
            # 40% join dynamically throughout the 90 days.
            if random.random() < 0.60:
                signup_offset = -random.uniform(0, 365)
            else:
                signup_offset = random.uniform(0, sim_days * 0.95)
            signup_ts = start_date + timedelta(days=signup_offset, seconds=random.randint(0, 86400))

            phone = f"+9198{random.randint(10000000, 99999999)}"
            city = random.choice(cities)
            street = random.choice(streets)
            pin = random.randint(400001, 700001)
            address = f"Flat {random.randint(101, 999)}, {street}, {city} - {pin}"
            device_id = f"DEV_{random.randint(100000, 999999)}"
            ip_address = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"

            users.append({
                "user_id": u_id,
                "phone": phone,
                "address": address,
                "device_id": device_id,
                "ip_address": ip_address,
                "signup_timestamp": signup_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "user_role": "normal",
                "persona": persona,
            })

        self.users_df = pd.DataFrame(users)
        self._build_social_graph()
        return self.users_df

    def _build_social_graph(self) -> None:
        """Construct overlapping communities: family, workplace, friends."""
        cfg = self.config
        all_ids = list(self.users_df["user_id"].values)
        n = len(all_ids)

        contacts_map: Dict[str, Set[str]] = {uid: set() for uid in all_ids}

        # 1. Family clusters (2-6 members each)
        unassigned = list(all_ids)
        random.shuffle(unassigned)
        while unassigned:
            f_size = min(len(unassigned), random.randint(cfg.community_sizes["family"][0], cfg.community_sizes["family"][1]))
            if f_size < 2:
                break
            cluster = [unassigned.pop() for _ in range(f_size)]
            self.family_clusters.append(cluster)
            for m1 in cluster:
                for m2 in cluster:
                    if m1 != m2:
                        contacts_map[m1].add(m2)

        # 2. Workplace clusters (10-40 members each)
        random.shuffle(all_ids)
        idx = 0
        while idx < n:
            w_size = min(n - idx, random.randint(cfg.community_sizes["workplace"][0], cfg.community_sizes["workplace"][1]))
            if w_size < 4:
                break
            cluster = all_ids[idx : idx + w_size]
            idx += w_size
            self.workplace_clusters.append(cluster)
            # Add subset of coworkers as contacts
            for m1 in cluster:
                colleagues = random.sample(cluster, min(len(cluster), random.randint(3, 8)))
                for m2 in colleagues:
                    if m1 != m2:
                        contacts_map[m1].add(m2)
                        contacts_map[m2].add(m1)

        # 3. Friends (5-15 random acquaintances)
        for uid in all_ids:
            target_friends = random.randint(cfg.community_sizes["friends"][0], cfg.community_sizes["friends"][1])
            curr = len(contacts_map[uid])
            if curr < target_friends:
                candidates = random.sample(all_ids, min(n, target_friends - curr + 5))
                for c in candidates:
                    if c != uid:
                        contacts_map[uid].add(c)
                        contacts_map[c].add(uid)
                        if len(contacts_map[uid]) >= target_friends:
                            break

        self.contacts = {uid: sorted(list(c)) for uid, c in contacts_map.items()}
