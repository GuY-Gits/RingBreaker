"""RingBreaker Simulator v2: Realistic Synthetic Payment Generator.

Orchestrates population generation, benign look-alike groups, randomized fraud ring families,
normal traffic, camouflage, and label noise.
Outputs:
  - data/users.csv
  - data/payments.csv
  - data/rings.csv
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ringbreaker.simulator.benign import BenignGroupGenerator
from ringbreaker.simulator.config import DEFAULT_CONFIG, SimulatorConfig
from ringbreaker.simulator.labels import apply_label_noise, rings_to_dataframe
from ringbreaker.simulator.normal import NormalTrafficGenerator
from ringbreaker.simulator.population import Population
from ringbreaker.simulator.rings import (
    RingSpec,
    add_camouflage_payments,
    generate_chain_instance,
    generate_distributed_device_instance,
    generate_fan_in_instance,
    generate_loop_instance,
    generate_scam_instance,
    generate_sleeper_instance,
    generate_slow_chain_instance,
    generate_star_instance,
)
from ringbreaker.simulator.validate import validate_simulator_data

# Re-export key constants for backward compatibility
NUM_USERS = DEFAULT_CONFIG.num_users
NUM_TRANSACTIONS = DEFAULT_CONFIG.target_payments
SIMULATION_DAYS = DEFAULT_CONFIG.sim_days
HELD_OUT_START_DAY = DEFAULT_CONFIG.held_out_start_day
BASE_START_DATE = DEFAULT_CONFIG.base_start_date
RANDOM_SEED = DEFAULT_CONFIG.random_seed
OUTPUT_DIR = DEFAULT_CONFIG.output_dir


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_users(num_users: int = NUM_USERS, sim_days: int = SIMULATION_DAYS, start_date: datetime = BASE_START_DATE) -> pd.DataFrame:
    """Backward-compatible user generation helper."""
    cfg = SimulatorConfig(num_users=num_users, sim_days=sim_days, base_start_date=start_date)
    pop = Population(cfg)
    return pop.generate()


def generate_all_rings_v2(
    df_users: pd.DataFrame,
    population: Population,
    config: SimulatorConfig,
    tx_counter_start: int = 1,
) -> Tuple[List[Dict[str, Any]], List[RingSpec], int]:
    """Generates all randomized fraud ring families distributed across train (70%),

    validation (15%), and held-out (15%), plus novel families strictly in held-out.
    """
    payments: List[Dict[str, Any]] = []
    specs: List[RingSpec] = []
    tx_id = tx_counter_start

    user_meta = df_users.set_index("user_id").to_dict("index")
    available_users: Set[str] = set(df_users["user_id"].values)

    def reserve(n: int) -> List[str]:
        chosen = random.sample(sorted(available_users), min(n, len(available_users)))
        for u in chosen:
            available_users.discard(u)
        return chosen

    def reserve_established(n: int) -> List[str]:
        """Accounts that existed well before the simulation (scam victims are established users)."""
        cutoff = (config.base_start_date - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
        pool = sorted(u for u in available_users if user_meta[u]["signup_timestamp"] <= cutoff)
        chosen = random.sample(pool, min(n, len(pool)))
        for u in chosen:
            available_users.discard(u)
        return chosen

    start_date = config.base_start_date
    sim_days = config.sim_days

    # Time splits (days)
    # Train: 0 to 63 (70%)
    # Validation: 63 to 76.5 (15%)
    # Heldout: 76.5 to 90 (15%)
    t_train_max = 63.0
    t_val_max = config.held_out_start_day
    t_sim_max = float(sim_days)

    ring_counter = 1
    all_fraud_accounts: Set[str] = set()

    def sample_start_time(split: str) -> datetime:
        if split == "train":
            day = random.uniform(5.0, t_train_max - 4.0)
        elif split == "validation":
            day = random.uniform(t_train_max + 0.5, t_val_max - 2.0)
        else:  # heldout
            day = random.uniform(t_val_max + 0.5, t_sim_max - 3.0)
        return start_date + timedelta(days=day, hours=random.randint(8, 22), minutes=random.randint(0, 59))

    # Standard families distribution: 70% train, 15% validation, 15% held-out
    splits_pool = ["train"] * 7 + ["validation"] * 2 + ["heldout"] * 2

    # Scaling factor for unit test configurations with small user populations
    scale = max(0.15, min(1.0, config.num_users / 5000.0)) if config.num_users < 5000 else 1.0

    # Reusable fraud syndicate account pool.
    # In real fraud ecosystems, syndicates operate mule networks that are reused across
    # cycles, campaigns, and structural topologies.
    syndicate_size = max(18, int(config.num_users * 0.015))
    syndicate_pool = reserve(syndicate_size)
    if len(syndicate_pool) < 10:
        syndicate_pool = list(available_users)[:max(5, len(available_users))]

    # 1. Closed Loops (5-7 instances)
    n_loops = max(1, int(round(random.randint(config.ring_family_counts["closed_loop"][0], config.ring_family_counts["closed_loop"][1]) * scale)))
    for _ in range(n_loops):
        split = random.choice(splits_pool)
        size = min(random.randint(3, 6), len(syndicate_pool))
        if size < 3:
            continue
        members = random.sample(syndicate_pool, size)
        all_fraud_accounts.update(members)
        r_id = f"RING_{ring_counter:03d}"
        ring_counter += 1
        t_start = sample_start_time(split)
        p, s, tx_id = generate_loop_instance(r_id, members, user_meta, t_start, split, tx_id)
        payments.extend(p)
        specs.append(s)

    # 2. Mule Chains (10-12 instances)
    n_chains = max(1, int(round(random.randint(config.ring_family_counts["mule_chain"][0], config.ring_family_counts["mule_chain"][1]) * scale)))
    has_named_mule_ho = False
    for chain_idx in range(n_chains):
        split = "heldout" if (chain_idx == n_chains - 1 and not has_named_mule_ho) else random.choice(splits_pool)
        hops = min(random.randint(3, 6), len(syndicate_pool))
        if hops < 3:
            continue
        members = random.sample(syndicate_pool, hops)
        all_fraud_accounts.update(members)
        waves = 2 if (split == "heldout" and not has_named_mule_ho) else random.choice([1, 1, 2, 2, 3])
        if split == "heldout" and not has_named_mule_ho:
            r_id = "RING_MULE_HO"
            has_named_mule_ho = True
        else:
            r_id = f"RING_{ring_counter:03d}"
            ring_counter += 1
        t_start = sample_start_time(split)
        p, s, tx_id = generate_chain_instance(r_id, members, user_meta, t_start, split, tx_id, waves=waves)
        payments.extend(p)
        specs.append(s)

    # 3. Fan-In Collectors + Cash-Out (8-10 instances)
    n_fanin = max(1, int(round(random.randint(config.ring_family_counts["fan_in_collector"][0], config.ring_family_counts["fan_in_collector"][1]) * scale)))
    for _ in range(n_fanin):
        split = random.choice(splits_pool)
        n_feed = min(random.randint(4, 7), len(syndicate_pool) - 3)
        if n_feed < 3:
            continue
        feeders = random.sample(syndicate_pool, n_feed)
        rem = [u for u in syndicate_pool if u not in feeders]
        collector = [random.choice(rem)]
        rem.remove(collector[0])
        n_exit = min(random.randint(1, 2), len(rem))
        exits = random.sample(rem, n_exit)
        col_id = collector[0]
        all_fraud_accounts.update(feeders + [col_id] + exits)
        r_id = f"RING_{ring_counter:03d}"
        ring_counter += 1
        t_start = sample_start_time(split)
        p, s, tx_id = generate_fan_in_instance(r_id, feeders, col_id, exits, user_meta, t_start, split, tx_id)
        payments.extend(p)
        specs.append(s)

    # 4. Device Farm Stars (4-6 instances)
    n_stars = max(1, int(round(random.randint(config.ring_family_counts["device_farm_star"][0], config.ring_family_counts["device_farm_star"][1]) * scale)))
    for _ in range(n_stars):
        split = random.choice(splits_pool)
        n_spk = min(random.randint(4, 8), len(syndicate_pool) - 2)
        if n_spk < 3:
            continue
        spokes = random.sample(syndicate_pool, n_spk)
        rem = [u for u in syndicate_pool if u not in spokes]
        hub_id = random.choice(rem)
        all_fraud_accounts.update(spokes + [hub_id])
        r_id = f"RING_{ring_counter:03d}"
        ring_counter += 1
        t_start = sample_start_time(split)
        p, s, tx_id = generate_star_instance(r_id, spokes, hub_id, user_meta, t_start, split, tx_id)
        payments.extend(p)
        specs.append(s)

    # 5. Scam Victims -> Mule (8-10 instances)
    n_scams = max(1, int(round(random.randint(config.ring_family_counts["scam_victim_mule"][0], config.ring_family_counts["scam_victim_mule"][1]) * scale)))
    for _ in range(n_scams):
        split = random.choice(splits_pool)
        victims = reserve_established(random.randint(1, 2))
        if not victims or len(syndicate_pool) < 2:
            continue
        mule = random.choice(syndicate_pool)
        rem = [u for u in syndicate_pool if u != mule]
        exit_acc = random.choice(rem)
        all_fraud_accounts.update(victims + [mule, exit_acc])
        r_id = f"RING_{ring_counter:03d}"
        ring_counter += 1
        t_start = sample_start_time(split)
        p, s, tx_id = generate_scam_instance(r_id, victims, mule, exit_acc, user_meta, t_start, split, tx_id)
        payments.extend(p)
        specs.append(s)

    # 6. Synthetic-Identity Sleepers (4-6 instances)
    n_sleepers = max(1, int(round(random.randint(config.ring_family_counts["synthetic_sleeper"][0], config.ring_family_counts["synthetic_sleeper"][1]) * scale)))
    for s_idx in range(n_sleepers):
        # sleepers span train into validation/heldout; split is marked train as signups/sleep are in train
        split = "train"
        members = reserve(random.randint(6, 8))
        cashout = reserve(2)
        if len(members) < 5 or not cashout:
            continue
        all_fraud_accounts.update(members + cashout)
        r_id = f"RING_{ring_counter:03d}" if s_idx > 0 else "RING_SLEEPER"
        ring_counter += 1
        signup_day = 8 + s_idx * 5
        p, s, tx_id = generate_sleeper_instance(
            r_id, members, cashout, df_users, user_meta, start_date, split, tx_id, signup_day_start=signup_day
        )
        payments.extend(p)
        specs.append(s)

    # 7. Novel Families (Strictly Held-Out)
    # 7a. Slow Chains (4-5 instances)
    n_slow = max(1, int(round(random.randint(config.novel_family_counts["slow_chain"][0], config.novel_family_counts["slow_chain"][1]) * scale)))
    for _ in range(n_slow):
        # Novel family: fresh mule accounts, never part of a training ring.
        members = reserve(random.randint(4, 6))
        if len(members) < 3:
            continue
        all_fraud_accounts.update(members)
        r_id = f"RING_{ring_counter:03d}"
        ring_counter += 1
        t_start = sample_start_time("heldout")
        p, s, tx_id = generate_slow_chain_instance(r_id, members, user_meta, t_start, tx_id)
        payments.extend(p)
        specs.append(s)

    # 7b. Distributed-Device Rings (3-5 instances)
    n_dist = max(1, int(round(random.randint(config.novel_family_counts["distributed_device_ring"][0], config.novel_family_counts["distributed_device_ring"][1]) * scale)))
    for _ in range(n_dist):
        # Novel family: fresh accounts, never part of a training ring.
        feeders = reserve(random.randint(4, 6))
        collector_acc = reserve(1)
        if len(feeders) < 3 or not collector_acc:
            continue
        col_id = collector_acc[0]
        all_fraud_accounts.update(feeders + [col_id])
        r_id = f"RING_{ring_counter:03d}"
        ring_counter += 1
        t_start = sample_start_time("heldout")
        p, s, tx_id = generate_distributed_device_instance(r_id, feeders, col_id, user_meta, t_start, tx_id, df_users=df_users)
        payments.extend(p)
        specs.append(s)

    # Tag roles in df_users
    for uid in all_fraud_accounts:
        idx = df_users.index[df_users["user_id"] == uid][0]
        if df_users.at[idx, "user_role"] == "normal":
            df_users.at[idx, "user_role"] = "ring_member"

    # Add camouflage payments for 60% of fraud accounts
    all_users_list = list(df_users["user_id"].values)
    camo_payments, tx_id = add_camouflage_payments(
        ring_accounts=list(all_fraud_accounts),
        user_meta=user_meta,
        contacts_map=population.contacts,
        all_users=all_users_list,
        base_start_date=start_date,
        sim_days=sim_days,
        tx_counter_start=tx_id,
        camouflage_ratio=config.ring_camouflage_ratio,
        tx_count_range=config.camouflage_tx_range,
    )
    payments.extend(camo_payments)

    return payments, specs, tx_id


def generate_dataset(config: SimulatorConfig = DEFAULT_CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates the full Simulator v2 synthetic dataset."""
    set_seed(config.random_seed)

    print(f"Generating synthetic population ({config.num_users} accounts)...")
    population = Population(config)
    df_users = population.generate()

    tx_id = 1

    # Step 1: Fraud rings (45-60 instances)
    print("Generating randomized fraud ring families and novel held-out rings...")
    fraud_payments, ring_specs, tx_id = generate_all_rings_v2(df_users, population, config, tx_counter_start=tx_id)

    # Step 2: Benign look-alike groups (80-120 groups)
    print("Generating hard-negative benign look-alike groups (80-120 groups)...")
    user_meta = df_users.set_index("user_id").to_dict("index")
    # Benign groups can use users who are not ring members
    non_ring_users = set(df_users[df_users["user_role"] == "normal"]["user_id"].values)
    benign_gen = BenignGroupGenerator(config, df_users, user_meta, non_ring_users)
    benign_payments, benign_meta, tx_id = benign_gen.generate_all(tx_counter_start=tx_id)

    # Step 3: Background normal traffic
    target_normal = max(100, config.target_payments - len(fraud_payments) - len(benign_payments))
    print(f"Generating background normal traffic (~{target_normal} payments)...")
    synthetic = {m for s in ring_specs if s.family == "synthetic_sleeper" for m in s.members}
    normal_gen = NormalTrafficGenerator(config, population, exclude=synthetic)
    normal_payments, tx_id = normal_gen.generate(target_count=target_normal, tx_counter_start=tx_id)

    # Step 4: Combine all payments, sort chronologically, and re-number TX IDs
    all_payments = fraud_payments + benign_payments + normal_payments
    df_payments = pd.DataFrame(all_payments)
    df_payments["dt"] = pd.to_datetime(df_payments["timestamp"])
    df_payments = df_payments.sort_values(by="dt", kind="stable").reset_index(drop=True)
    df_payments.drop(columns=["dt"], inplace=True)
    # Keep everything inside the simulated timeline (multi-wave rings can overrun it).
    sim_end = pd.Timestamp(config.base_start_date + timedelta(days=config.sim_days))
    df_payments = df_payments[pd.to_datetime(df_payments["timestamp"]) < sim_end].reset_index(drop=True)
    df_payments["transaction_id"] = [f"TX_{i+1:07d}" for i in range(len(df_payments))]

    # Tag benign look-alike payments with their group (evaluation metadata only).
    df_payments = tag_benign_groups(df_payments, benign_meta)
    df_benign = pd.DataFrame(
        [{"group_id": g["group_id"], "type": g["type"], "members": ";".join(g["members"])} for g in benign_meta]
    )

    # No account may transact before it exists.
    fix_signups(df_users, df_payments, config)

    # Step 5: Convert ring specs to DataFrame; split labels come from actual timestamps.
    df_rings = rings_to_dataframe(ring_specs)
    df_rings = resync_ring_windows(df_rings, df_payments, config)

    # Step 6: Apply label noise (label_observed)
    df_payments = apply_label_noise(
        df_payments,
        fraud_unlabelled_rate=config.fraud_unlabelled_rate,
        normal_labelled_fraud_rate=config.normal_labelled_fraud_rate,
        seed=config.random_seed,
    )

    # Step 7: Automated validation checks
    heldout_threshold = pd.Timestamp(config.base_start_date + timedelta(days=config.held_out_start_day))
    validate_simulator_data(df_users, df_payments, df_rings, config, heldout_threshold)

    df_payments.attrs["benign_groups"] = df_benign
    return df_users, df_payments, df_rings


def tag_benign_groups(df_payments: pd.DataFrame, benign_meta: List[Dict[str, Any]]) -> pd.DataFrame:
    """Mark payments made inside a benign look-alike group (both parties in the group)."""
    group_of: Dict[str, str] = {}
    for g in benign_meta:
        for m in g["members"]:
            group_of[m] = g["group_id"]
    s_group = df_payments["sender"].map(group_of)
    r_group = df_payments["receiver"].map(group_of)
    same = s_group.notna() & (s_group == r_group) & (df_payments["is_fraud"] == 0)
    df_payments["group_id"] = s_group.where(same, None)
    return df_payments


def fix_signups(df_users: pd.DataFrame, df_payments: pd.DataFrame, config: SimulatorConfig) -> None:
    """Move any signup that is later than the account's first payment to before it.

    Receivers and group members are drawn without regard to signup time, so
    without this ~1/3 of accounts would transact before they exist — an
    artefact a model could learn from.
    """
    ts = pd.to_datetime(df_payments["timestamp"])
    first = pd.concat([
        pd.DataFrame({"acc": df_payments["sender"], "ts": ts}),
        pd.DataFrame({"acc": df_payments["receiver"], "ts": ts}),
    ]).groupby("acc")["ts"].min()
    rng = np.random.default_rng(config.random_seed + 7)
    signup = pd.to_datetime(df_users["signup_timestamp"])
    earliest = pd.Timestamp(config.base_start_date - timedelta(days=365))
    for i, uid in enumerate(df_users["user_id"]):
        f = first.get(uid)
        if f is not None and signup.iloc[i] > f - pd.Timedelta(hours=1):
            new = max(earliest, f - pd.Timedelta(hours=float(rng.uniform(1, 21 * 24))))
            df_users.at[df_users.index[i], "signup_timestamp"] = new.strftime("%Y-%m-%d %H:%M:%S")


def resync_ring_windows(df_rings: pd.DataFrame, df_payments: pd.DataFrame, config: SimulatorConfig) -> pd.DataFrame:
    """Recompute each ring's start/end and split from the payments that survived."""
    ts = pd.to_datetime(df_payments["timestamp"])
    t0, t1 = ts.min(), ts.max()
    train_end = t0 + (t1 - t0) * 0.70
    stream_start = t0 + (t1 - t0) * 0.85
    fraud = df_payments[df_payments["ring_id"].notna()].assign(_ts=ts)
    spans = fraud.groupby("ring_id")["_ts"].agg(["min", "max"])
    keep = []
    for i, row in df_rings.iterrows():
        if row["ring_id"] not in spans.index:
            continue
        lo, hi = spans.loc[row["ring_id"]]
        df_rings.at[i, "start_time"] = lo.strftime("%Y-%m-%d %H:%M:%S")
        df_rings.at[i, "end_time"] = hi.strftime("%Y-%m-%d %H:%M:%S")
        if hi < train_end:
            split = "train"
        elif lo >= stream_start:
            split = "heldout"
        elif lo >= train_end and hi < stream_start:
            split = "validation"
        else:
            split = "spans_" + ("heldout" if hi >= stream_start else "validation")
        df_rings.at[i, "split"] = split
        keep.append(i)
    return df_rings.loc[keep].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RingBreaker Synthetic Payment Generator v2")
    parser.add_argument("--users", type=int, default=NUM_USERS, help="Total accounts")
    parser.add_argument("--transactions", type=int, default=NUM_TRANSACTIONS, help="Target payments")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Destination directory for CSVs")
    args = parser.parse_args()

    cfg = SimulatorConfig(
        num_users=args.users,
        target_payments=args.transactions,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    df_users, df_payments, df_rings = generate_dataset(cfg)

    users_path = os.path.join(cfg.output_dir, "users.csv")
    payments_path = os.path.join(cfg.output_dir, "payments.csv")
    rings_path = os.path.join(cfg.output_dir, "rings.csv")

    df_users.to_csv(users_path, index=False)
    df_payments.to_csv(payments_path, index=False)
    df_rings.to_csv(rings_path, index=False)
    df_payments.attrs["benign_groups"].to_csv(os.path.join(cfg.output_dir, "benign_groups.csv"), index=False)

    print(f"Simulator v2 CSVs successfully written to '{cfg.output_dir}/':")
    print(f"  1. {users_path} ({len(df_users):,} rows)")
    print(f"  2. {payments_path} ({len(df_payments):,} rows)")
    print(f"  3. {rings_path} ({len(df_rings):,} rows)")


if __name__ == "__main__":
    main()