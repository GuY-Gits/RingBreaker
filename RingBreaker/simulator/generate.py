"""
RingBreaker Synthetic Payment Simulator
=======================================
Generates realistic P2P transaction data containing normal users, sleeper/mule accounts,
scam victims, shared identity fragments, and planted fraud rings (Loop, Fan-In, Chain, Star, Held-Out).
Outputs:
  - data/users.csv
  - data/payments.csv
  - data/rings.csv
"""

import os
import random
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
NUM_USERS = 2000
NUM_TRANSACTIONS = 10000
NUM_FRAUD_RINGS = 5  # Total rings: 4 training/validation rings + 1 held-out test ring
RANDOM_SEED = 42

SIMULATION_DAYS = 90
HELD_OUT_START_DAY = 76.5  # Last 15% window of the 90 days (70/15/15 time split)
BASE_START_DATE = datetime(2026, 1, 1, 0, 0, 0)

OUTPUT_DIR = "data"


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================
# 2. SYNTHETIC USER & IDENTITY GENERATION
# ==========================================
def generate_users(num_users=NUM_USERS, sim_days=SIMULATION_DAYS, start_date=BASE_START_DATE):
    """
    Generates synthetic accounts with realistic identity fragments.
    Initializes roles as 'normal', to be specialized later by fraud-ring generators.
    """
    users = []
    
    # Pre-generate shared resource pools for realistic identity clusters
    cities = ["Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]
    streets = ["MG Road", "Station Road", "Park Street", "Ring Road", "Nehru Nagar", "Indira Nagar", "Civil Lines"]

    for i in range(1, num_users + 1):
        u_id = f"U{i:05d}"
        
        # Signup timestamp spread over the first 45 days or earlier
        signup_offset_days = random.uniform(0, sim_days * 0.5)
        signup_ts = start_date + timedelta(days=signup_offset_days, seconds=random.randint(0, 86400))
        
        # Synthetic credentials
        phone = f"+9198{random.randint(10000000, 99999999)}"
        city = random.choice(cities)
        street = random.choice(streets)
        address = f"Flat {random.randint(101, 999)}, {street}, {city} - {random.randint(400001, 700001)}"
        device_id = f"DEV_{random.randint(100000, 999999)}"
        ip_address = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
        
        users.append({
            "user_id": u_id,
            "phone": phone,
            "address": address,
            "device_id": device_id,
            "ip_address": ip_address,
            "signup_timestamp": signup_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "user_role": "normal"
        })
        
    df_users = pd.DataFrame(users)
    return df_users


def apply_identity_sharing(df_users, fraud_user_ids, num_clusters=3):
    """
    Subtly colludes fraud ring members by sharing device_id, ip_address, or address
    across small clusters, generating realistic identity-fragment graphs.
    """
    cluster_size = max(2, len(fraud_user_ids) // num_clusters)
    for i in range(num_clusters):
        subset = fraud_user_ids[i * cluster_size : (i + 1) * cluster_size]
        if len(subset) < 2:
            continue
            
        shared_dev = f"DEV_SHARED_{i+1:03d}"
        shared_ip = f"10.0.{i+1}.{random.randint(2, 250)}"
        shared_addr = f"Flat {400 + i}, Trade Center, Ring Road, Mumbai - 400051"
        
        for uid in subset:
            idx = df_users.index[df_users["user_id"] == uid].tolist()[0]
            # Probabilistic partial sharing (never 100% full match to stay realistic)
            if random.random() < 0.75:
                df_users.at[idx, "device_id"] = shared_dev
            if random.random() < 0.70:
                df_users.at[idx, "ip_address"] = shared_ip
            if random.random() < 0.50:
                df_users.at[idx, "address"] = shared_addr


# ==========================================
# 3. NORMAL TRANSACTION GENERATION
# ==========================================
def generate_normal_transactions(df_users, target_count, start_date=BASE_START_DATE, sim_days=SIMULATION_DAYS):
    """
    Generates realistic, daily P2P payments:
    - Log-normal amounts (INR 50 to 5,000)
    - Frequent counterparties (power-law distribution)
    - Realistic diurnal activity peaks (afternoon/evening)
    """
    payments = []
    normal_users = df_users[df_users["user_role"] == "normal"]["user_id"].values
    num_norm = len(normal_users)
    
    # Construct persistent counterparty affinities (friends, regular merchants, family)
    affinities = {}
    for uid in normal_users:
        partner_count = random.randint(1, 5)
        partners = np.random.choice(normal_users, size=partner_count, replace=False)
        affinities[uid] = [p for p in partners if p != uid]

    user_meta = df_users.set_index("user_id").to_dict("index")
    tx_counter = 1

    while len(payments) < target_count:
        sender = np.random.choice(normal_users)
        
        # 65% chance of transacting with a known affinity; otherwise random normal user
        if affinities.get(sender) and random.random() < 0.65:
            receiver = random.choice(affinities[sender])
        else:
            receiver = np.random.choice(normal_users)
            while receiver == sender:
                receiver = np.random.choice(normal_users)

        sender_meta = user_meta[sender]
        signup_dt = datetime.strptime(sender_meta["signup_timestamp"], "%Y-%m-%d %H:%M:%S")
        
        # Choose a timestamp strictly after sender's signup
        min_seconds = max(0, int((signup_dt - start_date).total_seconds()))
        max_seconds = int(sim_days * 86400)
        if min_seconds >= max_seconds - 3600:
            continue
            
        random_sec = random.randint(min_seconds, max_seconds)
        tx_dt = start_date + timedelta(seconds=random_sec)
        
        # Diurnal distribution adjustment (lower traffic late night 01:00-06:00)
        hour = tx_dt.hour
        if 1 <= hour <= 6 and random.random() > 0.15:
            # Shift transaction to daytime
            tx_dt = tx_dt.replace(hour=random.randint(9, 21))

        # Realistic retail amount: lognormal centered around ₹450-₹1,200
        amount = float(np.round(np.clip(np.random.lognormal(mean=6.5, sigma=1.0), 30, 25000), 2))
        
        payments.append({
            "transaction_id": f"TX_{tx_counter:07d}",
            "timestamp": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "device_id": sender_meta["device_id"],
            "ip_address": sender_meta["ip_address"],
            "is_fraud": 0,
            "ring_id": None
        })
        tx_counter += 1

    return payments


# ==========================================
# 4. PLANTED FRAUD RINGS GENERATION
# ==========================================
def generate_loop_ring(ring_id, members, df_users, base_time, tx_counter_start):
    """
    RING TYPE 1: Circular Loop (A -> B -> C -> D -> A)
    Rapid cycling of funds with slight deductions (mule cut/fees), preserving money balance.
    """
    payments = []
    current_time = base_time
    base_amount = random.uniform(4200, 6800)
    tx_id = tx_counter_start
    user_meta = df_users.set_index("user_id").to_dict("index")

    # Complete the circle: A->B, B->C, C->D, D->A
    cycle = members + [members[0]]
    for i in range(len(cycle) - 1):
        u_from = cycle[i]
        u_to = cycle[i + 1]
        
        # Short dwell time (5 to 35 minutes)
        current_time += timedelta(minutes=random.randint(5, 35), seconds=random.randint(0, 59))
        
        # Subtle variance in amount to mimic transaction slippage / cash out fees
        amt = round(base_amount * random.uniform(0.95, 0.99), 2)
        base_amount = amt
        
        payments.append({
            "transaction_id": f"TX_{tx_id:07d}",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": u_from,
            "receiver": u_to,
            "amount": amt,
            "device_id": user_meta[u_from]["device_id"],
            "ip_address": user_meta[u_from]["ip_address"],
            "is_fraud": 1,
            "ring_id": ring_id
        })
        tx_id += 1

    return payments, tx_id


def generate_fan_in_ring(ring_id, members, mule, df_users, base_time, tx_counter_start):
    """
    RING TYPE 2: Fan-In / Collector
    Multiple feeders (compromised or sleeper accounts) funnel funds to one central mule,
    followed by the mule cashing out or moving funds onward.
    """
    payments = []
    user_meta = df_users.set_index("user_id").to_dict("index")
    tx_id = tx_counter_start
    mule_total = 0.0
    latest_time = base_time

    # Incoming burst to the mule
    for sender in members:
        tx_time = base_time + timedelta(minutes=random.randint(5, 120), seconds=random.randint(0, 59))
        if tx_time > latest_time:
            latest_time = tx_time
            
        amt = float(round(random.uniform(2800, 4900), 2))
        mule_total += amt

        payments.append({
            "transaction_id": f"TX_{tx_id:07d}",
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sender,
            "receiver": mule,
            "amount": amt,
            "device_id": user_meta[sender]["device_id"],
            "ip_address": user_meta[sender]["ip_address"],
            "is_fraud": 1,
            "ring_id": ring_id
        })
        tx_id += 1

    # Mule forwards the consolidated amount onward shortly after collection
    exit_node = members[0]  # or an off-ramp ring leader
    drain_time = latest_time + timedelta(minutes=random.randint(20, 60))
    payments.append({
        "transaction_id": f"TX_{tx_id:07d}",
        "timestamp": drain_time.strftime("%Y-%m-%d %H:%M:%S"),
        "sender": mule,
        "receiver": exit_node,
        "amount": round(mule_total * 0.96, 2),
        "device_id": user_meta[mule]["device_id"],
        "ip_address": user_meta[mule]["ip_address"],
        "is_fraud": 1,
        "ring_id": ring_id
    })
    tx_id += 1

    return payments, tx_id


def generate_chain_ring(ring_id, members, df_users, base_time, tx_counter_start):
    """
    RING TYPE 3: Chain / Layering Pass-Through (A -> B -> C -> D -> E)
    Funds are passed linearly through intermediaries with minimal dwell time.
    """
    payments = []
    user_meta = df_users.set_index("user_id").to_dict("index")
    tx_id = tx_counter_start
    current_time = base_time
    amount = random.uniform(5000, 8500)

    for i in range(len(members) - 1):
        u_from = members[i]
        u_to = members[i + 1]
        
        # Tight dwell window (10 to 45 mins)
        current_time += timedelta(minutes=random.randint(10, 45), seconds=random.randint(0, 59))
        amount = round(amount * random.uniform(0.96, 0.99), 2)
        
        payments.append({
            "transaction_id": f"TX_{tx_id:07d}",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": u_from,
            "receiver": u_to,
            "amount": amount,
            "device_id": user_meta[u_from]["device_id"],
            "ip_address": user_meta[u_from]["ip_address"],
            "is_fraud": 1,
            "ring_id": ring_id
        })
        tx_id += 1

    return payments, tx_id


def generate_star_ring(ring_id, hub_user, spoke_users, df_users, base_time, tx_counter_start, direction="inbound"):
    """
    RING TYPE 4: Star Network
    Central hub interacting with multiple satellites (either inbound aggregation or outbound distribution).
    """
    payments = []
    user_meta = df_users.set_index("user_id").to_dict("index")
    tx_id = tx_counter_start

    for spoke in spoke_users:
        tx_time = base_time + timedelta(minutes=random.randint(5, 180), seconds=random.randint(0, 59))
        amt = float(round(random.uniform(1800, 4200), 2))
        
        sender = spoke if direction == "inbound" else hub_user
        receiver = hub_user if direction == "inbound" else spoke

        payments.append({
            "transaction_id": f"TX_{tx_id:07d}",
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sender,
            "receiver": receiver,
            "amount": amt,
            "device_id": user_meta[sender]["device_id"],
            "ip_address": user_meta[sender]["ip_address"],
            "is_fraud": 1,
            "ring_id": ring_id
        })
        tx_id += 1

    return payments, tx_id


def generate_scam_victim_tx(victim_id, mule_id, ring_id, df_users, tx_time, tx_counter):
    """
    Simulates a social engineering/phishing victim sending an unusual, out-of-pattern
    payment to a ring mule. The victim is NOT marked as is_fraud=1 on their account,
    though this specific transaction is part of the fraudulent ring extraction.
    """
    user_meta = df_users.set_index("user_id").to_dict("index")
    amt = float(round(random.uniform(9000, 18000), 2))
    
    return {
        "transaction_id": f"TX_{tx_counter:07d}",
        "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
        "sender": victim_id,
        "receiver": mule_id,
        "amount": amt,
        "device_id": user_meta[victim_id]["device_id"],
        "ip_address": user_meta[victim_id]["ip_address"],
        "is_fraud": 1,
        "ring_id": ring_id
    }


# ==========================================
# 5. ORCHESTRATION PIPELINE
# ==========================================
def generate_all_rings(df_users, start_date=BASE_START_DATE, tx_counter_start=1):
    """
    Allocates accounts to specific fraud topologies, handles sleeper accounts,
    victim profiles, and creates RING_HELDOUT for the final test partition.
    """
    all_ring_payments = []
    rings_meta = []
    tx_id = tx_counter_start
    all_fraud_user_ids = []

    # Select candidate pool from users
    available_users = list(df_users["user_id"].values)
    random.shuffle(available_users)

    # Helper to reserve unique users
    def reserve(n):
        return [available_users.pop() for _ in range(n)]

    # --- RING 001: Circular Loop (Days 15 - 18) ---
    r1_members = reserve(4)
    all_fraud_user_ids.extend(r1_members)
    for u in r1_members:
        df_users.loc[df_users["user_id"] == u, "user_role"] = "ring_member"
    r1_start = start_date + timedelta(days=16, hours=10)
    p1, tx_id = generate_loop_ring("RING_001", r1_members, df_users, r1_start, tx_id)
    all_ring_payments.extend(p1)
    rings_meta.append({
        "ring_id": "RING_001",
        "ring_type": "circular_loop",
        "members": ";".join(r1_members),
        "start_time": min(p["timestamp"] for p in p1),
        "end_time": max(p["timestamp"] for p in p1)
    })

    # --- RING 002: Fan-In / Collector with Sleeper Mule (Days 28 - 32) ---
    r2_feeders = reserve(5)
    r2_mule = reserve(1)[0]
    all_fraud_user_ids.extend(r2_feeders + [r2_mule])
    for u in r2_feeders:
        df_users.loc[df_users["user_id"] == u, "user_role"] = "ring_member"
    df_users.loc[df_users["user_id"] == r2_mule, "user_role"] = "mule"
    
    # Configure sleeper profile for mule: signup early, zero activity for 25 days
    df_users.loc[df_users["user_id"] == r2_mule, "signup_timestamp"] = (
        start_date + timedelta(days=2)
    ).strftime("%Y-%m-%d %H:%M:%S")

    r2_start = start_date + timedelta(days=30, hours=14)
    p2, tx_id = generate_fan_in_ring("RING_002", r2_feeders, r2_mule, df_users, r2_start, tx_id)
    
    # Inject a victim who pays the mule
    victim_1 = reserve(1)[0]
    df_users.loc[df_users["user_id"] == victim_1, "user_role"] = "victim"
    v_tx = generate_scam_victim_tx(victim_1, r2_mule, "RING_002", df_users, r2_start - timedelta(hours=2), tx_id)
    tx_id += 1
    p2.append(v_tx)

    all_ring_payments.extend(p2)
    rings_meta.append({
        "ring_id": "RING_002",
        "ring_type": "fan_in_collector",
        "members": ";".join(r2_feeders + [r2_mule, victim_1]),
        "start_time": min(p["timestamp"] for p in p2),
        "end_time": max(p["timestamp"] for p in p2)
    })

    # --- RING 003: Sequential Chain (Days 45 - 47) ---
    r3_members = reserve(5)
    all_fraud_user_ids.extend(r3_members)
    for u in r3_members:
        df_users.loc[df_users["user_id"] == u, "user_role"] = "ring_member"
    r3_start = start_date + timedelta(days=46, hours=19)
    p3, tx_id = generate_chain_ring("RING_003", r3_members, df_users, r3_start, tx_id)
    all_ring_payments.extend(p3)
    rings_meta.append({
        "ring_id": "RING_003",
        "ring_type": "linear_chain",
        "members": ";".join(r3_members),
        "start_time": min(p["timestamp"] for p in p3),
        "end_time": max(p["timestamp"] for p in p3)
    })

    # --- RING 004: Inbound Star Hub (Days 58 - 62) ---
    r4_hub = reserve(1)[0]
    r4_spokes = reserve(6)
    all_fraud_user_ids.extend(r4_spokes + [r4_hub])
    df_users.loc[df_users["user_id"] == r4_hub, "user_role"] = "mule"
    for u in r4_spokes:
        df_users.loc[df_users["user_id"] == u, "user_role"] = "ring_member"
    r4_start = start_date + timedelta(days=60, hours=11)
    p4, tx_id = generate_star_ring("RING_004", r4_hub, r4_spokes, df_users, r4_start, tx_id, direction="inbound")
    all_ring_payments.extend(p4)
    rings_meta.append({
        "ring_id": "RING_004",
        "ring_type": "star_hub",
        "members": ";".join([r4_hub] + r4_spokes),
        "start_time": min(p["timestamp"] for p in p4),
        "end_time": max(p["timestamp"] for p in p4)
    })

    # --- RING 005 (RING_HELDOUT): Held-Out Test Window (Day 80+) ---
    # Strictly occurs within the last 15% window (Day 76.5 - Day 90)
    r5_members = reserve(5)
    all_fraud_user_ids.extend(r5_members)
    for u in r5_members:
        df_users.loc[df_users["user_id"] == u, "user_role"] = "ring_member"
    r5_start = start_date + timedelta(days=82, hours=15)
    p5, tx_id = generate_loop_ring("RING_HELDOUT", r5_members, df_users, r5_start, tx_id)
    all_ring_payments.extend(p5)
    rings_meta.append({
        "ring_id": "RING_HELDOUT",
        "ring_type": "heldout_circular_loop",
        "members": ";".join(r5_members),
        "start_time": min(p["timestamp"] for p in p5),
        "end_time": max(p["timestamp"] for p in p5)
    })

    # Apply identity-fragment collisions across fraud participants
    apply_identity_sharing(df_users, all_fraud_user_ids, num_clusters=4)

    return all_ring_payments, rings_meta, tx_id


# ==========================================
# 6. VALIDATION AND INTEGRITY CHECKS
# ==========================================
def validate_data(df_users, df_payments, df_rings, heldout_start_time):
    """
    Runs automated integrity verification against the PRD rules.
    """
    print("\n" + "=" * 50)
    print("RUNNING RINGBREAKER SIMULATOR QUALITY CHECKS")
    print("=" * 50)

    # 1. Null Checks
    assert df_users.isnull().sum().sum() == 0, "Error: Missing values found in users.csv"
    assert df_payments[["transaction_id", "timestamp", "sender", "receiver", "amount", "is_fraud"]].isnull().sum().sum() == 0, \
        "Error: Missing required fields in payments.csv"

    # 2. Graph Self-Loop Check
    self_tx = df_payments[df_payments["sender"] == df_payments["receiver"]]
    assert len(self_tx) == 0, "Error: Detected transactions where sender == receiver"

    # 3. Label Consistency
    fraud_without_ring = df_payments[(df_payments["is_fraud"] == 1) & (df_payments["ring_id"].isnull())]
    normal_with_ring = df_payments[(df_payments["is_fraud"] == 0) & (df_payments["ring_id"].notnull())]
    assert len(fraud_without_ring) == 0, "Error: Fraudulent transactions missing ring_id"
    assert len(normal_with_ring) == 0, "Error: Normal transactions must have ring_id = None"

    # 4. User Existence
    all_users = set(df_users["user_id"])
    all_senders = set(df_payments["sender"])
    all_receivers = set(df_payments["receiver"])
    assert all_senders.issubset(all_users), "Error: Unregistered sender found in payments"
    assert all_receivers.issubset(all_users), "Error: Unregistered receiver found in payments"

    # 5. Chronological Order
    tx_times = pd.to_datetime(df_payments["timestamp"])
    assert tx_times.is_monotonic_increasing, "Error: payments.csv is not chronologically sorted"

    # 6. Held-Out Ring Temporal Isolation
    heldout_txs = df_payments[df_payments["ring_id"] == "RING_HELDOUT"]
    earliest_heldout = pd.to_datetime(heldout_txs["timestamp"]).min()
    assert earliest_heldout >= heldout_start_time, (
        f"Error: RING_HELDOUT leaked into training partition! "
        f"Earliest: {earliest_heldout}, Limit: {heldout_start_time}"
    )

    # 7. Summary Metrics
    total_tx = len(df_payments)
    fraud_tx = int(df_payments["is_fraud"].sum())
    norm_tx = total_tx - fraud_tx
    fraud_pct = (fraud_tx / total_tx) * 100

    print(f"Total Users:                 {len(df_users):,}")
    print(f"Total Transactions:          {total_tx:,}")
    print(f"Normal Transactions:         {norm_tx:,}")
    print(f"Fraudulent Transactions:     {fraud_tx:,}")
    print(f"Fraud Transaction Ratio:     {fraud_pct:.2f}%")
    print(f"Total Planted Rings:         {len(df_rings)}")
    print(f"Dataset Date Range:          {df_payments['timestamp'].min()} to {df_payments['timestamp'].max()}")
    print("\nTransactions per Ring:")
    for rid, count in df_payments[df_payments["ring_id"].notnull()]["ring_id"].value_counts().items():
        print(f"  - {rid:15s}: {count:3d} transactions")
    print("=" * 50)
    print("ALL INTEGRITY CHECKS PASSED SUCCESSFULLY!\n")


# ==========================================
# 7. MAIN ENTRYPOINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="RingBreaker Synthetic Payment Generator")
    parser.add_argument("--users", type=int, default=NUM_USERS, help="Total number of users to generate")
    parser.add_argument("--transactions", type=int, default=NUM_TRANSACTIONS, help="Total normal transactions")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Destination directory for CSVs")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating synthetic P2P dataset ({args.users} users, ~{args.transactions} txs)...")

    # Step 1: Base Users
    df_users = generate_users(num_users=args.users, sim_days=SIMULATION_DAYS, start_date=BASE_START_DATE)

    # Step 2: Planted Fraud Rings (Loop, Fan-In, Chain, Star, Held-Out)
    fraud_payments, rings_meta, next_tx_id = generate_all_rings(df_users, start_date=BASE_START_DATE, tx_counter_start=1)

    # Step 3: Normal Payments
    normal_payments = generate_normal_transactions(
        df_users, target_count=args.transactions, start_date=BASE_START_DATE, sim_days=SIMULATION_DAYS
    )

    # Step 4: Merge, Sort, and Re-index IDs to preserve chronological ordering
    all_payments = fraud_payments + normal_payments
    df_payments = pd.DataFrame(all_payments)
    df_payments["dt"] = pd.to_datetime(df_payments["timestamp"])
    df_payments.sort_values(by="dt", inplace=True)
    df_payments.drop(columns=["dt"], inplace=True)
    
    # Re-number transaction IDs chronologically
    df_payments["transaction_id"] = [f"TX_{i+1:07d}" for i in range(len(df_payments))]
    df_rings = pd.DataFrame(rings_meta)

    # Step 5: Run Automated Integrity Verification
    heldout_threshold = BASE_START_DATE + timedelta(days=HELD_OUT_START_DAY)
    validate_data(df_users, df_payments, df_rings, heldout_threshold)

    # Step 6: Save Clean CSV Outputs
    users_path = os.path.join(args.output_dir, "users.csv")
    payments_path = os.path.join(args.output_dir, "payments.csv")
    rings_path = os.path.join(args.output_dir, "rings.csv")

    df_users.to_csv(users_path, index=False)
    df_payments.to_csv(payments_path, index=False)
    df_rings.to_csv(rings_path, index=False)

    print(f"CSVs successfully written to '{args.output_dir}/':")
    print(f"  1. {users_path}")
    print(f"  2. {payments_path}")
    print(f"  3. {rings_path}")


if __name__ == "__main__":
    main()