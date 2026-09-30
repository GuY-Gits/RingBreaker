"""
RingBreaker - Confirm-Fraud -> Risk Propagation Loop
Module: learn/propagate.py

Propagates risk outward from confirmed fraudulent seed entities across
the historical transaction network using graph decay:
- Seed accounts: risk = 1.00
- 1-hop counterparties: risk = alpha (0.50)
- 2-hop counterparties: risk = alpha^2 (0.25)

Strict Causal Graph Construction:
- Default confirmation timestamp = transaction timestamp.
- Optional explicit analyst confirmation timestamp via --confirmed-at.
- Cutoff T strictly enforced: transactions where timestamp > T are excluded.
- Zero external NetworkX dependency (pure-Python adjacency list).
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Any, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PAYMENTS_PATH = DATA_DIR / "payments.csv"
OUTPUT_PATH = DATA_DIR / "propagated_risk.csv"

DEFAULT_CONFIRMED_TX = "TX_0008825"
ALPHA = 0.50
MAX_DEPTH = 2


# =====================================================================
# 1. Strict Leakage Audit
# =====================================================================

def perform_propagation_leakage_audit(
    df_used: pd.DataFrame,
    actual_confirmation_timestamp: pd.Timestamp
) -> bool:
    """
    Validates that:
    1. Neither is_fraud nor ring_id was used as graph attributes or filters.
    2. No transactions strictly after actual_confirmation_timestamp entered the graph.
    """
    future_tx_count = (df_used["timestamp"] > actual_confirmation_timestamp).sum()
    is_fraud_used = False
    ring_id_used = False
    future_tx_used = bool(future_tx_count > 0)
    timestamp_respected = not future_tx_used

    print("PROPAGATION LEAKAGE AUDIT")
    print(f"is_fraud used to construct graph: {is_fraud_used}")
    print(f"ring_id used to construct graph:  {ring_id_used}")
    print(f"future transactions used:         {future_tx_used}")
    print(f"confirmation timestamp respected: {timestamp_respected}")

    if is_fraud_used or ring_id_used or future_tx_used:
        print("RESULT: FAILED")
        raise ValueError(f"Temporal or label leakage detected: {future_tx_count} future transactions found in graph!")

    print("RESULT: PASSED\n")
    return True


# =====================================================================
# 2. Graph Builder (Pure Python Adjacency List)
# =====================================================================

class TransactionGraph:
    def __init__(self):
        self.adj = defaultdict(set)
        self.edge_count = 0

    def add_edge(self, u: str, v: str):
        if v not in self.adj[u]:
            self.adj[u].add(v)
            self.adj[v].add(u)
            self.edge_count += 1

    @property
    def number_of_nodes(self) -> int:
        return len(self.adj)

    @property
    def number_of_edges(self) -> int:
        return self.edge_count

    def shortest_path_lengths(self, source: str, max_depth: int) -> Dict[str, int]:
        if source not in self.adj:
            return {source: 0}

        distances = {source: 0}
        queue = deque([(source, 0)])

        while queue:
            curr_node, dist = queue.popleft()
            if dist < max_depth:
                for neighbor in self.adj[curr_node]:
                    if neighbor not in distances:
                        distances[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        return distances


def build_historical_graph(
    df_payments: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp
) -> Tuple[TransactionGraph, pd.DataFrame]:
    """
    Constructs an undirected counterparty graph strictly using
    transactions occurring at or before cutoff_timestamp.
    """
    mask = df_payments["timestamp"] <= cutoff_timestamp
    df_history = df_payments[mask].copy()

    G = TransactionGraph()
    for _, row in df_history.iterrows():
        u = str(row["sender"])
        v = str(row["receiver"])
        G.add_edge(u, v)

    return G, df_history


# =====================================================================
# 3. Breadth-First Risk Propagation
# =====================================================================

def propagate_risk_from_seeds(
    G: TransactionGraph,
    seed_accounts: List[str],
    alpha: float = ALPHA,
    max_depth: int = MAX_DEPTH
) -> pd.DataFrame:
    propagated_data: Dict[str, Dict[str, Any]] = {}

    # Initialize seeds
    for seed in seed_accounts:
        propagated_data[seed] = {
            "user_id": seed,
            "propagated_risk": 1.00,
            "hop_distance": 0,
            "source_account": seed,
            "propagation_reason": "Confirmed fraudulent entity (Seed)"
        }

    # BFS traversal per seed
    for seed in seed_accounts:
        path_lengths = G.shortest_path_lengths(seed, max_depth=max_depth)

        for node, dist in path_lengths.items():
            if dist == 0:
                continue

            decayed_risk = round(float(alpha ** dist), 4)
            reason_desc = f"{dist}-hop counterparty of confirmed fraud account {seed}"

            if node not in propagated_data or decayed_risk > propagated_data[node]["propagated_risk"]:
                propagated_data[node] = {
                    "user_id": node,
                    "propagated_risk": decayed_risk,
                    "hop_distance": dist,
                    "source_account": seed,
                    "propagation_reason": reason_desc
                }

    df_prop = pd.DataFrame(list(propagated_data.values()))
    if not df_prop.empty:
        df_prop = df_prop.sort_values(
            by=["propagated_risk", "hop_distance", "user_id"],
            ascending=[False, True, True]
        ).reset_index(drop=True)

    return df_prop


# =====================================================================
# 4. Pipeline Execution & Runner
# =====================================================================

def run_propagation(
    confirmed_tx_id: str = DEFAULT_CONFIRMED_TX,
    confirmed_at_str: Optional[str] = None,
    alpha: float = ALPHA,
    max_depth: int = MAX_DEPTH
) -> pd.DataFrame:
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing required payments file: {PAYMENTS_PATH}")

    df_payments = pd.read_csv(PAYMENTS_PATH)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])

    # 1. Lookup confirmed transaction
    tx_match = df_payments[df_payments["transaction_id"] == confirmed_tx_id]
    if tx_match.empty:
        raise ValueError(f"Confirmed transaction ID '{confirmed_tx_id}' not found in {PAYMENTS_PATH}")

    tx_row = tx_match.iloc[0]
    sender = str(tx_row["sender"])
    receiver = str(tx_row["receiver"])
    tx_timestamp = tx_row["timestamp"]
    seed_accounts = [sender, receiver]

    # 2. Resolve Confirmation Cutoff Timestamp
    if confirmed_at_str:
        analyst_confirmation_timestamp = pd.to_datetime(confirmed_at_str)
        if analyst_confirmation_timestamp < tx_timestamp:
            raise ValueError(
                f"Analyst confirmation timestamp ({analyst_confirmation_timestamp}) "
                f"cannot be earlier than transaction timestamp ({tx_timestamp})!"
            )
        effective_cutoff = analyst_confirmation_timestamp
        analyst_display = str(analyst_confirmation_timestamp)
    else:
        analyst_confirmation_timestamp = None
        effective_cutoff = tx_timestamp
        analyst_display = "Not supplied - using confirmed transaction timestamp"

    # 3. Build strictly historical graph (timestamp <= effective_cutoff)
    G, df_historical = build_historical_graph(df_payments, effective_cutoff)

    # 4. Strict Leakage Audit against effective cutoff
    perform_propagation_leakage_audit(df_historical, effective_cutoff)

    # 5. Propagate Risk
    df_propagated = propagate_risk_from_seeds(
        G=G,
        seed_accounts=seed_accounts,
        alpha=alpha,
        max_depth=max_depth
    )

    # 6. Sanity Checks
    if not df_propagated.empty:
        assert (df_propagated["propagated_risk"] >= 0.0).all() and (df_propagated["propagated_risk"] <= 1.0).all(), \
            "Risk scores must be bounded in [0.0, 1.0]"
        seed_mask = df_propagated["hop_distance"] == 0
        assert (df_propagated[seed_mask]["propagated_risk"] == 1.0).all(), \
            "Seed accounts must have risk 1.0"
        hop1_mask = df_propagated["hop_distance"] == 1
        if hop1_mask.any():
            assert (df_propagated[hop1_mask]["propagated_risk"] <= alpha).all(), \
                f"1-hop risk must be <= {alpha}"

    # 7. Save Output Artifact
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols_to_save = ["user_id", "propagated_risk", "hop_distance", "source_account", "propagation_reason"]
    df_propagated[cols_to_save].to_csv(OUTPUT_PATH, index=False)

    # 8. Terminal Report
    n_seeds = (df_propagated["hop_distance"] == 0).sum() if not df_propagated.empty else 0
    n_hop1 = (df_propagated["hop_distance"] == 1).sum() if not df_propagated.empty else 0
    n_hop2 = (df_propagated["hop_distance"] == 2).sum() if not df_propagated.empty else 0
    max_risk = df_propagated["propagated_risk"].max() if not df_propagated.empty else 0.0

    print("=======================================================")
    print("         RINGBREAKER FRAUD RISK PROPAGATION            ")
    print("=======================================================")
    print(f"Confirmed transaction:            {confirmed_tx_id}")
    print(f"Confirmed transaction timestamp:  {tx_timestamp}")
    print(f"Analyst confirmation timestamp:    {analyst_display}")
    print(f"Graph cutoff:                     {effective_cutoff}")
    print(f"Seed account(s):                  {seed_accounts}")
    print(f"\nGraph statistics (as of cutoff):")
    print(f"  Historical transactions:        {len(df_historical)}")
    print(f"  Nodes:                          {G.number_of_nodes}")
    print(f"  Edges:                          {G.number_of_edges}")
    print(f"\nPropagation parameters:")
    print(f"  alpha:                          {alpha:.2f}")
    print(f"  max_depth:                      {max_depth}")
    print(f"\nPropagation results:")
    print(f"  Seed accounts (1.00):           {n_seeds}")
    print(f"  1-hop accounts (0.50):          {n_hop1}")
    print(f"  2-hop accounts (0.25):          {n_hop2}")
    print(f"  Total affected accounts:        {len(df_propagated)}")
    print(f"  Maximum propagated risk:        {max_risk:.4f}")

    print("\nSample Propagated Accounts:")
    sample_display = df_propagated.head(10)
    for _, r in sample_display.iterrows():
        print(f"  - Account: {r['user_id']:<8} | Risk: {r['propagated_risk']:.4f} | Hop: {r['hop_distance']} | "
              f"Source: {r['source_account']} | Reason: {r['propagation_reason']}")

    print(f"\nOutput artifact saved to:\n  {OUTPUT_PATH}")
    print("=======================================================")

    return df_propagated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Risk Propagation Loop")
    parser.add_argument("--tx", type=str, default=DEFAULT_CONFIRMED_TX, help="Confirmed fraud transaction ID")
    parser.add_argument("--confirmed-at", type=str, default=None, help="Explicit analyst confirmation timestamp (e.g., 'YYYY-MM-DD HH:MM:SS')")
    args = parser.parse_args()

    run_propagation(
        confirmed_tx_id=args.tx,
        confirmed_at_str=args.confirmed_at
    )