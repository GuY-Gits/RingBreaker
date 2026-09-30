"""F13: Confirm-fraud loop.

Analyst verdict spreads risk to neighbours using Personalized PageRank.
Scores change within seconds of a confirmation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import networkx as nx


def propagate_risk(
    graph: nx.DiGraph | nx.MultiDiGraph,
    confirmed_fraud_nodes: Set[str] | List[str],
    alpha: float = 0.85,
) -> Dict[str, float]:
    """Personalized PageRank on directed graph from confirmed fraud accounts."""
    if not graph or graph.number_of_nodes() == 0:
        return {}

    seeds = set(str(n) for n in confirmed_fraud_nodes if graph.has_node(str(n)))
    if not seeds:
        return {str(n): 0.0 for n in graph.nodes()}

    personalization = {str(n): (1.0 if str(n) in seeds else 0.0) for n in graph.nodes()}
    # Run pagerank on DiGraph
    if isinstance(graph, nx.MultiDiGraph):
        digraph = nx.DiGraph(graph)
    else:
        digraph = graph

    scores = nx.pagerank(digraph, alpha=alpha, personalization=personalization)
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        norm_scores = {k: float(v / max_score) for k, v in scores.items()}
    else:
        norm_scores = {k: 0.0 for k in scores}

    return norm_scores


def propagate_from_confirmed(
    alert: Dict[str, Any],
    graph: Any,
    feature_store: Dict[str, Dict[str, Any]],
    alpha: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    F13 entrypoint called by POST /alerts/{id}/verdict when analyst confirms fraud.
    Updates feature_store risk_boost and returns before/after deltas.
    """
    # 1. Resolve seed nodes
    seeds: Set[str] = set()
    for m in alert.get("members", []):
        seeds.add(str(m))
    for ev in alert.get("timeline", []):
        if ev.get("sender"):
            seeds.add(str(ev["sender"]))
        if ev.get("receiver"):
            seeds.add(str(ev["receiver"]))
    trigger = alert.get("transaction") or {}
    if trigger.get("sender"):
        seeds.add(str(trigger["sender"]))
    if trigger.get("receiver"):
        seeds.add(str(trigger["receiver"]))

    # 2. Extract DiGraph
    if hasattr(graph, "as_of_digraph"):
        digraph = graph.as_of_digraph()
    elif isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        digraph = graph
    else:
        digraph = nx.DiGraph()

    # 3. Compute Personalized PageRank
    boosted_ranks = propagate_risk(digraph, seeds, alpha=alpha)

    # 4. Update feature store and build change record
    risk_changes: List[Dict[str, Any]] = []
    for acc, ppr in boosted_ranks.items():
        if ppr <= 0.01:
            continue
        cur_feat = feature_store.get(acc, {})
        before = float(cur_feat.get("risk_boost", 0.0))
        # Direct seeds get full boost, neighbors get decayed PPR boost
        if acc in seeds:
            delta = 0.50
        else:
            delta = float(ppr * 0.40)
        after = min(1.0, round(before + delta, 4))
        cur_feat["risk_boost"] = after
        feature_store[acc] = cur_feat
        risk_changes.append(
            {
                "account_id": acc,
                "risk_before": round(before, 4),
                "risk_after": round(after, 4),
            }
        )

    risk_changes.sort(key=lambda x: x["risk_after"], reverse=True)
    return risk_changes


def run_propagation(
    confirmed_tx_id: str,
    confirmed_at_str: Optional[str] = None,
    payments_path: Optional[Any] = None,
    output_path: Optional[Any] = None,
) -> Any:
    """Offline batch runner for analyst confirmation propagation."""
    from pathlib import Path
    import pandas as pd

    project_root = Path(__file__).resolve().parent.parent
    p_path = Path(payments_path) if payments_path else project_root / "data" / "payments.csv"
    out_path = Path(output_path) if output_path else project_root / "data" / "propagated_risk.csv"

    if not p_path.exists():
        raise FileNotFoundError(f"Missing {p_path}")

    df_p = pd.read_csv(p_path)
    df_p["timestamp"] = pd.to_datetime(df_p["timestamp"])
    df_p = df_p.sort_values("timestamp").reset_index(drop=True)

    match = df_p[df_p["transaction_id"] == confirmed_tx_id]
    if match.empty:
        raise ValueError(f"Transaction ID {confirmed_tx_id} not found")

    row = match.iloc[0]
    cutoff_time = pd.to_datetime(confirmed_at_str) if confirmed_at_str else row["timestamp"]

    # Strictly filter transactions up to cutoff_time (zero future leakage)
    df_hist = df_p[df_p["timestamp"] <= cutoff_time]

    G = nx.DiGraph()
    for _, tx in df_hist.iterrows():
        G.add_edge(str(tx["sender"]), str(tx["receiver"]), weight=float(tx["amount"]))

    seeds = [str(row["sender"]), str(row["receiver"])]
    scores = propagate_risk(G, seeds)

    rows = []
    for node, rank in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        if node in seeds:
            rows.append({
                "user_id": node,
                "propagated_risk": 1.0,
                "hop_distance": 0,
                "source_account": node,
                "propagation_reason": "Confirmed fraudulent entity (Seed)",
            })
        elif rank > 0.05:
            rows.append({
                "user_id": node,
                "propagated_risk": round(float(rank), 4),
                "hop_distance": 1,
                "source_account": seeds[0],
                "propagation_reason": "High personalized PageRank neighbor",
            })

    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return df_out