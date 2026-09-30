"""RingBreaker API — F9 fast path, F14 dashboard backend.

Endpoints (PRD API contract):
  POST /score
  GET  /alerts
  GET  /alerts/{id}
  POST /alerts/{id}/verdict
  GET  /graph/snapshot
  POST /admin/retrain

Zero temporal leakage:
Payments are scored as of T using TransactionGraph and IdentityGraph BEFORE
the candidate payment is inserted into the graph.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ringbreaker.explain.case_file import PaymentEvent, assemble_case_file
from ringbreaker.explain.counterfactual import compute_counterfactual
from ringbreaker.graphs.build import IdentityGraph, Payment, TransactionGraph, parse_timestamp
from ringbreaker.learn.propagate import propagate_from_confirmed
from ringbreaker.learn.retrain import retrain
from ringbreaker.patterns import detect_all_patterns, detect_for_payment
from ringbreaker.scoring.action import decide_action, determine_action, format_canonical_risk, ALLOW_THRESHOLD, BLOCK_THRESHOLD
from ringbreaker.scoring.pair_model import score_payment
from ringbreaker.scoring.risk_engine import RiskEngine

logger = logging.getLogger("ringbreaker.api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAYMENTS_CSV = DATA_DIR / "payments.csv"
USERS_CSV = DATA_DIR / "users.csv"

# Global Graph & State Engine
_transaction_graph: TransactionGraph = TransactionGraph()
_identity_graph: IdentityGraph = IdentityGraph()
_account_feature_cache: Dict[str, Dict[str, Any]] = {}
_alerts: Dict[str, Dict[str, Any]] = {}
_payments_log: List[PaymentEvent] = []
_verified_labels: Dict[str, Literal["confirm", "clear"]] = {}
_model_version: str = "1.0.0-xgb"
_risk_engine: RiskEngine = RiskEngine()
_init_lock = Lock()
_initialized = False


def _account_features(account_id: str) -> Dict[str, Any]:
    acc = str(account_id)
    if acc in _account_feature_cache:
        return _account_feature_cache[acc]
    base = {
        "account_id": acc,
        "risk_boost": 0.0,
        "in_degree": len(_transaction_graph.get_in_neighbors(acc)),
        "out_degree": len(_transaction_graph.get_out_neighbors(acc)),
    }
    _account_feature_cache[acc] = base
    return base


def load_state(max_initial_tx: int = 1500) -> None:
    """Warm graph and feature cache with historical base data if available."""
    global _initialized, _model_version
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        logger.info("RingBreaker API — warming in-memory graph and feature caches")
        if USERS_CSV.exists():
            try:
                df_u = pd.read_csv(USERS_CSV)
                for _, u in df_u.iterrows():
                    u_id = str(u["user_id"])
                    _transaction_graph.register_account(u_id, signup_at=u.get("signup_timestamp"))
                    if pd.notnull(u.get("device_id")):
                        _identity_graph.add_identity_link(u_id, "device", str(u["device_id"]))
                    if pd.notnull(u.get("ip_address")):
                        _identity_graph.add_identity_link(u_id, "ip", str(u["ip_address"]))
                    if pd.notnull(u.get("phone")):
                        _identity_graph.add_identity_link(u_id, "phone", str(u["phone"]))
                    _account_feature_cache[u_id] = {
                        "account_id": u_id,
                        "risk_boost": 0.0,
                    }
                logger.info("Loaded %d registered users into graph", len(df_u))
            except Exception as e:
                logger.warning("Could not load users.csv: %s", e)

        if PAYMENTS_CSV.exists():
            try:
                df_p = pd.read_csv(PAYMENTS_CSV)
                df_p["timestamp"] = pd.to_datetime(df_p["timestamp"])
                df_p = df_p.sort_values("timestamp").reset_index(drop=True)

                # Preload initial training transactions into graph
                preload_count = min(len(df_p), max_initial_tx)
                for i in range(preload_count):
                    row = df_p.iloc[i]
                    s, r = str(row["sender"]), str(row["receiver"])
                    amt = float(row["amount"])
                    ts = row["timestamp"].to_pydatetime()
                    tx_id = str(row.get("transaction_id", f"TX_{i:07d}"))
                    dev = str(row["device_id"]) if pd.notnull(row.get("device_id")) else None

                    _transaction_graph.add_payment(s, r, amt, ts, transaction_id=tx_id)
                    if dev:
                        _identity_graph.add_identity_link(s, "device", dev, observed_at=ts)
                logger.info("Preloaded %d historical transactions into graph", preload_count)
            except Exception as e:
                logger.warning("Could not preload payments.csv: %s", e)

        # Preload anomaly scores
        anomaly_csv = DATA_DIR / "anomaly_scores.csv"
        if anomaly_csv.exists():
            try:
                df_anom = pd.read_csv(anomaly_csv)
                for _, r in df_anom.iterrows():
                    s, rec = str(r["sender"]), str(r["receiver"])
                    s_score = float(r.get("sender_anomaly_score", 0.0))
                    r_score = float(r.get("receiver_anomaly_score", 0.0))
                    if s not in _account_feature_cache:
                        _account_feature_cache[s] = {"account_id": s, "risk_boost": 0.0}
                    if rec not in _account_feature_cache:
                        _account_feature_cache[rec] = {"account_id": rec, "risk_boost": 0.0}
                    _account_feature_cache[s]["anomaly_score"] = s_score
                    _account_feature_cache[rec]["anomaly_score"] = r_score
            except Exception as e:
                logger.warning("Could not preload anomaly_scores.csv: %s", e)

        # Preload lockstep coordination scores
        lockstep_csv = DATA_DIR / "lockstep_scores.csv"
        if lockstep_csv.exists():
            try:
                df_lock = pd.read_csv(lockstep_csv)
                for _, r in df_lock.iterrows():
                    u_id = str(r["user_id"])
                    c_score = float(r.get("coordination_score", 0.0))
                    if u_id not in _account_feature_cache:
                        _account_feature_cache[u_id] = {"account_id": u_id, "risk_boost": 0.0}
                    _account_feature_cache[u_id]["coordination_score"] = c_score
            except Exception as e:
                logger.warning("Could not preload lockstep_scores.csv: %s", e)

        # Preload existing alerts into alert queue
        alerts_json = DATA_DIR / "alerts.json"
        case_dir = DATA_DIR / "case_files"
        if alerts_json.exists():
            try:
                import json
                with open(alerts_json, "r", encoding="utf-8") as f:
                    pre_alerts = json.load(f)
                for item in pre_alerts:
                    al_id = item["alert_id"]
                    cf_path = case_dir / f"{al_id}.json"
                    c_data = {}
                    if cf_path.exists():
                        try:
                            with open(cf_path, "r", encoding="utf-8") as cff:
                                c_data = json.load(cff)
                        except Exception:
                            pass
                    c_risk = float(item.get("overall_risk", c_data.get("overall_risk", 0.85)))

                    # Ensure standard CaseFile fields (F10, F12, F16)
                    sub_scores = c_data.get("sub_scores")
                    if not sub_scores:
                        r_dict = c_data.get("risk", {})
                        sub_scores = {
                            "sender_anomaly": round(float(r_dict.get("sender_anomaly_score", r_dict.get("behavioural_anomaly", 0.7))), 4),
                            "receiver_mule_propensity": round(float(r_dict.get("receiver_anomaly_score", 0.8)), 4),
                            "relationship_plausibility": round(float(r_dict.get("pair_risk", 0.9)), 4),
                        }
                    c_action = item.get("action") or c_data.get("action") or determine_action(c_risk, sub_scores)

                    members = c_data.get("members")
                    if not members:
                        members = c_data.get("ring_context", {}).get("members", [])
                    if not members and "transaction" in c_data:
                        tx_obj = c_data["transaction"]
                        s, r = tx_obj.get("sender"), tx_obj.get("receiver")
                        members = [s, r] if s and r else []

                    top_factors = c_data.get("top_factors")
                    if not top_factors:
                        inc_factors = c_data.get("evidence", {}).get("risk_increasing_factors", [])
                        top_factors = [
                            {
                                "name": str(f.get("feature", "factor")),
                                "value": float(f.get("feature_value", 0.0)),
                                "contribution": round(float(f.get("shap_value", 0.0)), 4),
                            }
                            for f in inc_factors
                        ]

                    timeline = c_data.get("timeline")
                    if not timeline and "transaction" in c_data:
                        tx_obj = c_data["transaction"]
                        timeline = [
                            {
                                "timestamp": str(tx_obj.get("timestamp", c_data.get("timestamp", ""))),
                                "sender": str(tx_obj.get("sender", "")),
                                "receiver": str(tx_obj.get("receiver", "")),
                                "amount": float(tx_obj.get("amount", 0.0)),
                                "device": tx_obj.get("device_id") or tx_obj.get("device"),
                            }
                        ]

                    counterfactual = c_data.get("counterfactual")
                    if not counterfactual:
                        try:
                            counterfactual = compute_counterfactual(sub_scores, c_risk, c_action)
                        except Exception:
                            counterfactual = None

                    summary = c_data.get("summary")
                    if not summary:
                        ring_info = c_data.get("ring_context", {})
                        if ring_info.get("associated_ring_id"):
                            summary = f"Flagged for {c_action} with {round(c_risk * 100.0, 1)}% risk. Associated with syndicate {ring_info.get('associated_ring_id')} ({ring_info.get('ring_type', 'ring')}). {ring_info.get('description', '')}"
                        else:
                            summary = f"Alert {al_id} flagged for {c_action} with canonical risk score {round(c_risk * 100.0, 1)}%."

                    _alerts[al_id] = {
                        **c_data,
                        "alert_id": al_id,
                        "risk_score": c_risk,
                        "overall_risk": c_risk,
                        "risk_percent": round(c_risk * 100.0, 2),
                        "action": c_action,
                        "sub_scores": sub_scores,
                        "members": members,
                        "top_factors": top_factors,
                        "timeline": timeline,
                        "counterfactual": counterfactual,
                        "summary": summary,
                        "pattern": item.get("associated_ring", c_data.get("pattern", "anomalous ring cohort")),
                        "status": "open",
                        "created_at": item.get("timestamp", ""),
                        "trigger_payment_id": item.get("transaction_id", ""),
                        "transaction": {
                            "sender": item.get("sender", c_data.get("transaction", {}).get("sender", "")),
                            "receiver": item.get("receiver", c_data.get("transaction", {}).get("receiver", "")),
                            "amount": float(item.get("amount", c_data.get("transaction", {}).get("amount", 0.0))),
                            "timestamp": item.get("timestamp", c_data.get("transaction", {}).get("timestamp", "")),
                        },
                    }
                logger.info("Preloaded %d open alerts into queue", len(_alerts))
            except Exception as e:
                logger.warning("Could not preload alerts.json: %s", e)

        _initialized = True


def get_transaction_graph() -> TransactionGraph:
    load_state()
    return _transaction_graph


def get_identity_graph() -> IdentityGraph:
    load_state()
    return _identity_graph


def reset_state() -> None:
    """Helper for testing to reset graph and alert states cleanly."""
    global _transaction_graph, _identity_graph, _account_feature_cache, _alerts, _payments_log, _verified_labels, _initialized
    with _init_lock:
        _transaction_graph = TransactionGraph()
        _identity_graph = IdentityGraph()
        _account_feature_cache.clear()
        _alerts.clear()
        _payments_log.clear()
        _verified_labels.clear()
        _initialized = True



# ── Request / response models ───────────────────────────────────────────────


class ScoreRequest(BaseModel):
    sender: str
    receiver: str
    amount: float = Field(gt=0)
    device: Optional[str] = None
    timestamp: str


class ScoreResponse(BaseModel):
    risk_score: float
    overall_risk: float
    risk_percent: float
    pair_risk: float = 0.0
    behavioural_anomaly: float = 0.0
    coordination_score: float = 0.0
    sender_anomaly: float
    receiver_mule_propensity: float
    relationship_plausibility: float
    action: str
    alert_id: Optional[str] = None


class AlertSummary(BaseModel):
    alert_id: str
    risk_score: float
    overall_risk: float
    risk_percent: float
    action: str
    pattern: Optional[str] = None
    created_at: str
    status: str


class VerdictRequest(BaseModel):
    verdict: Literal["confirm", "clear"]


class RiskChange(BaseModel):
    account_id: str
    risk_before: float
    risk_after: float


class VerdictResponse(BaseModel):
    alert_id: str
    verdict: str
    risk_changes: List[RiskChange]


class RetrainResponse(BaseModel):
    model_version: str
    metrics: Dict[str, Any]


def _should_alert(action: str, risk_score: float) -> bool:
    act = str(action).upper()
    return act in ("REVIEW", "BLOCK", "WARN_SENDER", "HOLD_RECEIVER") or float(risk_score) >= ALLOW_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="RingBreaker", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/score", response_model=ScoreResponse)
def score_payment_endpoint(req: ScoreRequest) -> ScoreResponse:
    t0 = time.perf_counter()
    load_state()

    # 1. SCORE PAYMENT FIRST (Zero temporal leakage: candidate tx is NOT in graph yet)
    score_out = score_payment(
        sender=req.sender,
        receiver=req.receiver,
        amount=req.amount,
        timestamp=req.timestamp,
        device=req.device,
        graph=_transaction_graph,
        identity_graph=_identity_graph,
        feature_cache=_account_feature_cache,
    )

    pair_risk = float(score_out["risk_score"])
    sub_scores = score_out["sub_scores"]
    shap_values = score_out.get("shap_values", {})

    # Retrieve anomaly and coordination signals
    s_feat = _account_features(req.sender)
    r_feat = _account_features(req.receiver)
    anom_score = float(max(
        s_feat.get("anomaly_score", sub_scores.get("sender_anomaly", 0.0)),
        r_feat.get("anomaly_score", sub_scores.get("receiver_mule_propensity", 0.0))
    ))
    coord_score = float(max(
        s_feat.get("coordination_score", 0.0),
        r_feat.get("coordination_score", 0.0)
    ))

    # Single canonical source of truth: 3-Signal Risk Engine
    engine_result = _risk_engine.score(
        payment={"transaction_id": req.sender},
        pair_risk=pair_risk,
        behavioural_anomaly=anom_score,
        coordination_score=coord_score,
        sub_scores=sub_scores,
    )
    overall_risk = float(engine_result["overall_risk"])
    risk_percent = float(engine_result["risk_percent"])
    action = str(engine_result["action"]).upper()

    # 2. NOW INSERT PAYMENT INTO GRAPH
    payment_id = f"PAY_{len(_payments_log) + 1:07d}"
    event: PaymentEvent = {
        "sender": req.sender,
        "receiver": req.receiver,
        "amount": req.amount,
        "timestamp": req.timestamp,
        "device": req.device,
        "payment_id": payment_id,
    }
    _payments_log.append(event)

    _transaction_graph.add_payment(
        sender=req.sender,
        receiver=req.receiver,
        amount=req.amount,
        timestamp=req.timestamp,
        transaction_id=payment_id,
    )
    if req.device:
        _identity_graph.add_account_identities(
            account_id=req.sender,
            device=req.device,
            observed_at=req.timestamp,
        )

    # Update cache degrees
    s_feat = _account_features(req.sender)
    r_feat = _account_features(req.receiver)
    s_feat["out_degree"] = len(_transaction_graph.get_out_neighbors(req.sender))
    r_feat["in_degree"] = len(_transaction_graph.get_in_neighbors(req.receiver))

    # 3. IF FLAGGED, ASSEMBLE FULL CASE FILE
    alert_id: Optional[str] = None
    if _should_alert(action, overall_risk):
        alert_id = f"ALERT_{payment_id}"
        pattern_dict = detect_for_payment(
            transaction_graph=_transaction_graph,
            identity_graph=_identity_graph,
            sender=req.sender,
            receiver=req.receiver,
            as_of=req.timestamp,
        )

        case = assemble_case_file(
            alert_id=alert_id,
            risk_score=overall_risk,
            action=action,
            sub_scores=sub_scores,
            pattern=pattern_dict,
            payment_events=_payments_log[-50:],
            shap_values=shap_values,
        )
        _alerts[alert_id] = {
            **case,
            "overall_risk": overall_risk,
            "risk_score": overall_risk,
            "risk_percent": risk_percent,
            "action": action,
            "status": "open",
            "created_at": req.timestamp,
            "trigger_payment_id": payment_id,
            "transaction": {
                "sender": req.sender,
                "receiver": req.receiver,
                "amount": req.amount,
                "device": req.device,
                "timestamp": req.timestamp,
            },
        }

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.debug("score latency_ms=%.2f sender=%s risk=%.3f action=%s", latency_ms, req.sender, overall_risk, action)

    return ScoreResponse(
        risk_score=overall_risk,
        overall_risk=overall_risk,
        risk_percent=risk_percent,
        pair_risk=round(pair_risk, 4),
        behavioural_anomaly=round(anom_score, 4),
        coordination_score=round(coord_score, 4),
        sender_anomaly=round(sub_scores["sender_anomaly"], 4),
        receiver_mule_propensity=round(sub_scores["receiver_mule_propensity"], 4),
        relationship_plausibility=round(sub_scores["relationship_plausibility"], 4),
        action=action,
        alert_id=alert_id,
    )


@app.get("/alerts", response_model=List[AlertSummary])
def list_alerts() -> List[AlertSummary]:
    open_alerts = [a for a in _alerts.values() if a.get("status") == "open"]
    open_alerts.sort(key=lambda a: float(a.get("overall_risk", a.get("risk_score", 0))), reverse=True)
    return [
        AlertSummary(
            alert_id=a["alert_id"],
            risk_score=float(a.get("overall_risk", a.get("risk_score", 0))),
            overall_risk=float(a.get("overall_risk", a.get("risk_score", 0))),
            risk_percent=float(a.get("risk_percent", round(float(a.get("overall_risk", a.get("risk_score", 0))) * 100.0, 2))),
            action=str(a.get("action", determine_action(float(a.get("overall_risk", a.get("risk_score", 0))), a.get("sub_scores")))),
            pattern=a.get("pattern"),
            created_at=str(a.get("created_at", "")),
            status=str(a.get("status", "open")),
        )
        for a in open_alerts
    ]


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: str) -> Dict[str, Any]:
    alert = _alerts.get(alert_id)
    if not alert:
        raise HTTPException(404, f"Alert {alert_id} not found")
    return alert


@app.post("/alerts/{alert_id}/verdict", response_model=VerdictResponse)
def post_verdict(alert_id: str, body: VerdictRequest) -> VerdictResponse:
    alert = _alerts.get(alert_id)
    if not alert:
        raise HTTPException(404, f"Alert {alert_id} not found")

    _verified_labels[alert_id] = body.verdict
    alert["status"] = "confirmed" if body.verdict == "confirm" else "cleared"

    risk_changes: List[RiskChange] = []
    if body.verdict == "confirm":
        # Run real Personalized PageRank risk propagation
        changes = propagate_from_confirmed(
            alert=alert,
            graph=_transaction_graph,
            feature_store=_account_feature_cache,
        )
        for ch in changes:
            risk_changes.append(
                RiskChange(
                    account_id=str(ch["account_id"]),
                    risk_before=float(ch["risk_before"]),
                    risk_after=float(ch["risk_after"]),
                )
            )

    return VerdictResponse(
        alert_id=alert_id,
        verdict=body.verdict,
        risk_changes=risk_changes,
    )


@app.get("/graph/snapshot")
def graph_snapshot(limit: int = 500) -> Dict[str, Any]:
    """Nodes and edges with current risk for the live dashboard view (F14)."""
    load_state()
    snap = _transaction_graph.snapshot()
    all_nodes = snap.get("nodes", [])
    all_edges = snap.get("edges", [])

    nodes_out: List[Dict[str, Any]] = []
    for node_info in all_nodes[:limit]:
        n_id = str(node_info["id"])
        feat = _account_features(n_id)
        boost = float(feat.get("risk_boost", 0.0))
        base_risk = min(1.0, 0.15 + boost + 0.05 * len(_transaction_graph.get_in_neighbors(n_id)))
        nodes_out.append({
            "id": n_id,
            "risk": round(base_risk, 4),
            "label": n_id,
        })

    node_ids = {n["id"] for n in nodes_out}
    edges_out: List[Dict[str, Any]] = []
    for edge in all_edges:
        u, v = str(edge.get("source")), str(edge.get("target"))
        if u in node_ids and v in node_ids:
            edges_out.append({
                "source": u,
                "target": v,
                "amount": round(float(edge.get("amount", 0)), 2),
                "timestamp": edge.get("timestamp"),
            })
        if len(edges_out) >= limit:
            break

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "stats": {
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "open_alerts": sum(1 for a in _alerts.values() if a.get("status") == "open"),
            "verified_count": len(_verified_labels),
        },
    }


@app.post("/admin/retrain", response_model=RetrainResponse)
def admin_retrain() -> RetrainResponse:
    global _model_version
    out = retrain(verified_labels=_verified_labels)
    _model_version = str(out.get("model_version", _model_version))
    return RetrainResponse(
        model_version=_model_version,
        metrics=dict(out.get("metrics", {})),
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "initialized": _initialized,
        "nodes_count": len(_transaction_graph.account_ids()),
        "cached_features": len(_account_feature_cache),
        "open_alerts": sum(1 for a in _alerts.values() if a.get("status") == "open"),
        "model_version": _model_version,
    }
