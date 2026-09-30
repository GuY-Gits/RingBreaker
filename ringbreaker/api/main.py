"""
RingBreaker API — F9 fast path, F14 dashboard backend.

Endpoints (PRD API contract):
  POST /score
  GET  /alerts
  GET  /alerts/{id}
  POST /alerts/{id}/verdict
  GET  /graph/snapshot
  POST /admin/retrain

In-memory feature store and NetworkX transaction graph. Slow-path feature
precomputation is cached at startup / refresh; the fast path is cache lookup
plus the pair model (pattern borrowed from Mule Hunter inference_service.py).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from explain.case_file import PaymentEvent, assemble_case_file

logger = logging.getLogger("ringbreaker.api")

# ── Optional teammate modules (PRD repo layout) ─────────────────────────────

try:
    from scoring import action as action_module
    from scoring import pair_model as pair_model_module
except ImportError:  # pragma: no cover — teammates wire scoring/
    action_module = None  # type: ignore
    pair_model_module = None  # type: ignore

try:
    from learn import propagate as propagate_module
    from learn import retrain as retrain_module
except ImportError:  # pragma: no cover
    propagate_module = None  # type: ignore
    retrain_module = None  # type: ignore

try:
    from patterns import detect as pattern_detect_module
except ImportError:  # pragma: no cover
    pattern_detect_module = None  # type: ignore

# ── Cache / graph state (Mule Hunter–style fast path) ─────────────────────────

RING_TIMEOUT_SEC = 20
MAX_RINGS_CACHED = 200
UNKNOWN_ACCOUNT_CACHE_MAX = 10_000

_payment_graph: nx.DiGraph = nx.DiGraph()
_account_feature_cache: Dict[str, Dict[str, Any]] = {}
_unknown_account_cache: Dict[str, Dict[str, Any]] = {}
_rings_cache: List[Dict[str, Any]] = []
_alerts: Dict[str, Dict[str, Any]] = {}
_payments_log: List[PaymentEvent] = []
_verified_labels: Dict[str, Literal["confirm", "clear"]] = {}
_model_version: str = "0.0.0"
_init_lock = Lock()
_initialized = False


def _precache_rings(g: nx.DiGraph) -> List[Dict[str, Any]]:
    """Bounded ring search — adapted from Mule Hunter inference_service."""
    rings: List[Dict[str, Any]] = []
    seen: set[frozenset] = set()
    sub = g.copy()
    deadline = time.monotonic() + RING_TIMEOUT_SEC

    for start in list(sub.nodes()):
        if time.monotonic() > deadline or len(rings) >= MAX_RINGS_CACHED:
            break
        stack: List[Tuple[str, List[str]]] = [(str(start), [str(start)])]
        while stack:
            if time.monotonic() > deadline or len(rings) >= MAX_RINGS_CACHED:
                break
            node, path = stack.pop()
            for nb in sub.successors(node):
                if len(path) > 6:
                    break
                nb_s = str(nb)
                if nb_s == start and len(path) >= 3:
                    key = frozenset(path)
                    if key not in seen:
                        seen.add(key)
                        vol = sum(
                            float(sub[path[i]][path[(i + 1) % len(path)]].get("amount", 0))
                            for i in range(len(path))
                        )
                        rings.append(
                            {
                                "nodes": path[:],
                                "size": len(path),
                                "volume": round(vol, 2),
                            }
                        )
                elif nb_s not in path:
                    stack.append((nb_s, path + [nb_s]))

    rings.sort(key=lambda r: r["volume"], reverse=True)
    logger.info("Ring pre-cache: %d rings", len(rings))
    return rings


def _warm_feature_cache() -> None:
    """Slow-path hook: teammates populate _account_feature_cache from feature store."""
    global _rings_cache
    _rings_cache = _precache_rings(_payment_graph)


def _account_features(account_id: str) -> Dict[str, Any]:
    if account_id in _account_feature_cache:
        return _account_feature_cache[account_id]
    if account_id in _unknown_account_cache:
        return _unknown_account_cache[account_id]

    base = {
        "account_id": account_id,
        "risk_boost": 0.0,
        "in_degree": _payment_graph.in_degree(account_id) if _payment_graph.has_node(account_id) else 0,
        "out_degree": _payment_graph.out_degree(account_id) if _payment_graph.has_node(account_id) else 0,
    }
    if len(_unknown_account_cache) >= UNKNOWN_ACCOUNT_CACHE_MAX:
        _unknown_account_cache.pop(next(iter(_unknown_account_cache)))
    _unknown_account_cache[account_id] = base
    return base


def load_state() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        logger.info("RingBreaker API — warming caches")
        _warm_feature_cache()
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
    sender_anomaly: float
    receiver_mule_propensity: float
    relationship_plausibility: float
    action: str
    alert_id: Optional[str] = None


class AlertSummary(BaseModel):
    alert_id: str
    risk_score: float
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


def _default_score(
    sender: str,
    receiver: str,
    amount: float,
    device: Optional[str],
    timestamp: str,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Fallback when scoring.pair_model is not yet wired."""
    s_feat = _account_features(sender)
    r_feat = _account_features(receiver)
    boost_s = float(s_feat.get("risk_boost", 0.0))
    boost_r = float(r_feat.get("risk_boost", 0.0))
    sender_anomaly = min(1.0, 0.15 + boost_s + 0.01 * float(s_feat.get("out_degree", 0)))
    receiver_mule = min(1.0, 0.15 + boost_r + 0.02 * float(r_feat.get("in_degree", 0)))
    relationship = 0.25 if not _payment_graph.has_edge(sender, receiver) else 0.1
    combined = min(1.0, (sender_anomaly + receiver_mule + relationship) / 3.0 + amount / 1_000_000)
    sub_scores = {
        "sender_anomaly": sender_anomaly,
        "receiver_mule_propensity": receiver_mule,
        "relationship_plausibility": relationship,
    }
    shap = {k: v * 0.33 for k, v in sub_scores.items()}
    return combined, sub_scores, shap


def _score_payment(req: ScoreRequest) -> Tuple[float, Dict[str, float], Dict[str, float], str]:
    if pair_model_module and hasattr(pair_model_module, "score_payment"):
        result = pair_model_module.score_payment(
            sender=req.sender,
            receiver=req.receiver,
            amount=req.amount,
            device=req.device,
            timestamp=req.timestamp,
            graph=_payment_graph,
            feature_cache=_account_feature_cache,
        )
        sub = {
            "sender_anomaly": float(result["sender_anomaly"]),
            "receiver_mule_propensity": float(result["receiver_mule_propensity"]),
            "relationship_plausibility": float(result["relationship_plausibility"]),
        }
        return float(result["risk_score"]), sub, dict(result.get("shap_values", {})), str(
            result.get("model_version", _model_version)
        )

    combined, sub, shap = _default_score(
        req.sender, req.receiver, req.amount, req.device, req.timestamp
    )
    return combined, sub, shap, _model_version


def _decide_action(sub_scores: Dict[str, float], risk_score: float) -> str:
    if action_module and hasattr(action_module, "decide_action"):
        return str(action_module.decide_action(sub_scores, risk_score))
    dominant = max(sub_scores, key=sub_scores.get)
    score = float(sub_scores[dominant])
    if score >= 0.75 or risk_score >= 0.8:
        return "block"
    if score >= 0.55:
        return "hold_receiver" if dominant == "receiver_mule_propensity" else "hold_receiver"
    if score >= 0.35:
        return "warn_sender"
    return "allow"


def _detect_pattern(sender: str, receiver: str) -> Optional[Dict[str, Any]]:
    if pattern_detect_module and hasattr(pattern_detect_module, "detect_for_payment"):
        return pattern_detect_module.detect_for_payment(_payment_graph, sender, receiver)
    for ring in _rings_cache:
        nodes = set(ring.get("nodes", []))
        if sender in nodes or receiver in nodes:
            members = list(nodes)
            return {
                "pattern_name": "closed_loop",
                "members": members,
                "roles": {m: "MULE" for m in members},
                "subgraph": {
                    "nodes": [{"id": m, "role": "MULE"} for m in members],
                    "edges": [
                        {"source": u, "target": v}
                        for u, v in _payment_graph.edges()
                        if u in nodes and v in nodes
                    ],
                },
            }
    return None


def _append_graph(req: ScoreRequest) -> None:
    s, r = req.sender, req.receiver
    if not _payment_graph.has_node(s):
        _payment_graph.add_node(s)
    if not _payment_graph.has_node(r):
        _payment_graph.add_node(r)
    if _payment_graph.has_edge(s, r):
        data = _payment_graph[s][r]
        data["amount"] = float(data.get("amount", 0)) + req.amount
        data["count"] = int(data.get("count", 1)) + 1
        data["last_timestamp"] = req.timestamp
    else:
        _payment_graph.add_edge(
            s,
            r,
            amount=req.amount,
            count=1,
            timestamp=req.timestamp,
            device=req.device,
        )


def _should_alert(action: str, risk_score: float) -> bool:
    return action != "allow" or risk_score >= 0.5


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
def score_payment(req: ScoreRequest) -> ScoreResponse:
    t0 = time.perf_counter()
    load_state()
    _append_graph(req)

    payment_id = str(uuid.uuid4())
    event: PaymentEvent = {
        "sender": req.sender,
        "receiver": req.receiver,
        "amount": req.amount,
        "timestamp": req.timestamp,
        "device": req.device,
        "payment_id": payment_id,
    }
    _payments_log.append(event)

    risk_score, sub_scores, shap_values, _ = _score_payment(req)
    action = _decide_action(sub_scores, risk_score)

    alert_id: Optional[str] = None
    if _should_alert(action, risk_score):
        alert_id = str(uuid.uuid4())
        pattern = _detect_pattern(req.sender, req.receiver)
        pattern_hit = pattern if pattern else None
        case = assemble_case_file(
            alert_id=alert_id,
            risk_score=risk_score,
            action=action,
            sub_scores=sub_scores,
            pattern=pattern_hit,
            payment_events=_payments_log[-50:],
            shap_values=shap_values,
        )
        _alerts[alert_id] = {
            **case,
            "status": "open",
            "created_at": req.timestamp,
            "trigger_payment_id": payment_id,
        }

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.debug("score latency_ms=%.2f sender=%s", latency_ms, req.sender)

    return ScoreResponse(
        risk_score=round(risk_score, 4),
        sender_anomaly=round(sub_scores["sender_anomaly"], 4),
        receiver_mule_propensity=round(sub_scores["receiver_mule_propensity"], 4),
        relationship_plausibility=round(sub_scores["relationship_plausibility"], 4),
        action=action,
        alert_id=alert_id,
    )


@app.get("/alerts", response_model=List[AlertSummary])
def list_alerts() -> List[AlertSummary]:
    open_alerts = [a for a in _alerts.values() if a.get("status") == "open"]
    open_alerts.sort(key=lambda a: float(a.get("risk_score", 0)), reverse=True)
    return [
        AlertSummary(
            alert_id=a["alert_id"],
            risk_score=float(a["risk_score"]),
            action=str(a["action"]),
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
        raise HTTPException(404, "Alert not found")
    return alert


@app.post("/alerts/{alert_id}/verdict", response_model=VerdictResponse)
def post_verdict(alert_id: str, body: VerdictRequest) -> VerdictResponse:
    alert = _alerts.get(alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found")

    _verified_labels[alert_id] = body.verdict
    alert["status"] = "confirmed" if body.verdict == "confirm" else "cleared"

    risk_changes: List[RiskChange] = []
    if body.verdict == "confirm" and propagate_module and hasattr(
        propagate_module, "propagate_from_confirmed"
    ):
        changes = propagate_module.propagate_from_confirmed(
            alert=alert,
            graph=_payment_graph,
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
    elif body.verdict == "confirm":
        members = alert.get("members") or []
        seed = members or [alert.get("timeline", [{}])[0].get("sender")]
        for acc in seed:
            if not acc:
                continue
            before = float(_account_features(str(acc)).get("risk_boost", 0.0))
            after = min(1.0, before + 0.25)
            _account_feature_cache[str(acc)] = {
                **_account_features(str(acc)),
                "risk_boost": after,
            }
            for nb in set(_payment_graph.predecessors(acc)) | set(_payment_graph.successors(acc)):
                nb_s = str(nb)
                b2 = float(_account_features(nb_s).get("risk_boost", 0.0))
                a2 = min(1.0, b2 + 0.12)
                _account_feature_cache[nb_s] = {
                    **_account_features(nb_s),
                    "risk_boost": a2,
                }
                risk_changes.append(
                    RiskChange(account_id=nb_s, risk_before=b2, risk_after=a2)
                )
            risk_changes.append(
                RiskChange(account_id=str(acc), risk_before=before, risk_after=after)
            )

    return VerdictResponse(
        alert_id=alert_id,
        verdict=body.verdict,
        risk_changes=risk_changes,
    )


@app.get("/graph/snapshot")
def graph_snapshot(limit: int = 500) -> Dict[str, Any]:
    """Nodes and edges for the live dashboard view (F14)."""
    load_state()
    nodes_out: List[Dict[str, Any]] = []
    for node in list(_payment_graph.nodes())[:limit]:
        feat = _account_features(str(node))
        risk = min(
            1.0,
            0.2
            + float(feat.get("risk_boost", 0.0))
            + 0.05 * (int(feat.get("in_degree", 0)) + int(feat.get("out_degree", 0))),
        )
        nodes_out.append({"id": str(node), "risk": round(risk, 4)})

    node_ids = {n["id"] for n in nodes_out}
    edges_out: List[Dict[str, Any]] = []
    for u, v, data in _payment_graph.edges(data=True):
        if str(u) in node_ids and str(v) in node_ids:
            edges_out.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "amount": round(float(data.get("amount", 0)), 2),
                    "timestamp": data.get("last_timestamp") or data.get("timestamp"),
                }
            )
        if len(edges_out) >= limit:
            break

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "stats": {
            "total_nodes": _payment_graph.number_of_nodes(),
            "total_edges": _payment_graph.number_of_edges(),
            "open_alerts": sum(1 for a in _alerts.values() if a.get("status") == "open"),
            "rings_cached": len(_rings_cache),
        },
    }


@app.post("/admin/retrain", response_model=RetrainResponse)
def admin_retrain() -> RetrainResponse:
    global _model_version
    if retrain_module and hasattr(retrain_module, "retrain"):
        out = retrain_module.retrain(verified_labels=_verified_labels)
        _model_version = str(out.get("model_version", _model_version))
        return RetrainResponse(model_version=_model_version, metrics=dict(out.get("metrics", {})))

    _model_version = f"0.0.{len(_verified_labels)}"
    _warm_feature_cache()
    return RetrainResponse(
        model_version=_model_version,
        metrics={"verified_labels": len(_verified_labels), "status": "stub_retrain"},
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "initialized": _initialized,
        "graph_nodes": _payment_graph.number_of_nodes(),
        "feature_cache_size": len(_account_feature_cache),
        "rings_cached": len(_rings_cache),
        "model_version": _model_version,
    }
