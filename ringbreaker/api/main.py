"""RingBreaker API (F9 fast path, F14 dashboard backend).

PRD contract (served at the root and mirrored under ``/api`` for the dashboard):
  POST /score                  score one payment
  GET  /alerts                 alert queue, highest risk first
  GET  /alerts/{id}            case file
  POST /alerts/{id}/verdict    confirm / clear -> risk changes
  GET  /graph/snapshot         nodes + edges with current risk
  POST /admin/retrain          retrain with verified labels

Dashboard support: /overview, /payments/recent, /accounts, /accounts/{id},
/patterns, /learning, /stream/*, /health. The built dashboard is served at /app.

All state lives in one ``Engine``; this module only validates and translates.

    uvicorn ringbreaker.api.main:app --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ringbreaker import config
from ringbreaker.engine import Engine, EngineError
from ringbreaker.engine.stream import StreamRunner

log = logging.getLogger("ringbreaker.api")

_engine: Optional[Engine] = None
_stream: Optional[StreamRunner] = None


def get_engine() -> Engine:
    global _engine, _stream
    if _engine is None:
        _engine = Engine()
        _stream = StreamRunner(_engine)
    return _engine


def get_stream() -> StreamRunner:
    get_engine()
    assert _stream is not None
    return _stream


def set_engine(engine: Engine) -> None:
    """Swap the engine (tests)."""
    global _engine, _stream
    if _stream is not None:
        _stream.pause()
    _engine = engine
    _stream = StreamRunner(engine)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except EngineError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


# ── request / response models ─────────────────────────────────────────────


class ScoreRequest(BaseModel):
    sender: str = Field(min_length=1, max_length=64)
    receiver: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=1e9, allow_inf_nan=False)
    timestamp: str = Field(min_length=4, max_length=40)
    device: Optional[str] = Field(default=None, max_length=128)
    ip: Optional[str] = Field(default=None, max_length=64)
    transaction_id: Optional[str] = Field(default=None, max_length=64)


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    overall_risk: float
    risk_percent: float
    sender_anomaly: float
    receiver_mule_propensity: float
    relationship_plausibility: float
    sub_scores: Dict[str, float]
    signals: Dict[str, float]
    action: str
    alert_id: Optional[str] = None
    reasons: List[Dict[str, Any]]
    latency_ms: float


class AccountRequest(BaseModel):
    account_id: str = Field(min_length=1, max_length=64)
    signup_at: Optional[str] = Field(default=None, max_length=40)
    device: Optional[str] = Field(default=None, max_length=128)
    phone: Optional[str] = Field(default=None, max_length=32)
    ip: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = Field(default=None, max_length=256)


class VerdictRequest(BaseModel):
    verdict: Literal["confirm", "clear"]
    accounts: Optional[List[str]] = Field(default=None, max_length=100)
    note: Optional[str] = Field(default=None, max_length=500)


class StreamConfig(BaseModel):
    rate: Optional[float] = Field(default=None, gt=0, le=200)
    pause_on_block: Optional[bool] = None


# ── routes ────────────────────────────────────────────────────────────────

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, Any]:
    e = get_engine()
    return {"status": "ok", "model_version": e.models.version, "clock": e.clock.isoformat() if e.clock else None,
            "accounts": len(e.accounts), "open_alerts": sum(1 for a in e.alerts.values() if a["status"] == "open")}


@router.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> Dict[str, Any]:
    out = _call(get_engine().score, req.model_dump(), "api")
    return {**out, **out["sub_scores"]}


@router.post("/accounts")
def register_account(req: AccountRequest) -> Dict[str, Any]:
    return _call(get_engine().register_account, req.account_id, req.signup_at, req.device, req.phone,
                 req.ip, req.address)


@router.get("/alerts")
def alerts(status: Literal["open", "confirmed", "cleared", "all"] = "open",
           period: Literal["live", "history", "all"] = "all",
           limit: int = Query(200, ge=1, le=1000)) -> List[Dict[str, Any]]:
    return get_engine().list_alerts(status=status, limit=limit, period=period)


@router.get("/alerts/{alert_id}")
def alert_case(alert_id: str) -> Dict[str, Any]:
    return _call(get_engine().get_case, alert_id)


@router.post("/alerts/{alert_id}/verdict")
def verdict(alert_id: str, body: VerdictRequest) -> Dict[str, Any]:
    return _call(get_engine().verdict, alert_id, body.verdict, body.accounts, body.note)


@router.get("/graph/snapshot")
def graph_snapshot(
    focus: Optional[str] = Query(None, max_length=64),
    hops: int = Query(1, ge=1, le=2),
    pattern: Optional[str] = Query(None, max_length=2000),
    alert: Optional[str] = Query(None, max_length=80),
    limit: int = Query(250, ge=10, le=600),
    include_identity: bool = True,
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    return _call(get_engine().graph_snapshot, focus=focus, hops=hops, pattern=pattern, alert=alert,
                 limit=limit, include_identity=include_identity, min_risk=min_risk)


@router.get("/overview")
def overview() -> Dict[str, Any]:
    data = get_engine().overview()
    data["stream"] = get_stream().status()
    return data


@router.get("/payments/recent")
def recent_payments(limit: int = Query(50, ge=1, le=500), flagged_only: bool = False) -> List[Dict[str, Any]]:
    return get_engine().recent_payments(limit=limit, flagged_only=flagged_only)


@router.get("/accounts")
def accounts(limit: int = Query(50, ge=1, le=500), q: Optional[str] = Query(None, max_length=64),
             min_risk: float = Query(0.0, ge=0.0, le=1.0)) -> List[Dict[str, Any]]:
    return get_engine().list_accounts(limit=limit, query=q, min_risk=min_risk)


@router.get("/accounts/{account_id}")
def account(account_id: str) -> Dict[str, Any]:
    return _call(get_engine().account_profile, account_id)


@router.get("/patterns")
def patterns(include_inactive: bool = False) -> List[Dict[str, Any]]:
    return get_engine().list_patterns(include_inactive=include_inactive)


@router.get("/learning")
def learning() -> Dict[str, Any]:
    return get_engine().learning_state()


@router.post("/admin/retrain")
def admin_retrain() -> Dict[str, Any]:
    get_stream().pause()
    out = _call(get_engine().retrain)
    return {"model_version": out["version"], **out}


@router.get("/stream/status")
def stream_status() -> Dict[str, Any]:
    return get_stream().status()


@router.post("/stream/start")
def stream_start(cfg: Optional[StreamConfig] = None) -> Dict[str, Any]:
    s = get_stream()
    if cfg:
        s.configure(cfg.rate, cfg.pause_on_block)
    s.start()
    return s.status()


@router.post("/stream/pause")
def stream_pause() -> Dict[str, Any]:
    s = get_stream()
    s.pause()
    return s.status()


@router.post("/stream/config")
def stream_config(cfg: StreamConfig) -> Dict[str, Any]:
    s = get_stream()
    s.configure(cfg.rate, cfg.pause_on_block)
    return s.status()


@router.post("/stream/step")
def stream_step(n: int = Query(1, ge=1, le=500)) -> Dict[str, Any]:
    s = get_stream()
    if s.running:
        raise HTTPException(409, "pause the stream before stepping")
    s.step(n)
    return s.status()


@router.post("/stream/reset")
def stream_reset() -> Dict[str, Any]:
    s = get_stream()
    s.reset()
    return s.status()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    yield
    if _stream is not None:
        _stream.pause()


app = FastAPI(title="RingBreaker", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.include_router(router)
app.include_router(router, prefix="/api")

if (config.DASHBOARD_DIST / "index.html").exists():
    app.mount("/app/assets", StaticFiles(directory=config.DASHBOARD_DIST / "assets"), name="assets")

    @app.get("/app/{path:path}", include_in_schema=False)
    def dashboard(path: str) -> FileResponse:
        candidate = (config.DASHBOARD_DIST / path).resolve()
        if path and candidate.is_file() and config.DASHBOARD_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(config.DASHBOARD_DIST / "index.html")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/app/")
