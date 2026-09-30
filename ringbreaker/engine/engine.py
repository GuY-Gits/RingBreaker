"""RingBreaker engine: the single owner of runtime state.

Data flow for every payment (``Engine.score``)::

    payment ─► validate + clock check (reject out-of-order → no future leakage)
            ─► online feature store: 39 pair + 12 behaviour features as of T
            ─► XGBoost pair model (+SHAP) ─┐
            ─► calibrated EIF anomaly ─────┤
            ─► lockstep cluster (slow path)┼─► fusion ─► action (F11)
            ─► propagated risk (verdicts) ─┘
            ─► graph + identity graph + feature store updated with the payment
            ─► if action != ALLOW: alert + case file (patterns as of T)
            ─► every N payments: slow path (named patterns, lockstep)

Analyst verdicts (F13) spread risk with personalized PageRank over the payment
and identity graphs; retraining (F19) folds verified labels into the pair
model and hot-swaps it. The API, the stream replayer and the evaluator all go
through this class, so there is one graph, one feature store, one alert queue.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

from ringbreaker import config
from ringbreaker.engine import case_file
from ringbreaker.engine.models import ModelBundle
from ringbreaker.engine.risk import action_for, build_reasons, compute_sub_scores, fuse
from ringbreaker.features.flow import account_flow_features
from ringbreaker.features.lifelike import account_lifelikeness
from ringbreaker.features.lockstep import detect_lockstep
from ringbreaker.features.online import FeaturePayment, OnlineFeatureState, sanitize
from ringbreaker.features.social import pair_social_features
from ringbreaker.graph_metrics import account_graph_metrics
from ringbreaker.graphs.build import IdentityGraph, TransactionGraph, parse_timestamp
from ringbreaker.patterns import detect_all_patterns
from ringbreaker.patterns.chain import detect_pass_through_chains
from ringbreaker.split import timeline_split

log = logging.getLogger("ringbreaker.engine")

ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
IDENTITY_TYPES = ("device", "phone", "ip", "address")
EXPOSURE_WINDOW = timedelta(hours=72)
MAX_AMOUNT = 1e9
# Most specific evidence first when an alert touches several patterns.
PATTERN_PRIORITY = {"closed_loop": 0, "pass_through_chain": 1, "lockstep_cluster": 2,
                    "fan_in_collector": 3, "shared_device_star": 4}
TAINT_MIN = 0.5
TAINT_DECAY = 0.8
PPR_SCALE = 0.6  # tuned on the held-out stream: 46/46 recall, 2 FPs after one confirm
PPR_CAP = 0.6


class EngineError(ValueError):
    """Invalid input; ``status`` is the HTTP status the API should return."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts else None


def _finite(x: float, default: float = 0.0) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) or math.isinf(x) else x


class Engine:
    def __init__(self, load_history: bool = True, models: Optional[ModelBundle] = None) -> None:
        self.lock = threading.RLock()
        self.models = models or ModelBundle()
        self._load_history = load_history
        self._base_model = (self.models.booster, self.models.version, self.models.pair_path)
        self.reset()

    # ════════════════════════════════════════════════════════════════════
    # state lifecycle
    # ════════════════════════════════════════════════════════════════════
    def reset(self) -> None:
        with self.lock:
            self.graph = TransactionGraph()
            self.identity = IdentityGraph()
            self.state = OnlineFeatureState()
            self.accounts: Dict[str, Dict[str, Any]] = {}
            self.payments: List[Dict[str, Any]] = []
            self.payment_index: Dict[str, Dict[str, Any]] = {}
            self.alerts: Dict[str, Dict[str, Any]] = {}
            self.exposure: Dict[str, List[tuple]] = defaultdict(list)
            self.propagated: Dict[str, float] = {}
            self.propagation_source: Dict[str, str] = {}
            self.confirmed_accounts: Set[str] = set()
            self.cleared_accounts: Set[str] = set()
            self.verdicts: List[Dict[str, Any]] = []
            self.verified_labels: Dict[str, int] = {}
            self.model_history: List[Dict[str, Any]] = []
            self.patterns: Dict[str, Dict[str, Any]] = {}
            self.lockstep: Dict[str, Dict[str, Any]] = {}
            self.latencies: List[float] = []
            self.clock: Optional[datetime] = None
            self.stream_rows: List[Dict[str, Any]] = []
            self.history_count = 0
            self._since_slow = 0
            self._slow_runs = 0
            booster, version, path = self._base_model
            self.models.use_booster(booster, version, path)
            self.model_history.append(
                {"version": version, "at": None, "kind": "base", "metrics": {}, "path": path}
            )
            if self._load_history:
                self._boot_from_csv()
            self.run_slow_path()

    def _boot_from_csv(self) -> None:
        if not config.PAYMENTS_CSV.exists() or not config.USERS_CSV.exists():
            log.warning("No simulator CSVs in %s; starting empty", config.DATA_DIR)
            return
        users = pd.read_csv(config.USERS_CSV)
        for u in users.itertuples(index=False):
            self.register_account(
                str(u.user_id),
                signup_at=u.signup_timestamp,
                device=u.device_id,
                phone=u.phone,
                ip=u.ip_address,
                address=u.address,
                _internal=True,
            )
        payments = pd.read_csv(config.PAYMENTS_CSV)
        payments["timestamp"] = pd.to_datetime(payments["timestamp"])
        payments = payments.sort_values("timestamp", kind="stable").reset_index(drop=True)
        split = timeline_split(payments["timestamp"])
        history = payments.iloc[: split.stream_start_row]
        for row in history.itertuples(index=False):
            self._apply(
                str(row.transaction_id), str(row.sender), str(row.receiver), float(row.amount),
                row.timestamp.to_pydatetime(),
                str(row.device_id) if pd.notnull(row.device_id) else None,
                str(row.ip_address) if pd.notnull(row.ip_address) else None,
            )
        self.history_count = len(history)
        # Stream rows carry only what a payment system would send: no labels.
        stream = payments.iloc[split.stream_start_row:]
        self.stream_rows = [
            {
                "transaction_id": str(r.transaction_id),
                "sender": str(r.sender),
                "receiver": str(r.receiver),
                "amount": float(r.amount),
                "timestamp": r.timestamp.isoformat(),
                "device": str(r.device_id) if pd.notnull(r.device_id) else None,
                "ip": str(r.ip_address) if pd.notnull(r.ip_address) else None,
            }
            for r in stream.itertuples(index=False)
        ]
        log.info("Booted: %d accounts, %d historical payments, %d stream payments",
                 len(self.accounts), self.history_count, len(self.stream_rows))

    # ════════════════════════════════════════════════════════════════════
    # accounts (F4)
    # ════════════════════════════════════════════════════════════════════
    def register_account(
        self,
        account_id: str,
        signup_at: Any = None,
        device: Optional[str] = None,
        phone: Optional[str] = None,
        ip: Optional[str] = None,
        address: Optional[str] = None,
        _internal: bool = False,
    ) -> Dict[str, Any]:
        with self.lock:
            account_id = str(account_id)
            if not _internal:
                self._check_id(account_id, "account_id")
            ts = parse_timestamp(signup_at) if signup_at not in (None, "") else None
            if ts is None and not _internal:
                ts = self.clock
            if not _internal and ts is not None and self.clock is not None and ts > self.clock + timedelta(days=1):
                raise EngineError("signup_at is in the future relative to the stream clock")
            self.graph.register_account(account_id, signup_at=ts)
            self.state.register(account_id, ts)
            observed = ts or self.clock
            for itype, value in zip(IDENTITY_TYPES, (device, phone, ip, address)):
                if value not in (None, "") and not (isinstance(value, float) and math.isnan(value)):
                    self.identity.add_identity_link(account_id, itype, str(value), observed_at=observed)
            meta = self.accounts.setdefault(account_id, {"id": account_id})
            meta["signup_at"] = ts
            return {"account_id": account_id, "signup_at": _iso(ts), "registered": True}

    def _ensure_account(self, account_id: str) -> None:
        if account_id not in self.accounts:
            self.accounts[account_id] = {"id": account_id, "signup_at": None}
            self.graph.register_account(account_id)

    @staticmethod
    def _check_id(value: str, field: str) -> None:
        if not ID_RE.match(value or ""):
            raise EngineError(f"{field} must be 1-64 characters of letters, digits, _ . : -")

    # ════════════════════════════════════════════════════════════════════
    # fast path (F9, F10, F11)
    # ════════════════════════════════════════════════════════════════════
    def _apply(self, tx_id: str, s: str, r: str, amount: float, ts: datetime,
               device: Optional[str], ip: Optional[str]) -> None:
        self._ensure_account(s)
        self._ensure_account(r)
        self.graph.add_payment(s, r, amount, ts, transaction_id=tx_id)
        if device:
            self.identity.add_identity_link(s, "device", device, observed_at=ts)
        if ip:
            self.identity.add_identity_link(s, "ip", ip, observed_at=ts)
        self.state.update(FeaturePayment(s, r, amount, ts, device, ip))
        self.clock = ts if self.clock is None or ts > self.clock else self.clock

    def score(self, req: Dict[str, Any], source: str = "api") -> Dict[str, Any]:
        s = str(req.get("sender", "")).strip()
        r = str(req.get("receiver", "")).strip()
        self._check_id(s, "sender")
        self._check_id(r, "receiver")
        if s == r:
            raise EngineError("sender and receiver must differ")
        amount = _finite(req.get("amount"), default=-1.0)
        if not (0 < amount <= MAX_AMOUNT):
            raise EngineError("amount must be a finite number between 0 and 1e9")
        try:
            ts = parse_timestamp(req.get("timestamp"))
        except (TypeError, ValueError):
            raise EngineError("timestamp must be an ISO-8601 datetime")
        device = str(req["device"]).strip() if req.get("device") else None
        ip = str(req["ip"]).strip() if req.get("ip") else None
        tx_id = req.get("transaction_id")
        if tx_id is not None:
            self._check_id(str(tx_id), "transaction_id")

        with self.lock:
            if self.clock is not None and ts < self.clock:
                raise EngineError(
                    f"payment at {ts.isoformat()} is older than the engine clock "
                    f"{self.clock.isoformat()}; out-of-order payments are rejected so "
                    "features never include later payments",
                    status=409,
                )
            tx_id = str(tx_id) if tx_id else f"PAY_{len(self.payments) + 1:07d}"
            if tx_id in self.payment_index:
                raise EngineError(f"transaction_id {tx_id} was already scored", status=409)
            t0 = time.perf_counter()
            self._ensure_account(s)
            self._ensure_account(r)

            fp = FeaturePayment(s, r, amount, ts, device, ip)
            feats = sanitize(self.state.pair_features(fp))
            behaviour = sanitize(self.state.behaviour_features(fp))
            pair_risk, shap = self.models.pair_predict(feats)
            anomaly = self.models.anomaly(behaviour)
            anomaly = 0.0 if anomaly is None else anomaly
            lock_s, lock_r = self.lockstep.get(s), self.lockstep.get(r)
            coordination = max((c["score"] for c in (lock_s, lock_r) if c), default=0.0)
            network = max(self.propagated.get(s, 0.0), self.propagated.get(r, 0.0))

            ctx = self._score_context(s, r, ts, network)
            ctx["anomaly"] = anomaly
            ctx["lockstep_size"] = max((c["size"] for c in (lock_s, lock_r) if c), default=0)
            sub = compute_sub_scores(feats, ctx)
            overall = fuse(pair_risk, anomaly, coordination, network)
            action = action_for(overall, sub)
            signals = {
                "pair_risk": round(pair_risk, 4),
                "anomaly": round(anomaly, 4),
                "coordination": round(coordination, 4),
                "network_risk": round(network, 4),
            }
            reasons = build_reasons(feats, ctx, signals)

            self._apply(tx_id, s, r, amount, ts, device, ip)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            record = {
                "transaction_id": tx_id,
                "timestamp": ts.isoformat(),
                "sender": s,
                "receiver": r,
                "amount": round(amount, 2),
                "device": device,
                "ip": ip,
                "source": source,
                "risk_score": overall,
                "overall_risk": overall,
                "risk_percent": round(overall * 100, 2),
                "signals": signals,
                "sub_scores": sub,
                "action": action,
                "reasons": reasons,
                "alert_id": None,
                "latency_ms": round(latency_ms, 2),
                "_features": feats,
                "_shap": shap,
                "_context": ctx,
            }
            self.payments.append(record)
            self.payment_index[tx_id] = record
            self.latencies.append(latency_ms)
            for acc in (s, r):
                self.exposure[acc].append((ts, overall, tx_id))
            self._taint(s, r, tx_id)

            if action != "ALLOW":
                record["alert_id"] = self._open_alert(record)

            self._since_slow += 1
            if self._since_slow >= config.SLOW_PATH_EVERY:
                self.run_slow_path()
            return self.public_payment(record)

    def _score_context(self, s: str, r: str, ts: datetime, network: float) -> Dict[str, Any]:
        lifelike = account_lifelikeness(self.graph, r, as_of=ts)
        flow = account_flow_features(self.graph, r, as_of=ts)
        return {
            "social": pair_social_features(self.graph, s, r, as_of=ts),
            "receiver_lifelikeness": float(lifelike["lifelikeness_score"]),
            "receiver_pass_through": float(flow["pass_through_ratio"]),
            "receiver_identity_share": self._identity_share(r, ts),
            "sender_propagated": self.propagated.get(s, 0.0),
            "receiver_propagated": self.propagated.get(r, 0.0),
            "network": network,
            "patterns": [self._pattern_brief(p) for p in self.patterns_for({s, r})],
        }

    def _identity_share(self, account: str, as_of: Optional[datetime]) -> int:
        best = 1
        for ident in self.identity.get_identities(account, as_of=as_of):
            n = len(self.identity.get_accounts_for_identity(
                ident["identity_type"], ident["identity_value"], as_of=as_of))
            best = max(best, n)
        return best

    @staticmethod
    def public_payment(rec: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    # ════════════════════════════════════════════════════════════════════
    # slow path (F8, F15)
    # ════════════════════════════════════════════════════════════════════
    def run_slow_path(self) -> None:
        with self.lock:
            self._since_slow = 0
            self._slow_runs += 1
            if self.clock is None:
                return
            as_of = self.clock
            found = detect_all_patterns(self.graph, self.identity, as_of=as_of)
            found = [p for p in found if p.pattern_name != "pass_through_chain"]
            found += detect_pass_through_chains(self.graph, as_of=as_of, maximal_only=True)
            live_ids = set()
            for p in found:
                entry = self._pattern_entry(p.to_dict())
                live_ids.add(entry["id"])
            clusters = detect_lockstep(self.graph, as_of=as_of)
            self.lockstep = {}
            for c in clusters:
                if not c["suspicious"]:
                    continue
                pid = self._register_pattern(
                    ptype="lockstep_cluster",
                    members=sorted(c["members"]),
                    roles={"cluster_members": sorted(c["members"])},
                    evidence={k: v for k, v in c["evidence"].items()},
                    subgraph=None,
                    score=c["lockstep_score"],
                )
                live_ids.add(pid)
                for m in c["members"]:
                    self.lockstep[m] = {"score": round(c["lockstep_score"], 4),
                                        "size": c["cluster_size"], "pattern_id": pid}
            for pid, pat in self.patterns.items():
                pat["active"] = pid in live_ids

    def _pattern_entry(self, d: Dict[str, Any]) -> Dict[str, Any]:
        pid = self._register_pattern(
            ptype=d["pattern_name"], members=list(d["members"]), roles=d.get("roles", {}),
            evidence=d.get("evidence", {}), subgraph=d.get("subgraph"),
        )
        return self.patterns[pid]

    def _register_pattern(self, ptype: str, members: List[str], roles: Dict[str, Any],
                          evidence: Dict[str, Any], subgraph: Optional[Dict[str, Any]],
                          score: Optional[float] = None) -> str:
        key = members if ptype in ("pass_through_chain",) else sorted(members)
        pid = f"{ptype}:{'-'.join(key)}"
        if ptype == "lockstep_cluster":
            # Clusters grow as members become active; keep one stable entry per core.
            for existing_id, existing in self.patterns.items():
                if existing["type"] == ptype and len(set(existing["members"]) & set(members)) >= 3:
                    pid = existing_id
                    break
        txs = evidence.get("transactions") or []
        stamps = [t["timestamp"] for t in txs if t.get("timestamp")]
        entry = self.patterns.get(pid)
        if entry is None:
            entry = {
                "id": pid,
                "type": ptype,
                "first_detected_at": _iso(self.clock),
                "detections": 0,
            }
            self.patterns[pid] = entry
        entry.update({
            "members": members,
            "roles": roles,
            "evidence": evidence,
            "subgraph": subgraph,
            "score": score,
            "window_start": min(stamps) if stamps else None,
            "window_end": max(stamps) if stamps else None,
            "total_amount": round(sum(float(t.get("amount", 0)) for t in txs), 2),
            "last_detected_at": _iso(self.clock),
            "active": True,
        })
        entry["detections"] += 1
        return pid

    def patterns_for(self, accounts: Iterable[str], active_only: bool = True) -> List[Dict[str, Any]]:
        accs = set(accounts)
        return [p for p in self.patterns.values()
                if (p["active"] or not active_only) and accs & set(p["members"])]

    @staticmethod
    def _pattern_brief(p: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": p["id"], "type": p["type"], "members": p["members"]}

    # ════════════════════════════════════════════════════════════════════
    # alerts + case files (F12)
    # ════════════════════════════════════════════════════════════════════
    def _open_alert(self, record: Dict[str, Any]) -> str:
        alert_id = f"ALERT_{record['transaction_id']}"
        ts = parse_timestamp(record["timestamp"])
        fresh = detect_all_patterns(self.graph, self.identity, as_of=ts)
        fresh = [p for p in fresh if p.pattern_name != "pass_through_chain"]
        fresh += detect_pass_through_chains(self.graph, as_of=ts, maximal_only=True)
        parties = {record["sender"], record["receiver"]}
        for p in fresh:
            if parties & set(p.members):
                self._pattern_entry(p.to_dict())
        patterns = sorted(self.patterns_for(parties), key=lambda p: PATTERN_PRIORITY.get(p["type"], 9))
        alert = {
            "alert_id": alert_id,
            "status": "open",
            "created_at": record["timestamp"],
            "transaction_id": record["transaction_id"],
            "sender": record["sender"],
            "receiver": record["receiver"],
            "amount": record["amount"],
            "risk_score": record["overall_risk"],
            "overall_risk": record["overall_risk"],
            "risk_percent": record["risk_percent"],
            "action": record["action"],
            "sub_scores": record["sub_scores"],
            "signals": record["signals"],
            "pattern": patterns[0]["type"] if patterns else None,
            "pattern_ids": [p["id"] for p in patterns],
            "verdict": None,
            "_record": record,
        }
        self.alerts[alert_id] = alert
        return alert_id

    def list_alerts(self, status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        with self.lock:
            items = [a for a in self.alerts.values() if status in (None, "all") or a["status"] == status]
            items.sort(key=lambda a: (a["status"] != "open", -a["overall_risk"], a["created_at"]))
            return [self.alert_summary(a) for a in items[:limit]]

    def alert_summary(self, a: Dict[str, Any]) -> Dict[str, Any]:
        summary = {k: v for k, v in a.items() if not k.startswith("_")}
        summary["reasons"] = a["_record"]["reasons"][:3]
        return summary

    def get_case(self, alert_id: str) -> Dict[str, Any]:
        with self.lock:
            alert = self.alerts.get(alert_id)
            if alert is None:
                raise EngineError(f"alert {alert_id} not found", status=404)
            return case_file.build(self, alert)

    # ════════════════════════════════════════════════════════════════════
    # analyst verdicts + propagation (F13)
    # ════════════════════════════════════════════════════════════════════
    def verdict(self, alert_id: str, verdict: str, accounts: Optional[List[str]] = None,
                note: Optional[str] = None) -> Dict[str, Any]:
        if verdict not in ("confirm", "clear"):
            raise EngineError("verdict must be 'confirm' or 'clear'")
        with self.lock:
            alert = self.alerts.get(alert_id)
            if alert is None:
                raise EngineError(f"alert {alert_id} not found", status=404)
            if alert["status"] != "open":
                raise EngineError(f"alert {alert_id} is already {alert['status']}", status=409)
            t0 = time.perf_counter()
            record = alert["_record"]
            self.verified_labels[record["transaction_id"]] = 1 if verdict == "confirm" else 0
            changes: List[Dict[str, Any]] = []
            seeds: List[str] = []
            if verdict == "confirm":
                suspects = case_file.suspects(self, alert)
                seeds = [a for a in (accounts or suspects) if a in self.accounts]
                if not seeds:
                    raise EngineError("no known accounts to mark as fraud")
                changes = self._propagate(seeds, alert_id)
                alert["status"] = "confirmed"
            else:
                alert["status"] = "cleared"
                self.cleared_accounts.update({alert["sender"], alert["receiver"]} - self.confirmed_accounts)
            entry = {
                "alert_id": alert_id,
                "verdict": verdict,
                "at": _iso(self.clock),
                "decided_at_wallclock": datetime.now(timezone.utc).isoformat(),
                "seeds": seeds,
                "note": (note or "")[:500],
                "accounts_changed": len(changes),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            alert["verdict"] = entry
            self.verdicts.append(entry)
            return {"alert_id": alert_id, "verdict": verdict, "status": alert["status"],
                    "seeds": seeds, "risk_changes": changes, "elapsed_ms": entry["elapsed_ms"]}

    def propagation_graph(self) -> nx.Graph:
        """Undirected graph of payments (weight = payment count) plus shared identities."""
        g = nx.Graph()
        for p in self.graph.payments(as_of=self.clock):
            if g.has_edge(p.sender, p.receiver):
                g[p.sender][p.receiver]["weight"] += 1.0
            else:
                g.add_edge(p.sender, p.receiver, weight=1.0)
        for (itype, value), accs in self.identity._by_identity.items():
            visible = self.identity.get_accounts_for_identity(itype, value, as_of=self.clock)
            if 2 <= len(visible) <= 30:
                for i, a in enumerate(visible):
                    for b in visible[i + 1:]:
                        w = g[a][b]["weight"] + 1.0 if g.has_edge(a, b) else 1.0
                        g.add_edge(a, b, weight=w)
        return g

    def _taint(self, sender: str, receiver: str, tx_id: str) -> None:
        """F13, live: money leaving an account with propagated risk carries it on.

        After a confirmation, payments out of confirmed or strongly-linked
        accounts pass decayed risk to their receiver, so mules further down a
        chain are caught even if the analyst confirmed before the chain existed.
        """
        src = self.propagated.get(sender, 0.0)
        if src < TAINT_MIN:
            return
        passed = round(src * TAINT_DECAY, 4)
        if passed > self.propagated.get(receiver, 0.0):
            self.propagated[receiver] = passed
            self.propagation_source[receiver] = f"payment {tx_id} from {sender}"

    def _propagate(self, seeds: List[str], alert_id: str) -> List[Dict[str, Any]]:
        before = {a: self.account_risk(a) for a in self.accounts}
        for a in seeds:
            self.confirmed_accounts.add(a)
            self.cleared_accounts.discard(a)
            self.propagated[a] = 1.0
            self.propagation_source[a] = alert_id
        g = self.propagation_graph()
        present = [a for a in seeds if a in g]
        if present:
            personalization = {a: 1.0 / len(present) for a in present}
            ppr = nx.pagerank(g, alpha=0.85, personalization=personalization, weight="weight")
            # Scale PPR mass relative to an average seed: tight ring neighbours
            # (many payments / shared identities with seeds) approach the cap,
            # an ordinary one-off counterparty of a mule stays low.
            seed_mass = np.mean([ppr[a] for a in present])
            for node, mass in ppr.items():
                if node in present:
                    continue
                risk = float(min(PPR_CAP, PPR_SCALE * mass / seed_mass))
                if risk >= 0.05 and risk > self.propagated.get(node, 0.0):
                    self.propagated[node] = round(risk, 4)
                    self.propagation_source[node] = alert_id
        changes = []
        for a in self.accounts:
            after = self.account_risk(a)
            if after - before.get(a, 0.0) >= 0.01:
                changes.append({"account_id": a, "risk_before": round(float(before.get(a, 0.0)), 4),
                                "risk_after": round(float(after), 4), "seed": a in seeds})
        changes.sort(key=lambda c: (-c["seed"], -c["risk_after"]))
        return changes

    # ════════════════════════════════════════════════════════════════════
    # account views
    # ════════════════════════════════════════════════════════════════════
    def exposure_risk(self, account: str) -> float:
        if self.clock is None:
            return 0.0
        cutoff = self.clock - EXPOSURE_WINDOW
        return max((risk for ts, risk, _ in self.exposure.get(account, ()) if ts >= cutoff), default=0.0)

    def account_risk(self, account: str) -> float:
        if account in self.confirmed_accounts:
            return 1.0
        return round(max(self.exposure_risk(account), self.propagated.get(account, 0.0)), 4)

    def account_status(self, account: str) -> str:
        if account in self.confirmed_accounts:
            return "confirmed_fraud"
        if account in self.cleared_accounts:
            return "cleared"
        return "none"

    def account_profile(self, account: str, as_of: Optional[datetime] = None, full: bool = True) -> Dict[str, Any]:
        with self.lock:
            if account not in self.accounts:
                raise EngineError(f"account {account} not found", status=404)
            as_of = as_of or self.clock
            meta = self.accounts[account]
            identities = []
            for ident in self.identity.get_identities(account, as_of=as_of):
                shared = [a for a in self.identity.get_accounts_for_identity(
                    ident["identity_type"], ident["identity_value"], as_of=as_of) if a != account]
                identities.append({"type": ident["identity_type"], "value": ident["identity_value"],
                                   "shared_with": shared[:30], "shared_count": len(shared)})
            identities.sort(key=lambda i: -i["shared_count"])
            profile = {
                "account_id": account,
                "signup_at": _iso(meta.get("signup_at")),
                "risk": self.account_risk(account),
                "exposure_risk": round(self.exposure_risk(account), 4),
                "propagated_risk": round(self.propagated.get(account, 0.0), 4),
                "propagation_source": self.propagation_source.get(account),
                "status": self.account_status(account),
                "lockstep": self.lockstep.get(account),
                "identities": identities,
                "patterns": [self._pattern_brief(p) for p in self.patterns_for({account})],
            }
            if not full or as_of is None:
                return profile
            profile["flow"] = account_flow_features(self.graph, account, as_of=as_of)
            profile["lifelike"] = account_lifelikeness(self.graph, account, as_of=as_of)
            metrics = account_graph_metrics(self.graph, account, as_of=as_of)
            profile["graph_metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            recent = self.graph.get_account_history(account, as_of=as_of)[-25:]
            profile["recent_payments"] = [self._payment_view(p) for p in reversed(recent)]
            profile["alerts"] = [self.alert_summary(a) for a in self.alerts.values()
                                 if account in (a["sender"], a["receiver"])]
            return profile

    def _payment_view(self, p) -> Dict[str, Any]:
        rec = self.payment_index.get(p.transaction_id or "")
        return {
            "transaction_id": p.transaction_id,
            "timestamp": p.timestamp.isoformat(),
            "sender": p.sender,
            "receiver": p.receiver,
            "amount": round(p.amount, 2),
            "scored": rec is not None,
            "risk": rec["overall_risk"] if rec else None,
            "action": rec["action"] if rec else None,
            "alert_id": rec["alert_id"] if rec else None,
        }

    def list_accounts(self, limit: int = 50, query: Optional[str] = None, min_risk: float = 0.0) -> List[Dict[str, Any]]:
        with self.lock:
            rows = []
            q = (query or "").strip().upper()
            for acc in self.accounts:
                if q and q not in acc.upper():
                    continue
                risk = self.account_risk(acc)
                if risk < min_risk:
                    continue
                rows.append((risk, acc))
            rows.sort(key=lambda t: (-t[0], t[1]))
            out = []
            for risk, acc in rows[:limit]:
                out.append({
                    "account_id": acc,
                    "risk": risk,
                    "status": self.account_status(acc),
                    "propagated_risk": round(self.propagated.get(acc, 0.0), 4),
                    "exposure_risk": round(self.exposure_risk(acc), 4),
                    "in_lockstep": acc in self.lockstep,
                    "pattern_count": len(self.patterns_for({acc})),
                    "alert_count": sum(1 for a in self.alerts.values() if acc in (a["sender"], a["receiver"])),
                    "payments": self.state.s_count.get(acc, 0) + self.state.r_count.get(acc, 0),
                })
            return out

    # ════════════════════════════════════════════════════════════════════
    # graph view (F14)
    # ════════════════════════════════════════════════════════════════════
    def graph_snapshot(self, focus: Optional[str] = None, hops: int = 1, pattern: Optional[str] = None,
                       alert: Optional[str] = None, limit: int = 250, include_identity: bool = True,
                       min_risk: float = 0.0) -> Dict[str, Any]:
        with self.lock:
            highlight: Set[str] = set()
            if alert:
                if alert not in self.alerts:
                    raise EngineError(f"alert {alert} not found", status=404)
                return case_file.build(self, self.alerts[alert])["subgraph"]
            if pattern:
                pat = self.patterns.get(pattern)
                if pat is None:
                    raise EngineError(f"pattern {pattern} not found", status=404)
                highlight = set(pat["members"])
                nodes = set(highlight)
                for m in list(highlight):
                    nodes.update(self.graph.get_neighbors(m, as_of=self.clock)[:8])
            elif focus:
                if focus not in self.accounts:
                    raise EngineError(f"account {focus} not found", status=404)
                highlight = {focus}
                nodes = {focus}
                frontier = {focus}
                for _ in range(max(1, min(hops, 2))):
                    nxt = set()
                    for n in frontier:
                        nxt.update(self.graph.get_neighbors(n, as_of=self.clock))
                    nodes |= nxt
                    frontier = nxt
                    if len(nodes) > limit:
                        break
            else:
                # Live view: what an analyst should look at — risky, confirmed or
                # propagated accounts and active ring members — plus the most
                # recent payments for context.
                nodes = {rec[k] for rec in self.payments[-25:] for k in ("sender", "receiver")}
                nodes |= {a for a in self.accounts if self.account_risk(a) >= 0.3}
                nodes |= {a for a, v in self.propagated.items() if v >= 0.2}
                for p in self.patterns.values():
                    if p["active"] and p["type"] != "shared_device_star":
                        nodes.update(p["members"])
            nodes = {n for n in nodes if self.account_risk(n) >= min_risk or n in highlight}
            if len(nodes) > limit:
                ranked = sorted(nodes, key=lambda n: (n not in highlight, -self.account_risk(n)))
                nodes = set(ranked[:limit])
            return case_file.subgraph_payload(self, nodes, self.clock, highlight=highlight,
                                              include_identity=include_identity)

    # ════════════════════════════════════════════════════════════════════
    # dashboard summaries
    # ════════════════════════════════════════════════════════════════════
    def overview(self) -> Dict[str, Any]:
        with self.lock:
            actions = Counter(p["action"] for p in self.payments)
            status = Counter(a["status"] for a in self.alerts.values())
            lat = sorted(self.latencies)

            def pct(q):
                return round(lat[min(len(lat) - 1, int(q * len(lat)))], 2) if lat else None

            buckets: Dict[str, Counter] = defaultdict(Counter)
            for p in self.payments:
                hour = p["timestamp"][:13] + ":00"
                buckets[hour][p["action"]] += 1
            active = [p for p in self.patterns.values() if p["active"]]
            return {
                "clock": _iso(self.clock),
                "accounts": len(self.accounts),
                "history_payments": self.history_count,
                "scored_payments": len(self.payments),
                "action_counts": dict(actions),
                "alert_counts": {"open": status.get("open", 0), "confirmed": status.get("confirmed", 0),
                                 "cleared": status.get("cleared", 0)},
                "blocked_amount": round(sum(p["amount"] for p in self.payments if p["action"] == "BLOCK"), 2),
                "held_amount": round(sum(p["amount"] for p in self.payments if p["action"] == "HOLD_RECEIVER"), 2),
                "pattern_counts": dict(Counter(p["type"] for p in active)),
                "lockstep_accounts": len(self.lockstep),
                "confirmed_accounts": len(self.confirmed_accounts),
                "propagated_accounts": sum(1 for v in self.propagated.values() if v >= 0.05),
                "latency_ms": {"p50": pct(0.5), "p95": pct(0.95)},
                "model_version": self.models.version,
                "slow_path_runs": self._slow_runs,
                "timeline": [{"hour": h, **dict(c)} for h, c in sorted(buckets.items())],
                "top_accounts": self.list_accounts(limit=8, min_risk=0.3),
            }

    def recent_payments(self, limit: int = 50, flagged_only: bool = False) -> List[Dict[str, Any]]:
        with self.lock:
            rows = [p for p in self.payments if not flagged_only or p["action"] != "ALLOW"]
            return [self.public_payment(p) for p in reversed(rows[-limit:])]

    def list_patterns(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        with self.lock:
            out = []
            for p in self.patterns.values():
                if not p["active"] and not include_inactive:
                    continue
                risks = [self.account_risk(m) for m in p["members"]]
                out.append({
                    **{k: v for k, v in p.items() if k != "subgraph"},
                    "member_count": len(p["members"]),
                    "max_member_risk": round(max(risks, default=0.0), 4),
                    "confirmed_members": sum(1 for m in p["members"] if m in self.confirmed_accounts),
                    "alert_ids": [a["alert_id"] for a in self.alerts.values() if p["id"] in a["pattern_ids"]],
                })
            out.sort(key=lambda p: (-p["max_member_risk"], PATTERN_PRIORITY.get(p["type"], 9), p["id"]))
            return out

    # ════════════════════════════════════════════════════════════════════
    # learning (F19)
    # ════════════════════════════════════════════════════════════════════
    def retrain(self) -> Dict[str, Any]:
        from ringbreaker.learn.retrain import retrain_with_verified

        with self.lock:
            verified = [(self.payment_index[t]["_features"], label)
                        for t, label in self.verified_labels.items() if t in self.payment_index]
            stream_rows = [(p["transaction_id"], p["_features"]) for p in self.payments]
            version = f"1.{len(self.model_history)}.{len(verified)}"
            result = retrain_with_verified(verified, version, stream_rows, current=self.models.booster,
                                           verified_ids=set(self.verified_labels))
            self.models.use_booster(result.pop("booster"), version, result["model_path"])
            entry = {"version": version, "at": _iso(self.clock), "kind": "retrain", **result}
            self.model_history.append(entry)
            return entry

    def learning_state(self) -> Dict[str, Any]:
        with self.lock:
            labels = Counter(self.verified_labels.values())
            evaluation_path = config.MODELS_DIR / "evaluation.json"
            metadata_path = config.MODEL_METADATA_PATH
            return {
                "evaluation": json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None,
                "training": json.loads(metadata_path.read_text()) if metadata_path.exists() else None,
                "model_version": self.models.version,
                "verified": {"confirmed": labels.get(1, 0), "cleared": labels.get(0, 0)},
                "verdicts": list(reversed(self.verdicts)),
                "model_history": list(reversed(self.model_history)),
                "propagated": sorted(
                    ({"account_id": a, "propagated_risk": v, "source": self.propagation_source.get(a),
                      "status": self.account_status(a)} for a, v in self.propagated.items() if v >= 0.05),
                    key=lambda r: -r["propagated_risk"])[:100],
            }
