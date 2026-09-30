"""Online feature store shared by offline training and the live fast path (F9).

One incremental state object produces the 39 pair-model features and the 12
behavioural (EIF) features for a candidate payment using only payments that were
already applied to it. The offline builder (``features/pair_features.py``) and
the live engine (``engine/engine.py``) both drive this class, so training and
serving compute features with the same code.

Contract, for every payment in timestamp order::

    feats = state.pair_features(p)     # history strictly before p
    behav = state.behaviour_features(p)
    state.update(p)                    # only now does p enter the history

Callers must feed payments in non-decreasing timestamp order; the live engine
rejects out-of-order payments so a feature can never see the future.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional

UNOBSERVED_VALUE = -1.0

PAIR_FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_night",
    "sender_account_age_days",
    "receiver_account_age_days",
    "sender_tx_count_before",
    "sender_unique_receivers_before",
    "sender_avg_amount_before",
    "sender_max_amount_before",
    "sender_avg_time_gap",
    "sender_tx_last_1h",
    "sender_tx_last_24h",
    "sender_tx_last_7d",
    "receiver_tx_count_before",
    "receiver_unique_senders_before",
    "receiver_avg_amount_before",
    "receiver_max_amount_before",
    "receiver_tx_last_1h",
    "receiver_tx_last_24h",
    "receiver_tx_last_7d",
    "pair_tx_count_before",
    "pair_avg_amount_before",
    "time_since_previous_pair_tx",
    "is_first_transaction_between_pair",
    "device_tx_count_before",
    "device_unique_users_before",
    "ip_tx_count_before",
    "ip_unique_users_before",
    "sender_in_degree_before",
    "sender_out_degree_before",
    "receiver_in_degree_before",
    "receiver_out_degree_before",
    "sender_out_in_ratio",
    "receiver_in_out_ratio",
    "sender_time_since_last_incoming",
]

BEHAVIOUR_FEATURE_NAMES = [
    "tx_amount",
    "amount_to_prior_avg",
    "tx_velocity_1h",
    "tx_velocity_24h",
    "tx_velocity_7d",
    "device_reuse_degree",
    "ip_reuse_degree",
    "sender_fan_out",
    "receiver_fan_in",
    "pass_through_ratio",
    "account_activity_span_hours",
    "rapid_drain_speed_seconds",
]

_HOUR = 3600.0
_DAY = 86400.0
_WEEK = 7 * _DAY


@dataclass(frozen=True)
class FeaturePayment:
    """The fields of a payment the feature store needs."""

    sender: str
    receiver: str
    amount: float
    timestamp: datetime
    device: Optional[str] = None
    ip: Optional[str] = None


def _finite(value: float) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def _prune(q: Deque[datetime], now: datetime, window_seconds: float) -> int:
    cutoff = now - timedelta(seconds=window_seconds)
    while q and q[0] < cutoff:
        q.popleft()
    return len(q)


class OnlineFeatureState:
    """Incremental per-account, per-pair and per-identity history."""

    def __init__(self, signups: Optional[Dict[str, datetime]] = None) -> None:
        self.signup: Dict[str, datetime] = dict(signups or {})

        self.s_count: Dict[str, int] = defaultdict(int)
        self.s_total: Dict[str, float] = defaultdict(float)
        self.s_max: Dict[str, float] = defaultdict(float)
        self.s_receivers: Dict[str, set] = defaultdict(set)
        self.s_last: Dict[str, datetime] = {}
        self.s_gap_sum: Dict[str, float] = defaultdict(float)
        self.s_q1h: Dict[str, Deque[datetime]] = defaultdict(deque)
        self.s_q24h: Dict[str, Deque[datetime]] = defaultdict(deque)
        self.s_q7d: Dict[str, Deque[datetime]] = defaultdict(deque)

        self.r_count: Dict[str, int] = defaultdict(int)
        self.r_total: Dict[str, float] = defaultdict(float)
        self.r_max: Dict[str, float] = defaultdict(float)
        self.r_senders: Dict[str, set] = defaultdict(set)
        self.r_last_in: Dict[str, datetime] = {}
        self.r_q1h: Dict[str, Deque[datetime]] = defaultdict(deque)
        self.r_q24h: Dict[str, Deque[datetime]] = defaultdict(deque)
        self.r_q7d: Dict[str, Deque[datetime]] = defaultdict(deque)

        self.pair_count: Dict[tuple, int] = defaultdict(int)
        self.pair_total: Dict[tuple, float] = defaultdict(float)
        self.pair_last: Dict[tuple, datetime] = {}

        self.dev_count: Dict[str, int] = defaultdict(int)
        self.dev_users: Dict[str, set] = defaultdict(set)
        self.ip_count: Dict[str, int] = defaultdict(int)
        self.ip_users: Dict[str, set] = defaultdict(set)

        self.first_seen: Dict[str, datetime] = {}
        self.clock: Optional[datetime] = None

    # ── registration ────────────────────────────────────────────────────
    def register(self, account_id: str, signup_at: Optional[datetime]) -> None:
        if signup_at is not None:
            self.signup[str(account_id)] = signup_at

    def _age_days(self, account: str, t: datetime) -> float:
        signup = self.signup.get(account)
        if signup is None or signup > t:
            # Unknown or not-yet-visible signup: the account is brand new at t.
            return 0.0
        return max(0.0, (t - signup).total_seconds() / _DAY)

    # ── features ────────────────────────────────────────────────────────
    def pair_features(self, p: FeaturePayment) -> Dict[str, float]:
        t, s, r, amt = p.timestamp, p.sender, p.receiver, float(p.amount)
        dev = str(p.device) if p.device else ""
        ip = str(p.ip) if p.ip else ""

        s_cnt = self.s_count.get(s, 0)
        s_avg = self.s_total[s] / s_cnt if s_cnt else 0.0
        s_max = self.s_max[s] if s_cnt else 0.0
        s_gap = self.s_gap_sum[s] / (s_cnt - 1) if s_cnt > 1 else UNOBSERVED_VALUE

        r_cnt = self.r_count.get(r, 0)
        r_avg = self.r_total[r] / r_cnt if r_cnt else 0.0
        r_max = self.r_max[r] if r_cnt else 0.0

        key = (s, r)
        p_cnt = self.pair_count.get(key, 0)
        p_avg = self.pair_total[key] / p_cnt if p_cnt else 0.0
        p_since = (t - self.pair_last[key]).total_seconds() if key in self.pair_last else UNOBSERVED_VALUE

        s_out_deg = self.s_count.get(s, 0)
        s_in_deg = self.r_count.get(s, 0)
        r_out_deg = self.s_count.get(r, 0)
        r_in_deg = self.r_count.get(r, 0)
        s_since_in = (t - self.r_last_in[s]).total_seconds() if s in self.r_last_in else UNOBSERVED_VALUE

        return {
            "amount": amt,
            "log_amount": math.log1p(max(amt, 0.0)),
            "hour": float(t.hour),
            "day_of_week": float(t.weekday()),
            "day_of_month": float(t.day),
            "is_weekend": 1.0 if t.weekday() >= 5 else 0.0,
            "is_night": 1.0 if t.hour <= 5 else 0.0,
            "sender_account_age_days": self._age_days(s, t),
            "receiver_account_age_days": self._age_days(r, t),
            "sender_tx_count_before": float(s_cnt),
            "sender_unique_receivers_before": float(len(self.s_receivers.get(s, ()))),
            "sender_avg_amount_before": s_avg,
            "sender_max_amount_before": s_max,
            "sender_avg_time_gap": s_gap,
            "sender_tx_last_1h": float(_prune(self.s_q1h[s], t, _HOUR)),
            "sender_tx_last_24h": float(_prune(self.s_q24h[s], t, _DAY)),
            "sender_tx_last_7d": float(_prune(self.s_q7d[s], t, _WEEK)),
            "receiver_tx_count_before": float(r_cnt),
            "receiver_unique_senders_before": float(len(self.r_senders.get(r, ()))),
            "receiver_avg_amount_before": r_avg,
            "receiver_max_amount_before": r_max,
            "receiver_tx_last_1h": float(_prune(self.r_q1h[r], t, _HOUR)),
            "receiver_tx_last_24h": float(_prune(self.r_q24h[r], t, _DAY)),
            "receiver_tx_last_7d": float(_prune(self.r_q7d[r], t, _WEEK)),
            "pair_tx_count_before": float(p_cnt),
            "pair_avg_amount_before": p_avg,
            "time_since_previous_pair_tx": p_since,
            "is_first_transaction_between_pair": 1.0 if p_cnt == 0 else 0.0,
            "device_tx_count_before": float(self.dev_count.get(dev, 0)),
            "device_unique_users_before": float(len(self.dev_users.get(dev, ()))),
            "ip_tx_count_before": float(self.ip_count.get(ip, 0)),
            "ip_unique_users_before": float(len(self.ip_users.get(ip, ()))),
            "sender_in_degree_before": float(s_in_deg),
            "sender_out_degree_before": float(s_out_deg),
            "receiver_in_degree_before": float(r_in_deg),
            "receiver_out_degree_before": float(r_out_deg),
            "sender_out_in_ratio": s_out_deg / (s_in_deg + 1.0),
            "receiver_in_out_ratio": r_in_deg / (r_out_deg + 1.0),
            "sender_time_since_last_incoming": s_since_in,
        }

    def behaviour_features(self, p: FeaturePayment) -> Dict[str, float]:
        """Per-payment sender behaviour vector for the EIF anomaly model (F18)."""
        t, s, r, amt = p.timestamp, p.sender, p.receiver, float(p.amount)
        dev = str(p.device) if p.device else ""
        ip = str(p.ip) if p.ip else ""
        s_cnt = self.s_count.get(s, 0)
        prior_avg = self.s_total[s] / s_cnt if s_cnt else amt
        total_in = self.r_total.get(s, 0.0)
        total_out = self.s_total.get(s, 0.0) + amt
        first = self.first_seen.get(s, t)
        drain = (t - self.r_last_in[s]).total_seconds() if s in self.r_last_in else _DAY
        return {
            "tx_amount": amt,
            "amount_to_prior_avg": amt / (prior_avg + 1e-2),
            "tx_velocity_1h": float(_prune(self.s_q1h[s], t, _HOUR)),
            "tx_velocity_24h": float(_prune(self.s_q24h[s], t, _DAY)),
            "tx_velocity_7d": float(_prune(self.s_q7d[s], t, _WEEK)),
            "device_reuse_degree": float(len(self.dev_users.get(dev, ()))),
            "ip_reuse_degree": float(len(self.ip_users.get(ip, ()))),
            "sender_fan_out": len(self.s_receivers.get(s, ())) / (s_cnt + 1.0),
            "receiver_fan_in": len(self.r_senders.get(r, ())) / (self.r_count.get(r, 0) + 1.0),
            "pass_through_ratio": min(total_out / (total_in + 10.0), 10.0),
            "account_activity_span_hours": max((t - first).total_seconds() / _HOUR, 0.01),
            "rapid_drain_speed_seconds": min(max(drain, 0.0), _DAY),
        }

    # ── state update ────────────────────────────────────────────────────
    def update(self, p: FeaturePayment) -> None:
        t, s, r, amt = p.timestamp, p.sender, p.receiver, float(p.amount)
        dev = str(p.device) if p.device else ""
        ip = str(p.ip) if p.ip else ""

        if s in self.s_last:
            self.s_gap_sum[s] += (t - self.s_last[s]).total_seconds()
        self.s_last[s] = t
        self.s_count[s] += 1
        self.s_total[s] += amt
        self.s_max[s] = max(self.s_max[s], amt)
        self.s_receivers[s].add(r)
        self.s_q1h[s].append(t)
        self.s_q24h[s].append(t)
        self.s_q7d[s].append(t)

        self.r_count[r] += 1
        self.r_total[r] += amt
        self.r_max[r] = max(self.r_max[r], amt)
        self.r_senders[r].add(s)
        self.r_last_in[r] = t
        self.r_q1h[r].append(t)
        self.r_q24h[r].append(t)
        self.r_q7d[r].append(t)

        key = (s, r)
        self.pair_count[key] += 1
        self.pair_total[key] += amt
        self.pair_last[key] = t

        self.dev_count[dev] += 1
        self.dev_users[dev].add(s)
        self.ip_count[ip] += 1
        self.ip_users[ip].add(s)

        self.first_seen.setdefault(s, t)
        self.first_seen.setdefault(r, t)
        if self.clock is None or t > self.clock:
            self.clock = t


def sanitize(features: Dict[str, float]) -> Dict[str, float]:
    """Replace NaN/inf with 0 so a malformed value can never reach a model."""
    return {k: _finite(v) for k, v in features.items()}
