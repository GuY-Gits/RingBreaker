"""F7: interpretable lifelikeness score.

Flags accounts that look too clean and too thin. This is a behavioural formula,
not a supervised fraud model. Higher ``lifelikeness_score`` (0–1) means more
like an established person: older account, deeper history, more counterparties,
and payments spread across hours of day.

Components (each clipped to [0, 1], then averaged):

- account_age: signup -> as_of, saturating at 90 days. Missing signup uses first
  observed payment, or 0 for unseen accounts.
- history_depth: payment count saturating at 20.
- counterparty_variety: unique counterparties / max(payments, 1).
- time_of_day_spread: sample standard deviation of payment hour-of-day,
  divided by 6 (a full-day-ish spread) and clipped.

``thin_identity`` is 1 when history is very short.
``too_clean`` is 1 when there is some activity but almost no hour spread and
low counterparty variety.

Never returns NaN or inf.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from ringbreaker.graphs.build import TransactionGraph, parse_timestamp

AGE_SATURATION_DAYS = 90.0
DEPTH_SATURATION = 20.0
HOUR_SPREAD_NORM = 6.0


def account_lifelikeness(
    graph: TransactionGraph,
    account_id: str,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
) -> dict[str, Any]:
    account_id = str(account_id)
    as_of_ts = parse_timestamp(as_of)
    history = graph.get_account_history(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    signup = graph.get_signup_at(account_id, as_of=as_of_ts)
    if signup is None and history:
        signup = min(p.timestamp for p in history)
    if signup is None:
        age_days = 0.0
    else:
        age_days = max(0.0, (as_of_ts - signup).total_seconds() / 86400.0)

    counterparties: set[str] = set()
    hours: list[float] = []
    for payment in history:
        other = payment.receiver if payment.sender == account_id else payment.sender
        counterparties.add(other)
        hours.append(payment.timestamp.hour + payment.timestamp.minute / 60.0)

    n = len(history)
    variety = (len(counterparties) / n) if n else 0.0
    hour_std = _std(hours)
    age_comp = min(1.0, age_days / AGE_SATURATION_DAYS)
    depth_comp = min(1.0, n / DEPTH_SATURATION)
    variety_comp = min(1.0, variety)
    spread_comp = min(1.0, hour_std / HOUR_SPREAD_NORM) if n >= 2 else 0.0
    score = (age_comp + depth_comp + variety_comp + spread_comp) / 4.0
    thin = int(n < 3)
    too_clean = int(n >= 3 and spread_comp < 0.15 and variety_comp < 0.4)
    return {
        "account_id": account_id,
        "lifelikeness_score": _finite(score),
        "account_age_days": _finite(age_days),
        "history_depth": n,
        "counterparty_count": len(counterparties),
        "counterparty_variety": _finite(variety),
        "time_of_day_spread": _finite(hour_std),
        "thin_identity": thin,
        "too_clean": too_clean,
        "components": {
            "account_age": _finite(age_comp),
            "history_depth": _finite(depth_comp),
            "counterparty_variety": _finite(variety_comp),
            "time_of_day_spread": _finite(spread_comp),
        },
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(var, 0.0))


def _finite(value: float) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return float(value)
