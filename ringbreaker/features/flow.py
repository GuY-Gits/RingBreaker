"""F5: dwell time and pass-through ratio from timestamped in/out payments.

Definitions (all as of timestamp T, excluding an optional candidate tx):

- incoming_count / outgoing_count: number of payments into / out of the account.
- incoming_amount / outgoing_amount: summed amounts.
- pass_through_ratio: outgoing_amount / incoming_amount when incoming_amount > 0,
  else 0.0. Capped at 1.0. This is a volume proxy, not proven money provenance —
  the simulator is not assumed to tag which outflow came from which inflow.
- mean_dwell_seconds: FIFO amount-matching of inflows to later outflows. Each
  outflow unit is matched to the oldest unmatched inflow unit. Dwell for a match
  is (out_timestamp - in_timestamp) in seconds, averaged with amount weights.
  Unmatched outflows (no prior inflow) are ignored. Returns 0.0 if nothing matched.
- median_dwell_seconds: amount-weighted median of the same matched dwells.

Edge cases: brand-new accounts, in-only, out-only, and zero amounts all return
finite numbers (never NaN/inf).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ringbreaker.graphs.build import TransactionGraph


def _finite(value: float) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return float(value)


def account_flow_features(
    graph: TransactionGraph,
    account_id: str,
    as_of: datetime | str,
    exclude_transaction_id: str | None = None,
) -> dict[str, Any]:
    account_id = str(account_id)
    incoming = graph.incoming_payments(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    outgoing = graph.outgoing_payments(
        account_id, as_of=as_of, exclude_transaction_id=exclude_transaction_id
    )
    in_amt = sum(p.amount for p in incoming)
    out_amt = sum(p.amount for p in outgoing)
    if in_amt > 0:
        ratio = min(1.0, out_amt / in_amt)
    else:
        ratio = 0.0
    dwells = _fifo_dwells(incoming, outgoing)
    mean_dwell = 0.0
    median_dwell = 0.0
    matched_amount = 0.0
    if dwells:
        matched_amount = sum(amount for _, amount in dwells)
        if matched_amount > 0:
            mean_dwell = sum(seconds * amount for seconds, amount in dwells) / matched_amount
            median_dwell = _weighted_median(dwells)
    return {
        "account_id": account_id,
        "incoming_count": len(incoming),
        "outgoing_count": len(outgoing),
        "incoming_amount": _finite(in_amt),
        "outgoing_amount": _finite(out_amt),
        "pass_through_ratio": _finite(ratio),
        "mean_dwell_seconds": _finite(mean_dwell),
        "median_dwell_seconds": _finite(median_dwell),
        "matched_pass_through_amount": _finite(matched_amount),
    }


def _fifo_dwells(incoming, outgoing) -> list[tuple[float, float]]:
    remaining = [[p.timestamp, p.amount] for p in incoming if p.amount > 0]
    dwells: list[tuple[float, float]] = []
    in_idx = 0
    for payment in outgoing:
        need = payment.amount
        if need <= 0:
            continue
        while need > 0 and in_idx < len(remaining):
            in_ts, available = remaining[in_idx]
            if available <= 0:
                in_idx += 1
                continue
            if in_ts > payment.timestamp:
                break
            take = min(available, need)
            seconds = (payment.timestamp - in_ts).total_seconds()
            if seconds < 0:
                seconds = 0.0
            dwells.append((seconds, take))
            remaining[in_idx][1] = available - take
            need -= take
            if remaining[in_idx][1] <= 0:
                in_idx += 1
    return dwells


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    ordered = sorted(pairs, key=lambda item: item[0])
    total = sum(amount for _, amount in ordered)
    if total <= 0:
        return 0.0
    halfway = total / 2.0
    running = 0.0
    for seconds, amount in ordered:
        running += amount
        if running >= halfway:
            return seconds
    return ordered[-1][0]
