#!/usr/bin/env python3
"""End-to-end demo of Person 2 graphs, features, patterns, and lockstep."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from ringbreaker import (
    IdentityGraph,
    TransactionGraph,
    detect_all_patterns,
    detect_lockstep,
    pair_features,
    receiver_features,
    sender_features,
)
from ringbreaker.features.flow import account_flow_features
from ringbreaker.features.lifelike import account_lifelikeness
from ringbreaker.features.social import pair_social_features
from ringbreaker.patterns.chain import detect_pass_through_chains
from ringbreaker.patterns.fan_in import detect_fan_in_collectors
from ringbreaker.patterns.loop import detect_closed_loops
from ringbreaker.patterns.star import detect_shared_device_stars


def _print(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    tx = TransactionGraph()
    ident = IdentityGraph()
    t0 = datetime(2024, 6, 1, 12, 0, 0)

    tx.register_account("A", signup_at=t0 - timedelta(days=120))
    tx.register_account("B", signup_at=t0 - timedelta(days=110))
    tx.add_payment("A", "B", 50, t0, transaction_id="n1")
    tx.add_payment("B", "A", 40, t0 + timedelta(hours=2), transaction_id="n2")

    tx.register_account("M", signup_at=t0 - timedelta(days=2))
    for i, sender in enumerate(["A", "B", "C", "D"], start=1):
        tx.register_account(sender, signup_at=t0 - timedelta(days=30))
        tx.add_payment(sender, "M", 100, t0 + timedelta(hours=2 + i), transaction_id=f"fan{i}")

    chain_t = t0 + timedelta(days=1)
    for acc in ("X", "Y", "Z", "Q"):
        tx.register_account(acc, signup_at=t0 - timedelta(days=10))
    tx.add_payment("X", "Y", 500, chain_t, transaction_id="c1")
    tx.add_payment("Y", "Z", 490, chain_t + timedelta(minutes=2), transaction_id="c2")
    tx.add_payment("Z", "Q", 480, chain_t + timedelta(minutes=4), transaction_id="c3")

    loop_t = t0 + timedelta(days=2)
    for acc in ("L1", "L2", "L3"):
        tx.register_account(acc, signup_at=t0 - timedelta(days=20))
    tx.add_payment("L1", "L2", 20, loop_t, transaction_id="l1")
    tx.add_payment("L2", "L3", 20, loop_t + timedelta(minutes=1), transaction_id="l2")
    tx.add_payment("L3", "L1", 20, loop_t + timedelta(minutes=2), transaction_id="l3")

    for acc in ("R1", "R2", "R3"):
        ident.add_account_identities(acc, device="device_1", observed_at=t0)

    created = t0 - timedelta(days=20)
    for acc in ("S1", "S2", "S3"):
        tx.register_account(acc, signup_at=created)
        for week in range(3):
            tx.add_payment(
                acc,
                "BANK",
                5,
                created + timedelta(days=7 * week, hours=3),
                transaction_id=f"{acc}-w{week}",
            )

    as_of = t0 + timedelta(days=3)
    _print("transaction snapshot (truncated edges)", {"edge_count": len(tx.snapshot(as_of=as_of)["edges"])})
    _print("identity snapshot", ident.snapshot(as_of=as_of))
    _print("flow A", account_flow_features(tx, "A", as_of=as_of))
    _print("flow Y (chain intermediary)", account_flow_features(tx, "Y", as_of=as_of))
    _print("social A->B", pair_social_features(tx, "A", "B", as_of=as_of))
    _print("lifelikeness A", account_lifelikeness(tx, "A", as_of=as_of))
    _print("person1 sender features A", sender_features(tx, "A", as_of=as_of))
    _print("person1 receiver features M", receiver_features(tx, "M", as_of=as_of))
    _print("person1 pair features A->M", pair_features(tx, "A", "M", as_of=as_of))
    _print("shared-device stars", [p.to_dict() for p in detect_shared_device_stars(ident, as_of=as_of)])
    _print(
        "fan-in",
        [p.to_dict() for p in detect_fan_in_collectors(tx, as_of=as_of, min_senders=4)],
    )
    _print("chains (full X-Y-Z-Q only)", [
        p.to_dict()
        for p in detect_pass_through_chains(tx, as_of=as_of)
        if p.members == ["X", "Y", "Z", "Q"]
    ])
    _print("closed loops", [p.to_dict() for p in detect_closed_loops(tx, as_of=as_of)])
    _print("lockstep clusters", detect_lockstep(tx, as_of=as_of, eps=1.2, min_samples=2))
    _print(
        "detect_all_patterns names",
        [p.pattern_name for p in detect_all_patterns(tx, ident, as_of=as_of)],
    )


if __name__ == "__main__":
    main()
