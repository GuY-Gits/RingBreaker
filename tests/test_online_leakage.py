from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ringbreaker.api.main import app, get_transaction_graph, reset_state


def test_online_scoring_temporal_non_leakage():
    reset_state()
    client = TestClient(app)
    tx_graph = get_transaction_graph()

    # Pre-condition: sender has 0 outgoing payments
    assert len(tx_graph.outgoing_payments("acc_sender_leak")) == 0

    # First transaction from A to B at t=100
    res1 = client.post(
        "/score",
        json={
            "sender": "acc_sender_leak",
            "receiver": "acc_receiver_leak",
            "amount": 500.0,
            "timestamp": "2024-01-01T10:00:00Z",
        },
    )
    assert res1.status_code == 200
    data1 = res1.json()

    # Candidate transaction was scored BEFORE being added to graph.
    # Now in tx_graph, after scoring, the payment is in tx_graph:
    assert "acc_sender_leak" in tx_graph.account_ids()
    assert len(tx_graph.outgoing_payments("acc_sender_leak")) == 1

    # Second transaction at t=101 from A to B:
    res2 = client.post(
        "/score",
        json={
            "sender": "acc_sender_leak",
            "receiver": "acc_receiver_leak",
            "amount": 250.0,
            "timestamp": "2024-01-01T10:01:00Z",
        },
    )
    assert res2.status_code == 200
    # Now in graph there are 2 outgoing transactions
    assert len(tx_graph.outgoing_payments("acc_sender_leak")) == 2
