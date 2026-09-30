from datetime import datetime, timedelta

from ringbreaker.features.lifelike import account_lifelikeness
from ringbreaker.graphs.build import TransactionGraph


def test_new_account_finite_score():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    g.register_account("N", signup_at=t)
    feats = account_lifelikeness(g, "N", as_of=t)
    assert 0.0 <= feats["lifelikeness_score"] <= 1.0
    assert feats["thin_identity"] == 1
    assert feats["history_depth"] == 0


def test_established_account_higher_than_thin():
    g = TransactionGraph()
    start = datetime(2023, 1, 1, 8, 0, 0)
    g.register_account("RICH", signup_at=start)
    g.register_account("THIN", signup_at=start + timedelta(days=360))
    as_of = datetime(2024, 1, 1, 20, 0, 0)
    counterparties = [f"P{i}" for i in range(12)]
    for i, other in enumerate(counterparties):
        hour = 8 + (i * 3) % 14
        g.add_payment(
            "RICH",
            other,
            10,
            datetime(2023, 6, 1, hour, 0, 0) + timedelta(days=i * 10),
            transaction_id=f"r{i}",
        )
    g.add_payment("THIN", "P0", 5, as_of - timedelta(days=1), transaction_id="t1")
    rich = account_lifelikeness(g, "RICH", as_of=as_of)
    thin = account_lifelikeness(g, "THIN", as_of=as_of)
    assert rich["lifelikeness_score"] > thin["lifelikeness_score"]
    assert rich["counterparty_count"] == 12
    assert thin["thin_identity"] == 1


def test_no_nan_with_limited_history():
    g = TransactionGraph()
    t = datetime(2024, 1, 1)
    g.add_payment("A", "B", 1, t, transaction_id="1")
    feats = account_lifelikeness(g, "A", as_of=t)
    assert feats["lifelikeness_score"] == feats["lifelikeness_score"]
    assert feats["time_of_day_spread"] == 0.0
