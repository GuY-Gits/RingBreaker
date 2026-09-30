from datetime import datetime, timedelta

from ringbreaker.features.lockstep import detect_lockstep
from ringbreaker.graphs.build import TransactionGraph


def test_lockstep_clusters_sleeper_batch():
    g = TransactionGraph()
    created = datetime(2024, 5, 1, 9, 0, 0)
    as_of = datetime(2024, 6, 15, 9, 0, 0)
    for acc in ("S1", "S2", "S3"):
        g.register_account(acc, signup_at=created)
        for week in range(4):
            g.add_payment(
                acc,
                "BANK",
                5,
                created + timedelta(days=7 * week, hours=3),
                transaction_id=f"{acc}-{week}",
            )
    g.register_account("OLD", signup_at=datetime(2022, 1, 1))
    g.add_payment("OLD", "BANK", 200, datetime(2024, 6, 1, 18, 0, 0), transaction_id="old1")
    g.add_payment("OLD", "X", 15, datetime(2024, 6, 10, 22, 30, 0), transaction_id="old2")
    g.register_account("RAND", signup_at=datetime(2023, 8, 20, 14, 0, 0))
    g.add_payment("RAND", "Y", 80, datetime(2024, 4, 2, 11, 0, 0), transaction_id="r1")
    g.add_payment("Y", "RAND", 12, datetime(2024, 5, 20, 1, 0, 0), transaction_id="r2")

    clusters = detect_lockstep(g, as_of=as_of, eps=1.2, min_samples=2)
    assert clusters
    sleeper = None
    for cluster in clusters:
        if set(cluster["members"]) >= {"S1", "S2", "S3"}:
            sleeper = cluster
            break
    assert sleeper is not None
    assert sleeper["cluster_size"] >= 3
    assert 0.0 < sleeper["lockstep_score"] <= 1.0
    assert "member_features" in sleeper
    assert "OLD" not in sleeper["members"]


def test_lockstep_respects_as_of():
    g = TransactionGraph()
    created = datetime(2024, 5, 1, 9, 0, 0)
    t_score = datetime(2024, 5, 2, 9, 0, 0)
    for acc in ("S1", "S2"):
        g.register_account(acc, signup_at=created)
    before = detect_lockstep(g, as_of=t_score, eps=1.2, min_samples=2)
    g.add_payment("S1", "Z", 1, t_score + timedelta(days=1), transaction_id="fut")
    after = detect_lockstep(g, as_of=t_score, eps=1.2, min_samples=2)
    assert before == after
