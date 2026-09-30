from datetime import datetime, timedelta

from ringbreaker.graphs.build import IdentityGraph, TransactionGraph


def ts(*parts):
    return datetime(*parts)


def build_fixture():
    """Synthetic graph covering normal, fan-in, chain, loop, shared device."""
    g = TransactionGraph()
    ident = IdentityGraph()
    t0 = datetime(2024, 6, 1, 12, 0, 0)

    for acc, signup in {
        "A": t0 - timedelta(days=100),
        "B": t0 - timedelta(days=90),
        "M": t0 - timedelta(days=2),
        "X": t0 - timedelta(days=10),
        "Y": t0 - timedelta(days=10),
        "Z": t0 - timedelta(days=10),
        "Q": t0 - timedelta(days=10),
        "L1": t0 - timedelta(days=20),
        "L2": t0 - timedelta(days=20),
        "L3": t0 - timedelta(days=20),
        "R1": t0 - timedelta(days=1),
        "R2": t0 - timedelta(days=1),
        "R3": t0 - timedelta(days=1),
        "C": t0 - timedelta(days=5),
        "D": t0 - timedelta(days=5),
    }.items():
        g.register_account(acc, signup_at=signup)

    g.add_payment("A", "B", 50, t0, transaction_id="n1")
    g.add_payment("B", "A", 40, t0 + timedelta(hours=2), transaction_id="n2")

    for sender, txn_id, hours in [
        ("A", "f1", 3),
        ("B", "f2", 4),
        ("C", "f3", 5),
        ("D", "f4", 6),
    ]:
        g.add_payment(sender, "M", 100, t0 + timedelta(hours=hours), transaction_id=txn_id)

    chain_t = t0 + timedelta(days=1)
    g.add_payment("X", "Y", 500, chain_t, transaction_id="c1")
    g.add_payment("Y", "Z", 490, chain_t + timedelta(minutes=2), transaction_id="c2")
    g.add_payment("Z", "Q", 480, chain_t + timedelta(minutes=4), transaction_id="c3")

    loop_t = t0 + timedelta(days=2)
    g.add_payment("L1", "L2", 20, loop_t, transaction_id="l1")
    g.add_payment("L2", "L3", 20, loop_t + timedelta(minutes=1), transaction_id="l2")
    g.add_payment("L3", "L1", 20, loop_t + timedelta(minutes=2), transaction_id="l3")

    for acc in ("R1", "R2", "R3"):
        ident.add_account_identities(acc, device="device_1", observed_at=t0)

    return g, ident, t0
