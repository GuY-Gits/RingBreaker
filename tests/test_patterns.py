from datetime import timedelta

from ringbreaker.patterns.chain import detect_pass_through_chains
from ringbreaker.patterns.fan_in import detect_fan_in_collectors
from ringbreaker.patterns.loop import detect_closed_loops
from ringbreaker.patterns.star import detect_shared_device_stars

from tests.conftest import build_fixture


def _names(results):
    return {r.pattern_name for r in results}


def test_shared_device_star():
    _, ident, t0 = build_fixture()
    stars = detect_shared_device_stars(ident, as_of=t0, min_accounts=3)
    assert stars
    star = stars[0]
    payload = star.to_dict()
    assert payload["pattern_name"] == "shared_device_star"
    assert set(payload["members"]) == {"R1", "R2", "R3"}
    assert payload["roles"]["identity_value"] == "device_1"
    assert "nodes" in payload["subgraph"] and "edges" in payload["subgraph"]


def test_pass_through_chain():
    g, _, t0 = build_fixture()
    chains = detect_pass_through_chains(g, as_of=t0 + timedelta(days=2))
    members = [tuple(c.members) for c in chains]
    assert any(m == ("X", "Y", "Z", "Q") or set(["X", "Y", "Z", "Q"]).issubset(m) for m in members)
    long = [c for c in chains if set(c.members) >= {"X", "Y", "Z", "Q"}]
    assert long
    assert long[0].roles["source"] == "X"
    assert long[0].roles["destination"] == "Q"
    assert long[0].evidence["transactions"]


def test_fan_in_collector():
    g, _, t0 = build_fixture()
    fans = detect_fan_in_collectors(g, as_of=t0 + timedelta(days=1), min_senders=4)
    assert fans
    fan = next(f for f in fans if f.roles["collector"] == "M")
    assert set(fan.roles["senders"]) >= {"A", "B", "C", "D"}
    assert fan.evidence["sender_count"] >= 4


def test_closed_loop():
    g, _, t0 = build_fixture()
    loops = detect_closed_loops(g, as_of=t0 + timedelta(days=3))
    assert loops
    found = False
    for loop in loops:
        if set(loop.members) == {"L1", "L2", "L3"}:
            found = True
            assert loop.pattern_name == "closed_loop"
            assert loop.evidence["cycle_length"] == 3
    assert found


def test_pattern_output_schema():
    g, ident, t0 = build_fixture()
    as_of = t0 + timedelta(days=3)
    samples = [
        detect_shared_device_stars(ident, as_of=as_of)[0].to_dict(),
        detect_pass_through_chains(g, as_of=as_of)[0].to_dict(),
        detect_fan_in_collectors(g, as_of=as_of)[0].to_dict(),
        detect_closed_loops(g, as_of=as_of)[0].to_dict(),
    ]
    for payload in samples:
        assert set(payload) == {"pattern_name", "members", "roles", "subgraph", "evidence"}
        assert isinstance(payload["members"], list)
        assert isinstance(payload["roles"], dict)
        assert "nodes" in payload["subgraph"]
        assert "edges" in payload["subgraph"]
