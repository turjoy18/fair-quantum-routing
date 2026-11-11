"""Topology / p-map tests."""

from fair_qroute.topology import (
    CENTRE_EDGE,
    PARALLEL_CROSSINGS,
    P_CENTRE,
    P_OTHER,
    P_PARALLEL,
    Topology,
    _canon,
    drifted_p_map,
    grid_edges_3x3,
    pinned_p_map,
)


def test_grid_edge_count() -> None:
    assert len(grid_edges_3x3()) == 12


def test_pinned_p_map() -> None:
    p = pinned_p_map()
    assert p[_canon(*CENTRE_EDGE)] == P_CENTRE
    for e in PARALLEL_CROSSINGS:
        assert p[_canon(*e)] == P_PARALLEL
    assert sum(1 for v in p.values() if v == P_OTHER) == 12 - 1 - len(PARALLEL_CROSSINGS)


def test_drift_flips_centre() -> None:
    d = drifted_p_map()
    assert d[_canon(*CENTRE_EDGE)] == 0.4
    for e in PARALLEL_CROSSINGS:
        assert d[_canon(*e)] == 0.9


def test_default_pairs() -> None:
    t = Topology()
    assert len(t.pairs) == 3
