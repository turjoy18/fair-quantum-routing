"""BestPath / shared UCB tests."""

import numpy as np

from fair_qroute.learning import SlidingWindowUCB
from fair_qroute.paths import best_path, k_shortest_hop_paths
from fair_qroute.topology import Topology


def test_k_shortest_at_most_k() -> None:
    topo = Topology()
    paths = k_shortest_hop_paths(topo, 0, 5, k=3)
    assert 1 <= len(paths) <= 3
    assert all(p[0] == 0 and p[-1] == 5 for p in paths)


def test_optimistic_ucb() -> None:
    topo = Topology()
    ucb = SlidingWindowUCB(topo.edges, window=50)
    e = topo.edges[0]
    assert ucb.ucb(*e) == 1.0
    assert ucb.count(*e) == 0


def test_best_path_deterministic() -> None:
    topo = Topology()
    ucb = SlidingWindowUCB(topo.edges)
    a = best_path(topo, 0, 5, ucb)
    b = best_path(topo, 0, 5, ucb)
    assert a == b


def test_shared_observe_updates() -> None:
    topo = Topology()
    ucb = SlidingWindowUCB(topo.edges, window=50)
    ucb.observe(3, 4, True)
    ucb.tick()
    assert ucb.count(3, 4) == 1
    assert ucb.mean(3, 4) == 1.0
