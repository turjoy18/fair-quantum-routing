"""k-shortest-by-hops candidates and deterministic BestPath (no fallback)."""

from __future__ import annotations

from collections import deque

from fair_qroute.learning import SlidingWindowUCB, true_path_score
from fair_qroute.topology import Topology, _canon


def edge_list_key(nodes: list[int]) -> tuple[tuple[int, int], ...]:
    return tuple(_canon(a, b) for a, b in zip(nodes, nodes[1:]))


def k_shortest_hop_paths(
    topo: Topology,
    src: int,
    dst: int,
    k: int = 3,
) -> list[list[int]]:
    """BFS enumeration of simple paths ordered by hop length, keep first k distinct."""
    adj = topo.neighbors()
    found: list[list[int]] = []
    q: deque[list[int]] = deque([[src]])
    while q and len(found) < k:
        path = q.popleft()
        u = path[-1]
        if u == dst:
            found.append(path)
            continue
        for v in adj[u]:
            if v in path:
                continue
            q.append(path + [v])
    return found


def best_path(
    topo: Topology,
    src: int,
    dst: int,
    ucb: SlidingWindowUCB | None,
    k: int = 3,
    use_true_p: bool = False,
) -> list[int]:
    """Pick max product score among k shortest-hop paths; ties: shorter, then lex edges."""
    cands = k_shortest_hop_paths(topo, src, dst, k=k)
    if not cands:
        raise ValueError(f"no path {src}->{dst}")

    def key(path: list[int]) -> tuple:
        score = true_path_score(path, topo) if use_true_p else ucb.path_score(path)  # type: ignore[union-attr]
        hops = len(path) - 1
        return (-score, hops, edge_list_key(path))

    return min(cands, key=key)
