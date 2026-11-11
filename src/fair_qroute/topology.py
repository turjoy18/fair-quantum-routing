"""3x3 grid topology, pinned p map, SD pairs, and drift."""

from __future__ import annotations

from dataclasses import dataclass, field

# Node layout:
# 0—1—2
# |  |  |
# 3—4—5
# |  |  |
# 6—7—8

CENTRE_EDGE = (3, 4)
PARALLEL_CROSSINGS = ((0, 1), (6, 7))
# Pair 2 (6→5) unique first hop on the centre path; lowered in asymmetric map only.
WEAK_PAIR_EDGE = (6, 3)
P_CENTRE = 0.9
P_PARALLEL = 0.5
P_OTHER = 0.8
P_WEAK_HOP = 0.45

# Left column → rightish targets; all cross cut {0,3,6} | rest via centre or parallels.
DEFAULT_PAIRS: tuple[tuple[int, int], ...] = ((0, 5), (3, 2), (6, 5))


def _canon(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def grid_edges_3x3() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for r in range(3):
        for c in range(3):
            u = r * 3 + c
            if c + 1 < 3:
                edges.append(_canon(u, r * 3 + c + 1))
            if r + 1 < 3:
                edges.append(_canon(u, (r + 1) * 3 + c))
    return sorted(set(edges))


def pinned_p_map(*, asymmetric: bool = False) -> dict[tuple[int, int], float]:
    """Engineered bottleneck: centre high, parallel crossings low, rest mid.

    asymmetric=True lowers WEAK_PAIR_EDGE so pair 2's centre path has a worse
    product — FIFO wastes centre slots on it; deficit-fair still equalizes successes.
    """
    p: dict[tuple[int, int], float] = {}
    centre = _canon(*CENTRE_EDGE)
    parallels = {_canon(*e) for e in PARALLEL_CROSSINGS}
    weak = _canon(*WEAK_PAIR_EDGE)
    for e in grid_edges_3x3():
        if e == centre:
            p[e] = P_CENTRE
        elif e in parallels:
            p[e] = P_PARALLEL
        elif asymmetric and e == weak:
            p[e] = P_WEAK_HOP
        else:
            p[e] = P_OTHER
    return p


def drifted_p_map(base: dict[tuple[int, int], float] | None = None) -> dict[tuple[int, int], float]:
    """Mid-run flip: centre drops, parallel crossings become preferred."""
    p = dict(base or pinned_p_map())
    p[_canon(*CENTRE_EDGE)] = 0.4
    for e in PARALLEL_CROSSINGS:
        p[_canon(*e)] = 0.9
    return p


@dataclass
class Topology:
    n_nodes: int = 9
    edges: list[tuple[int, int]] = field(default_factory=grid_edges_3x3)
    p: dict[tuple[int, int], float] = field(default_factory=pinned_p_map)
    pairs: tuple[tuple[int, int], ...] = DEFAULT_PAIRS

    def neighbors(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_nodes)}
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in adj:
            adj[i].sort()
        return adj

    def set_p(self, p: dict[tuple[int, int], float]) -> None:
        self.p = {_canon(*k): float(v) for k, v in p.items()}

    def p_of(self, u: int, v: int) -> float:
        return self.p[_canon(u, v)]
