"""Shared sliding-window UCB on link success probabilities."""

from __future__ import annotations

import math
from collections import deque

from fair_qroute.topology import Topology, _canon


class SlidingWindowUCB:
    """One shared estimator per undirected link. Optimistic init: UCB=1.0 at count 0."""

    def __init__(self, edges: list[tuple[int, int]], window: int = 200, c: float = 1.5) -> None:
        self.window = window
        self.c = c
        self._hist: dict[tuple[int, int], deque[int]] = {
            _canon(*e): deque(maxlen=window) for e in edges
        }
        self._t = 0

    def reset(self, edges: list[tuple[int, int]] | None = None) -> None:
        keys = edges if edges is not None else list(self._hist.keys())
        self._hist = {_canon(*e): deque(maxlen=self.window) for e in keys}
        self._t = 0

    def observe(self, u: int, v: int, success: bool) -> None:
        self._hist[_canon(u, v)].append(1 if success else 0)

    def tick(self) -> None:
        self._t += 1

    def mean(self, u: int, v: int) -> float:
        h = self._hist[_canon(u, v)]
        if not h:
            return 1.0
        return sum(h) / len(h)

    def count(self, u: int, v: int) -> int:
        return len(self._hist[_canon(u, v)])

    def ucb(self, u: int, v: int) -> float:
        h = self._hist[_canon(u, v)]
        if not h:
            return 1.0
        n = len(h)
        mean = sum(h) / n
        t = max(self._t, 1)
        bonus = self.c * math.sqrt(math.log(t + 1) / n)
        return min(1.0, mean + bonus)

    def path_score(self, nodes: list[int]) -> float:
        if len(nodes) < 2:
            return 1.0
        score = 1.0
        for a, b in zip(nodes, nodes[1:]):
            score *= self.ucb(a, b)
        return score

def true_path_score(nodes: list[int], topo: Topology) -> float:
    if len(nodes) < 2:
        return 1.0
    score = 1.0
    for a, b in zip(nodes, nodes[1:]):
        score *= topo.p_of(a, b)
    return score
