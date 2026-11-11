"""NetworkModel: exclusive link reservation + Bernoulli observations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fair_qroute.learning import SlidingWindowUCB
from fair_qroute.topology import Topology, _canon


class NetworkModel(ABC):
    """Growth seam: v1 grid Bernoulli; future NetSquid backend keeps allocators unchanged."""

    @abstractmethod
    def reset_round(self) -> None:
        ...

    @abstractmethod
    def can_grant(self, path: list[int]) -> bool:
        ...

    @abstractmethod
    def grant_and_observe(
        self,
        path: list[int],
        ucb: SlidingWindowUCB,
        rng: np.random.Generator,
    ) -> bool:
        """Reserve all links, sample Bernoulli per link, update shared UCB. Return e2e success."""
        ...


class GridBernoulliNetwork(NetworkModel):
    def __init__(self, topo: Topology) -> None:
        self.topo = topo
        self._busy: set[tuple[int, int]] = set()

    def reset_round(self) -> None:
        self._busy.clear()

    def can_grant(self, path: list[int]) -> bool:
        if len(path) < 2:
            return False
        for a, b in zip(path, path[1:]):
            if _canon(a, b) in self._busy:
                return False
        return True

    def grant_and_observe(
        self,
        path: list[int],
        ucb: SlidingWindowUCB,
        rng: np.random.Generator,
    ) -> bool:
        if not self.can_grant(path):
            raise RuntimeError("grant on busy path")
        links = [_canon(a, b) for a, b in zip(path, path[1:])]
        for e in links:
            self._busy.add(e)
        e2e = True
        for u, v in links:
            ok = bool(rng.random() < self.topo.p_of(u, v))
            ucb.observe(u, v, ok)
            if not ok:
                e2e = False
        return e2e
