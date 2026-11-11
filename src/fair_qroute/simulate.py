"""Multi-round simulation with shared sliding-window UCB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fair_qroute.allocators import Policy, allocate_round
from fair_qroute.learning import SlidingWindowUCB
from fair_qroute.metrics import summarize
from fair_qroute.network import GridBernoulliNetwork
from fair_qroute.topology import Topology, drifted_p_map, pinned_p_map


@dataclass
class SimResult:
    policy: str
    oracle: bool
    successes: list[int]
    total_per_round: list[int]
    summary: dict[str, float]
    seed: int


def run_sim(
    policy: Policy,
    *,
    n_pairs: int = 3,
    T: int = 1000,
    seed: int = 0,
    use_oracle: bool = False,
    drift_at: int | None = None,
    ucb_window: int = 200,
    k: int = 3,
) -> SimResult:
    topo = Topology()
    topo.set_p(pinned_p_map())
    active = list(range(min(n_pairs, len(topo.pairs))))
    net = GridBernoulliNetwork(topo)
    ucb = SlidingWindowUCB(topo.edges, window=ucb_window)
    rng = np.random.default_rng(seed)
    successes = [0] * len(topo.pairs)
    rr_token = 0
    total_per_round: list[int] = []

    for t in range(T):
        if drift_at is not None and t == drift_at:
            topo.set_p(drifted_p_map())
        round_ok, rr_token = allocate_round(
            topo,
            net,
            ucb,
            policy,
            active,
            successes,
            rr_token,
            rng,
            use_oracle=use_oracle,
            k=k,
        )
        ucb.tick()
        got = 0
        for pid in active:
            if round_ok[pid]:
                successes[pid] += 1
                got += 1
        total_per_round.append(got)

    active_success = [successes[i] for i in active]
    return SimResult(
        policy=policy.value,
        oracle=use_oracle,
        successes=active_success,
        total_per_round=total_per_round,
        summary=summarize(active_success),
        seed=seed,
    )
