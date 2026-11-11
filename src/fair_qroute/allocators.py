"""Exclusive all-or-nothing allocators: greedy, FIFO/RR, cumulative deficit-fair."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from fair_qroute.learning import SlidingWindowUCB, true_path_score
from fair_qroute.network import NetworkModel
from fair_qroute.paths import best_path
from fair_qroute.topology import Topology


class Policy(str, Enum):
    GREEDY = "greedy"
    FIFO = "fifo"
    FAIR = "fair"


@dataclass
class Proposal:
    pair_id: int
    path: list[int]
    score: float


def propose_paths(
    topo: Topology,
    pair_ids: list[int],
    ucb: SlidingWindowUCB | None,
    use_oracle: bool,
    k: int = 3,
) -> list[Proposal]:
    """Every pair proposes exactly one BestPath. No fallback."""
    props: list[Proposal] = []
    for pid in pair_ids:
        src, dst = topo.pairs[pid]
        path = best_path(topo, src, dst, ucb=ucb, k=k, use_true_p=use_oracle)
        score = true_path_score(path, topo) if use_oracle else ucb.path_score(path)  # type: ignore[union-attr]
        props.append(Proposal(pair_id=pid, path=path, score=score))
    return props


def order_proposals(
    props: list[Proposal],
    policy: Policy,
    successes: list[int],
    rr_token: int,
) -> list[Proposal]:
    n = len(successes)
    total = sum(successes)

    if policy is Policy.GREEDY:
        return sorted(props, key=lambda p: (-p.score, p.pair_id))

    if policy is Policy.FAIR:
        def deficit(pid: int) -> float:
            return (total / n) - successes[pid]

        return sorted(props, key=lambda p: (-deficit(p.pair_id), p.pair_id))

    # FIFO / round-robin: rotate by token, ignore path score for ordering
    return sorted(props, key=lambda p: ((p.pair_id - rr_token) % n, p.pair_id))


def allocate_round(
    topo: Topology,
    net: NetworkModel,
    ucb: SlidingWindowUCB,
    policy: Policy,
    active_pairs: list[int],
    successes: list[int],
    rr_token: int,
    rng: np.random.Generator,
    use_oracle: bool = False,
    k: int = 3,
) -> tuple[list[bool], int]:
    """
    Run one exclusive AON round.
    Returns (granted_success per global pair index or False if inactive/lost, next_rr_token).
    """
    n_pairs = len(topo.pairs)
    round_success = [False] * n_pairs
    net.reset_round()
    props = propose_paths(topo, active_pairs, ucb=None if use_oracle else ucb, use_oracle=use_oracle, k=k)
    ordered = order_proposals(props, policy, successes, rr_token)

    for prop in ordered:
        if not net.can_grant(prop.path):
            continue  # loser: 0 this round, no observe, no fallback
        ok = net.grant_and_observe(prop.path, ucb, rng)
        round_success[prop.pair_id] = ok

    next_token = (rr_token + 1) % max(len(active_pairs), 1)
    return round_success, next_token
