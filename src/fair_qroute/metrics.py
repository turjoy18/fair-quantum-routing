"""Throughput, Jain fairness, and same-policy regret helpers."""

from __future__ import annotations

import numpy as np


def jain_index(successes: list[int] | np.ndarray) -> float:
    x = np.asarray(successes, dtype=float)
    if x.size == 0 or np.sum(x) <= 0:
        return 1.0
    return float((np.sum(x) ** 2) / (x.size * np.sum(x ** 2)))


def summarize(successes: list[int]) -> dict[str, float]:
    arr = np.asarray(successes, dtype=float)
    return {
        "total": float(arr.sum()),
        "worst": float(arr.min()) if arr.size else 0.0,
        "mean_pair": float(arr.mean()) if arr.size else 0.0,
        "jain": jain_index(arr),
        **{f"pair_{i}": float(v) for i, v in enumerate(successes)},
    }


def windowed_regret(
    learner_success_trace: list[int],
    oracle_success_trace: list[int],
    window: int,
) -> list[float]:
    """Per-window sum(oracle) - sum(learner) on total e2e successes per round."""
    assert len(learner_success_trace) == len(oracle_success_trace)
    out: list[float] = []
    t = len(learner_success_trace)
    for start in range(0, t, window):
        end = min(start + window, t)
        out.append(
            float(sum(oracle_success_trace[start:end]) - sum(learner_success_trace[start:end]))
        )
    return out
