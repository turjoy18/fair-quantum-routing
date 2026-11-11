# fair-quantum-routing

Fair multi-pair entanglement routing under **exclusive** link contention — a mentorship proof-of-work for John C.S. Lui (CUHK CSE), adjacent to BeQuP (*Learning Best Paths in Quantum Networks*, INFOCOM 2025 / [arXiv:2506.12462](https://arxiv.org/abs/2506.12462)).

Question: **when several BeQuP-style pairs compete for the same links, who starves?** — not a BeQuP reproduction.

## Headline

![Total vs worst-pair](results/headline_fairness.png)

Asymmetric 3-pair bottleneck (see below). Also `results/fairness_table.csv`, `results/same_policy_diagnostic.csv`, `results/drift.csv`.

## What this claims

**Can claim**

- Under exclusive all-or-nothing multi-pair contention with a **shared sliding-window UCB** and **no fallback**, greedy-UCB **starves** the worst pair. Totals across policies are essentially tied (~0 cost to fairness) — greedy does **not** win total throughput.
- On an **asymmetric** path-quality map (one pair’s centre path has a weaker hop), cumulative **deficit-fair** lifts worst-pair vs greedy and **beats FIFO** on worst-pair: FIFO still rotates equally onto the weak path; deficit keeps serving until successes catch up.
- On the **symmetric** map, fair ≈ FIFO (expected) — rotation alone fixes starvation; that suite is reported for honesty.
- Low contention (1 pair): policies tie.
- Sliding-window UCB after mid-run drift: see windowed same-policy diagnostic in `drift.csv` (not cumulative regret-as-bound).

**Cannot claim**

- BeQuP / NetSquid reproduction or hardware realism.
- That \(p_\ell\) is fidelity — it is **link entanglement success probability per slot**; path succeeds iff all links succeed.
- Purification, swap fidelity degradation, or inter-domain QBGP / QPSEL (mentored growth).
- That deficit-first is optimal max-min.
- That the true-\(p\) same-policy column is a **throughput upper bound** or that “regret” is non-negative — optimistic UCB can create parallel grants under capped init; see diagnostic CSV note.
- That losers update link estimates, or that pairs retry a detour the same round (**no fallback**).

## Locked design choices

| Choice | Setting |
|--------|---------|
| Contention | Exclusive slots; all-or-nothing path grant |
| Learning | **One shared** sliding-window UCB per link; optimistic init 1.0 |
| Proposal | Single UCB BestPath among **k=3** shortest-hop candidates; **no fallback** |
| Policies (plot) | greedy (path-UCB order), FIFO (RR token), fair (cumulative deficit) |
| True-\(p\) column | Same policy, true \(p\); **diagnostic only**, not an oracle upper bound |
| Topology | Fixed 3×3 grid; 3 pairs `(0,5),(3,2),(6,5)` |

### Pinned \(p\) map (engineered bottleneck)

| Edge class | Edges | Symmetric \(p_\ell\) | Asymmetric \(p_\ell\) |
|------------|-------|----------------------|----------------------|
| Centre | `(3,4)` | **0.9** | **0.9** |
| Parallel crossings | `(0,1)`, `(6,7)` | **0.5** | **0.5** |
| Weak hop (pair 2) | `(6,3)` | 0.8 | **0.45** |
| Other | remaining | **0.8** | **0.8** |

Headline uses **asymmetric**. Symmetric is in the table to show fair ≈ FIFO when path products match. Drift (mid-run): centre → 0.4, parallels → 0.9.

### Deficit / FIFO

- Deficit at round start: \((\sum_j s_j)/n - s_i\) (cumulative, not EWMA).
- FIFO pairs also propose UCB BestPath; only grant order is round-robin.

## Quick start

```bash
cd fair-quantum-routing
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
pytest -q
python -m fair_qroute.experiment --out results
```

No NetSquid / Qiskit required. Runs use `T=1000` rounds × 10 seeds by default.

## Growth (not this PoW)

- NetSquid backend behind `NetworkModel`
- Separate \(p_\ell\) vs fidelity \(F_\ell\) + swap degradation
- Max-min fair allocation; QPSEL / QBGP inter-domain path selection (ToN 2026)
