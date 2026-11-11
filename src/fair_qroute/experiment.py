"""Sweeps: 1-pair, symmetric 3-pair, asymmetric 3-pair (headline), diagnostic, drift."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fair_qroute.allocators import Policy
from fair_qroute.metrics import windowed_regret
from fair_qroute.simulate import SimResult, run_sim

T_DEFAULT = 1000
N_SEEDS = 10
UCB_WINDOW = 200
DRIFT_AT = 500
REGRET_WINDOW = 100


def _mean_summary(results: list[SimResult]) -> dict[str, float]:
    keys = results[0].summary.keys()
    return {k: float(np.mean([r.summary[k] for r in results])) for k in keys}


def run_policy_seeds(
    policy: Policy,
    *,
    n_pairs: int,
    T: int,
    seeds: range,
    use_oracle: bool = False,
    drift_at: int | None = None,
    asymmetric: bool = False,
) -> list[SimResult]:
    return [
        run_sim(
            policy,
            n_pairs=n_pairs,
            T=T,
            seed=s,
            use_oracle=use_oracle,
            drift_at=drift_at,
            ucb_window=UCB_WINDOW,
            asymmetric=asymmetric,
        )
        for s in seeds
    ]


def suite_low_contention(T: int, seeds: range) -> list[dict]:
    rows: list[dict] = []
    for pol in Policy:
        res = run_policy_seeds(pol, n_pairs=1, T=T, seeds=seeds)
        m = _mean_summary(res)
        rows.append(
            {"suite": "low_1pair", "policy": pol.value, "oracle": False, **m, "n_seeds": len(seeds), "T": T}
        )
    return rows


def suite_high_contention(T: int, seeds: range, *, asymmetric: bool) -> list[dict]:
    label = "high_3pair_asymmetric" if asymmetric else "high_3pair_symmetric"
    rows: list[dict] = []
    for pol in Policy:
        res = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds, asymmetric=asymmetric)
        m = _mean_summary(res)
        rows.append(
            {
                "suite": label,
                "policy": pol.value,
                "oracle": False,
                "asymmetric": asymmetric,
                **m,
                "n_seeds": len(seeds),
                "T": T,
            }
        )
    return rows


def suite_same_policy_diagnostic(T: int, seeds: range) -> list[dict]:
    """2x2 {greedy, fair} x {UCB, true-p}: diagnostic only — not a throughput upper bound.

    True-p greedy tie-breaks by pair id under equal path products; UCB optimistic
    init can create parallel grants. Delta = truep_total - ucb_total may be negative.
    """
    rows: list[dict] = []
    for pol in (Policy.GREEDY, Policy.FAIR):
        learn = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds, use_oracle=False, asymmetric=True)
        truep = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds, use_oracle=True, asymmetric=True)
        lm, tm = _mean_summary(learn), _mean_summary(truep)
        rows.append(
            {
                "suite": "same_policy_diagnostic",
                "policy": pol.value,
                "mode": "ucb",
                "note": "not_an_upper_bound",
                **lm,
                "n_seeds": len(seeds),
                "T": T,
            }
        )
        rows.append(
            {
                "suite": "same_policy_diagnostic",
                "policy": pol.value,
                "mode": "true_p",
                "note": "not_an_upper_bound",
                **tm,
                "n_seeds": len(seeds),
                "T": T,
            }
        )
        deltas = [t.summary["total"] - l.summary["total"] for l, t in zip(learn, truep)]
        # Rounds with >=2 e2e successes (parallel completions), mean over seeds
        ucb_par = float(np.mean([sum(1 for x in r.total_per_round if x >= 2) for r in learn]))
        truep_par = float(np.mean([sum(1 for x in r.total_per_round if x >= 2) for r in truep]))
        if pol is Policy.FAIR:
            note = (
                "fair_truep_always_centre_max_1_grant; "
                f"ucb_parallel_e2e_rounds≈{ucb_par:.0f}/T vs truep≈{truep_par:.0f}; "
                "negative_delta_is_optimistic_UCB_parallelism_not_oracle_beating"
            )
        else:
            note = (
                "greedy_truep_tiebreak_by_pair_id_under_equal_products; "
                f"ucb_parallel_e2e_rounds≈{ucb_par:.0f}/T vs truep≈{truep_par:.0f}; "
                "negative_delta_not_an_upper_bound"
            )
        rows.append(
            {
                "suite": "same_policy_diagnostic_delta",
                "policy": pol.value,
                "mean_truep_minus_ucb_total": float(np.mean(deltas)),
                "std_truep_minus_ucb_total": float(np.std(deltas)),
                "mean_ucb_parallel_e2e_rounds": ucb_par,
                "mean_truep_parallel_e2e_rounds": truep_par,
                "note": note,
                "n_seeds": len(seeds),
                "T": T,
            }
        )
    return rows


def suite_drift(T: int, seeds: range, drift_at: int = DRIFT_AT) -> list[dict]:
    rows: list[dict] = []
    for pol in (Policy.GREEDY, Policy.FAIR):
        learn = run_policy_seeds(
            pol, n_pairs=3, T=T, seeds=seeds, use_oracle=False, drift_at=drift_at, asymmetric=True
        )
        truep = run_policy_seeds(
            pol, n_pairs=3, T=T, seeds=seeds, use_oracle=True, drift_at=drift_at, asymmetric=True
        )
        win_regs = []
        for l, o in zip(learn, truep):
            wr = windowed_regret(l.total_per_round, o.total_per_round, REGRET_WINDOW)
            start_w = drift_at // REGRET_WINDOW
            win_regs.append(float(np.mean(wr[start_w:])) if wr[start_w:] else 0.0)
        rows.append(
            {
                "suite": "drift",
                "policy": pol.value,
                "drift_at": drift_at,
                "post_drift_mean_window_delta": float(np.mean(win_regs)),
                "note": "windowed truep_minus_ucb; not cumulative regret upper bound",
                "learn_total_mean": _mean_summary(learn)["total"],
                "truep_total_mean": _mean_summary(truep)["total"],
                "learn_worst_mean": _mean_summary(learn)["worst"],
                "truep_worst_mean": _mean_summary(truep)["worst"],
                "n_seeds": len(seeds),
                "T": T,
                "ucb": "sliding_window",
                "ucb_window": UCB_WINDOW,
                "asymmetric": True,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_headline(rows: list[dict], out: Path, suite: str = "high_3pair_asymmetric") -> None:
    """Scatter: total vs worst-pair for UCB greedy / FIFO / fair (asymmetric headline)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    markers = {"greedy": "o", "fifo": "s", "fair": "^"}
    colors = {"greedy": "#c0392b", "fifo": "#7f8c8d", "fair": "#2980b9"}
    xs: list[float] = []
    for row in rows:
        if row.get("suite") != suite:
            continue
        pol = row["policy"]
        xs.append(row["total"])
        ax.scatter(
            row["total"],
            row["worst"],
            s=120,
            marker=markers.get(pol, "o"),
            color=colors.get(pol, "k"),
            label=f"{pol} (Jain={row['jain']:.2f})",
            zorder=3,
        )
        ax.annotate(pol, (row["total"], row["worst"]), textcoords="offset points", xytext=(8, 4), fontsize=9)
    if xs:
        pad = max(5.0, 0.02 * (max(xs) - min(xs) + 1))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_xlabel("Total successful e2e entanglements (mean over seeds)")
    ax.set_ylabel("Worst-pair successes (mean over seeds)")
    ax.set_title("Asymmetric 3-pair bottleneck: total vs worst-pair (UCB)")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_all(out_dir: Path, T: int = T_DEFAULT, n_seeds: int = N_SEEDS) -> None:
    seeds = range(n_seeds)
    print("Low contention (1 pair)...")
    low = suite_low_contention(T, seeds)
    print("Symmetric 3-pair (fair ≈ FIFO expected)...")
    sym = suite_high_contention(T, seeds, asymmetric=False)
    print("Asymmetric 3-pair (headline)...")
    asym = suite_high_contention(T, seeds, asymmetric=True)
    print("Same-policy diagnostic (not an upper bound)...")
    diag = suite_same_policy_diagnostic(T, seeds)
    print("Drift (sliding-window UCB)...")
    drift = suite_drift(T, seeds)

    write_csv(out_dir / "fairness_table.csv", low + sym + asym)
    write_csv(out_dir / "same_policy_diagnostic.csv", diag)
    write_csv(out_dir / "drift.csv", drift)
    # keep old filename as pointer-free removal; do not write oracle_regret.csv
    old = out_dir / "oracle_regret.csv"
    if old.exists():
        old.unlink()
    plot_headline(asym, out_dir / "headline_fairness.png")
    print(f"wrote {out_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fair quantum routing experiments")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=T_DEFAULT)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    args = p.parse_args(argv)
    run_all(args.out, T=args.T, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
