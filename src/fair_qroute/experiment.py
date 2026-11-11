"""ID-style sweeps: 1-pair, 3-pair contention, drift; headline plot + tables."""

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
        )
        for s in seeds
    ]


def suite_low_contention(T: int, seeds: range) -> list[dict]:
    rows: list[dict] = []
    for pol in Policy:
        res = run_policy_seeds(pol, n_pairs=1, T=T, seeds=seeds)
        m = _mean_summary(res)
        rows.append({"suite": "low_1pair", "policy": pol.value, "oracle": False, **m, "n_seeds": len(seeds), "T": T})
    return rows


def suite_high_contention(T: int, seeds: range) -> list[dict]:
    rows: list[dict] = []
    for pol in Policy:
        res = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds)
        m = _mean_summary(res)
        rows.append({"suite": "high_3pair", "policy": pol.value, "oracle": False, **m, "n_seeds": len(seeds), "T": T})
    return rows


def suite_oracle_table(T: int, seeds: range) -> list[dict]:
    """2x2: {greedy, fair} x {UCB, oracle}. Same-policy regret only."""
    rows: list[dict] = []
    for pol in (Policy.GREEDY, Policy.FAIR):
        learn = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds, use_oracle=False)
        oracle = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds, use_oracle=True)
        lm, om = _mean_summary(learn), _mean_summary(oracle)
        rows.append({"suite": "oracle_2x2", "policy": pol.value, "mode": "ucb", **lm, "n_seeds": len(seeds), "T": T})
        rows.append({"suite": "oracle_2x2", "policy": pol.value, "mode": "oracle", **om, "n_seeds": len(seeds), "T": T})
        # same-policy total regret (mean over seeds of oracle_total - learn_total)
        regrets = [o.summary["total"] - l.summary["total"] for l, o in zip(learn, oracle)]
        rows.append(
            {
                "suite": "same_policy_regret",
                "policy": pol.value,
                "mean_regret_total": float(np.mean(regrets)),
                "std_regret_total": float(np.std(regrets)),
                "n_seeds": len(seeds),
                "T": T,
            }
        )
    return rows


def suite_drift(T: int, seeds: range, drift_at: int = DRIFT_AT) -> list[dict]:
    rows: list[dict] = []
    for pol in (Policy.GREEDY, Policy.FAIR):
        learn = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds, use_oracle=False, drift_at=drift_at)
        oracle = run_policy_seeds(pol, n_pairs=3, T=T, seeds=seeds, use_oracle=True, drift_at=drift_at)
        # windowed regret on total-per-round, post-drift windows
        win_regs = []
        for l, o in zip(learn, oracle):
            wr = windowed_regret(l.total_per_round, o.total_per_round, REGRET_WINDOW)
            # windows after drift
            start_w = drift_at // REGRET_WINDOW
            win_regs.append(float(np.mean(wr[start_w:])) if wr[start_w:] else 0.0)
        rows.append(
            {
                "suite": "drift",
                "policy": pol.value,
                "drift_at": drift_at,
                "post_drift_mean_window_regret": float(np.mean(win_regs)),
                "learn_total_mean": _mean_summary(learn)["total"],
                "oracle_total_mean": _mean_summary(oracle)["total"],
                "learn_worst_mean": _mean_summary(learn)["worst"],
                "oracle_worst_mean": _mean_summary(oracle)["worst"],
                "n_seeds": len(seeds),
                "T": T,
                "ucb": "sliding_window",
                "ucb_window": UCB_WINDOW,
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


def plot_headline(high_rows: list[dict], out: Path) -> None:
    """Scatter: total vs worst-pair for UCB greedy / FIFO / fair."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    markers = {"greedy": "o", "fifo": "s", "fair": "^"}
    colors = {"greedy": "#c0392b", "fifo": "#7f8c8d", "fair": "#2980b9"}
    for row in high_rows:
        if row.get("suite") != "high_3pair":
            continue
        pol = row["policy"]
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
    ax.set_xlabel("Total successful e2e entanglements (mean over seeds)")
    ax.set_ylabel("Worst-pair successes (mean over seeds)")
    ax.set_title("3-pair bottleneck: total vs worst-pair (UCB policies)")
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
    print("High contention (3 pairs)...")
    high = suite_high_contention(T, seeds)
    print("Oracle 2x2 + same-policy regret...")
    oracle = suite_oracle_table(T, seeds)
    print("Drift (sliding-window UCB)...")
    drift = suite_drift(T, seeds)

    write_csv(out_dir / "fairness_table.csv", low + high)
    write_csv(out_dir / "oracle_regret.csv", oracle)
    write_csv(out_dir / "drift.csv", drift)
    plot_headline(high, out_dir / "headline_fairness.png")
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
