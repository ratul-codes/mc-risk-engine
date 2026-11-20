"""Robustness study: repeat the rolling backtest on many independent synthetic datasets.

A single backtest is one random draw with only ~22 expected exceptions, so its p-values
are noisy. This study reports how often each method passes the Kupiec and Christoffersen
tests across many datasets.

Run:  python -m mcrisk.study [--datasets 45] [--sims 2000]
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from .backtest import rolling_backtest
from .data import synthetic_returns
from .ewma import ewma_monte_carlo_var, ewma_parametric_var
from .models import historical_var, monte_carlo_var, parametric_var


def study_methods(sims: int) -> dict:
    return {
        "Historical": historical_var,
        "Parametric (Normal)": parametric_var,
        "MC Student-t": partial(monte_carlo_var, n_sims=sims, dist="t", df=5, seed=1),
        "EWMA Normal (0.94)": ewma_parametric_var,
        "EWMA Student-t (0.94)": partial(ewma_monte_carlo_var, lam=0.94, n_sims=sims, dist="t", df=5, seed=1),
        "EWMA Student-t (0.97)": partial(ewma_monte_carlo_var, lam=0.97, n_sims=sims, dist="t", df=5, seed=1),
    }


def run_study(seeds, n_days: int = 2500, sims: int = 2000, alpha: float = 0.99,
              window: int = 250, significance: float = 0.05) -> list[dict]:
    methods = study_methods(sims)
    stats = {m: {"exc": [], "kp": [], "cp": []} for m in methods}
    for seed in seeds:
        r, _ = synthetic_returns(n_days=n_days, seed=seed)
        w = np.full(r.shape[1], 1.0 / r.shape[1])
        for name, est in methods.items():
            res = rolling_backtest(r, w, est, window=window, alpha=alpha)
            stats[name]["exc"].append(res.n_exceptions)
            stats[name]["kp"].append(res.kupiec_p)
            stats[name]["cp"].append(res.christoffersen_p)
    rows = []
    for name, d in stats.items():
        kp, cp = np.array(d["kp"]), np.array(d["cp"])
        rows.append({
            "method": name,
            "avg_exceptions": float(np.mean(d["exc"])),
            "kupiec_pass": float((kp >= significance).mean()),
            "christoffersen_pass": float((cp >= significance).mean()),
            "both_pass": float(((kp >= significance) & (cp >= significance)).mean()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-dataset VaR backtest robustness study")
    ap.add_argument("--datasets", type=int, default=45, help="number of synthetic datasets")
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--sims", type=int, default=2000, help="Monte Carlo paths per forecast")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    seeds = range(args.first_seed, args.first_seed + args.datasets)
    rows = run_study(seeds, sims=args.sims)
    n_forecasts = 2500 - 250
    lines = [
        f"{args.datasets} synthetic datasets, {n_forecasts} forecasts each, "
        f"{n_forecasts * 0.01:.1f} exceptions expected at 99%, 5% significance",
        f"{'Method':<24}{'Avg exc.':>9}{'Kupiec pass':>13}{'Christ. pass':>14}{'Both pass':>11}",
    ]
    for r in rows:
        lines.append(f"{r['method']:<24}{r['avg_exceptions']:>9.1f}{r['kupiec_pass']:>13.0%}"
                     f"{r['christoffersen_pass']:>14.0%}{r['both_pass']:>11.0%}")
    print("\n".join(lines))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {out / 'robustness.txt'}")


if __name__ == "__main__":
    main()
