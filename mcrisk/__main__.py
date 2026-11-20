"""Run the full demo:  python -m mcrisk [--csv prices.csv]"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from . import plots
from .backtest import rolling_backtest
from .data import load_prices_csv, synthetic_returns
from .ewma import ewma_monte_carlo_var, ewma_parametric_var
from .models import component_var, historical_var, monte_carlo_var, parametric_var


def build_estimators(n_sims: int, seed: int):
    return {
        "Historical": historical_var,
        "Parametric (Normal)": parametric_var,
        "Monte Carlo (Normal)": partial(monte_carlo_var, n_sims=n_sims, dist="normal", seed=seed),
        "Monte Carlo (Student-t)": partial(monte_carlo_var, n_sims=n_sims, dist="t", df=5, seed=seed),
        "EWMA (Normal)": ewma_parametric_var,
        "EWMA (Student-t)": partial(ewma_monte_carlo_var, n_sims=n_sims, dist="t", df=5, seed=seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo portfolio risk engine demo")
    ap.add_argument("--csv", help="CSV of prices (date column + one column per asset); default = synthetic data")
    ap.add_argument("--alpha", type=float, default=0.99, help="VaR confidence level")
    ap.add_argument("--window", type=int, default=250, help="rolling estimation window (days)")
    ap.add_argument("--sims", type=int, default=5000, help="Monte Carlo paths per day in the backtest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs", help="folder for charts and results.txt")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    if args.csv:
        returns, names = load_prices_csv(args.csv)
        src = f"CSV: {args.csv}"
    else:
        returns, names = synthetic_returns(seed=7)
        src = "synthetic data (GARCH volatility clustering + Student-t shocks)"
    n_days, n_assets = returns.shape
    w = np.full(n_assets, 1.0 / n_assets)

    emit(f"Data: {src}")
    emit(f"{n_days} days, {n_assets} assets, equal weights, alpha = {args.alpha:.0%}, 1-day horizon")

    # 1. Full-sample estimates
    emit("\n=== 1. Full-sample VaR / CVaR (100,000 Monte Carlo paths) ===")
    emit(f"{'Method':<30}{'VaR':>8}{'CVaR':>8}")
    full = []
    for name, est in build_estimators(100_000, args.seed).items():
        e = est(returns, w, args.alpha)
        full.append(e)
        emit(f"{e.method:<30}{e.var:>8.2%}{e.cvar:>8.2%}")
    emit("(EWMA rows use mostly the most recent ~16 days, so they describe risk 'now')")
    plots.plot_loss_distribution(-(returns @ w), full[:4], out / "loss_distribution.png")

    # 2. Component VaR
    emit("\n=== 2. Component VaR (share of parametric VaR by asset) ===")
    comp = component_var(returns, w, args.alpha)
    for nm, c in zip(names, comp / comp.sum()):
        emit(f"{nm:<14}{c:>8.1%}")

    # 3. Rolling backtest
    emit(f"\n=== 3. Rolling backtest (window={args.window}, {args.sims} paths/day) ===")
    emit(f"{'Method':<30}{'Obs':>6}{'Exc.':>6}{'Exp.':>7}{'Kupiec p':>10}{'Christ. p':>11}  Verdict")
    results = []
    for name, est in build_estimators(args.sims, args.seed).items():
        res = rolling_backtest(returns, w, est, window=args.window, alpha=args.alpha)
        results.append(res)
        emit(f"{res.method:<30}{res.n_obs:>6}{res.n_exceptions:>6}{res.expected_exceptions:>7.1f}"
             f"{res.kupiec_p:>10.3f}{res.christoffersen_p:>11.3f}  {res.verdict()}")
    emit("(PASS = both p-values >= 0.05; Kupiec tests the exception count, Christoffersen tests clustering)")
    plots.plot_backtests([results[1], results[3], results[5]], out / "backtest.png")

    # 4. Convergence
    emit("\n=== 4. Monte Carlo convergence (std. error of VaR over 100 seeds) ===")
    sizes = [1_000, 10_000, 100_000]
    se = {False: [], True: []}
    for anti in (False, True):
        for n in sizes:
            vals = [monte_carlo_var(returns, w, args.alpha, n_sims=n, antithetic=anti, seed=s).var
                    for s in range(100)]
            se[anti].append(float(np.std(vals, ddof=1)))
    emit(f"{'N paths':>10}{'Plain SE':>12}{'Antithetic SE':>15}")
    for i, n in enumerate(sizes):
        emit(f"{n:>10,}{se[False][i]:>12.5%}{se[True][i]:>15.5%}")
    plots.plot_convergence(sizes, se[False], se[True], out / "convergence.png")

    (out / "results.txt").write_text("\n".join(lines) + "\n")
    emit(f"\nSaved charts and results.txt to {out}/")


if __name__ == "__main__":
    main()
