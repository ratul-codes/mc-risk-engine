"""Matplotlib charts saved as PNG files (no display needed)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_distribution(losses, estimates, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(losses * 100, bins=80, color="#9ecae1", edgecolor="white")
    for est, color in zip(estimates, ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]):
        ax.axvline(est.var * 100, color=color, lw=2, label=f"{est.method}: {est.var:.2%}")
    ax.set_xlabel("One-day portfolio loss (%)")
    ax.set_ylabel("Number of days")
    ax.set_title(f"Loss distribution with {estimates[0].alpha:.0%} VaR by method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_backtests(results, path):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.2 * len(results)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        x = np.arange(res.n_obs)
        ax.plot(x, res.losses * 100, color="#bdbdbd", lw=0.7, label="Realised loss")
        ax.plot(x, res.var_forecasts * 100, color="#1f77b4", lw=1.2, label="VaR forecast")
        ax.scatter(x[res.hits], res.losses[res.hits] * 100, color="#d62728", s=14,
                   zorder=3, label="Exception")
        ax.set_title(f"{res.method}: {res.n_exceptions} exceptions "
                     f"(expected {res.expected_exceptions:.1f}) - {res.verdict()}")
        ax.set_ylabel("Loss (%)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Backtest day")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_convergence(sizes, se_plain, se_anti, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(sizes, se_plain, "o-", label="Plain Monte Carlo")
    ax.loglog(sizes, se_anti, "s-", label="Antithetic variates")
    ref = se_plain[0] * np.sqrt(sizes[0] / np.asarray(sizes))
    ax.loglog(sizes, ref, "k--", lw=1, label=r"$1/\sqrt{N}$ reference")
    ax.set_xlabel("Number of simulations N")
    ax.set_ylabel("Std. error of VaR estimate")
    ax.set_title("Monte Carlo VaR convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
