"""EWMA (RiskMetrics-style) volatility: recent days count more than old days.

Weights decay geometrically: the day k steps back gets weight proportional to lam**k.
With lam = 0.94 (the RiskMetrics daily default) the effective memory is about 16 days,
so the covariance estimate reacts quickly when volatility jumps.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .models import RiskEstimate, _validate, simulate_returns, tail_metrics


def ewma_cov(returns, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance matrix (zero-mean, as in RiskMetrics).

    Rows of `returns` are ordered oldest -> newest; the newest row gets the largest weight.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    w = lam ** np.arange(n - 1, -1, -1)
    w = w / w.sum()
    return np.atleast_2d((r * w[:, None]).T @ r)


def ewma_parametric_var(returns, weights, alpha: float = 0.99, lam: float = 0.94) -> RiskEstimate:
    """Normal VaR/CVaR using the EWMA covariance (closed form)."""
    r, w = _validate(returns, weights, alpha)
    sigma = float(np.sqrt(w @ ewma_cov(r, lam) @ w))
    z = norm.ppf(alpha)
    return RiskEstimate("EWMA (Normal)", alpha, sigma * z, sigma * norm.pdf(z) / (1.0 - alpha))


def ewma_monte_carlo_var(returns, weights, alpha: float = 0.99, lam: float = 0.94,
                         n_sims: int = 100_000, dist: str = "t", df: float = 5.0,
                         antithetic: bool = True, seed: int | None = None) -> RiskEstimate:
    """Monte Carlo VaR/CVaR with EWMA covariance and (by default) Student-t shocks."""
    r, w = _validate(returns, weights, alpha)
    cov = ewma_cov(r, lam)
    rng = np.random.default_rng(seed)
    sims = simulate_returns(np.zeros(r.shape[1]), cov, n_sims, dist, df, antithetic, rng)
    var, cvar = tail_metrics(-(sims @ w), alpha)
    name = "EWMA (Normal)" if dist == "normal" else "EWMA (Student-t)"
    return RiskEstimate(name, alpha, var, cvar)
