"""VaR / CVaR estimators.

Conventions
-----------
* Inputs are simple returns, shape (T, n_assets), and weights that sum to 1.
* VaR and CVaR are positive loss fractions of portfolio value
  (0.025 means "a 2.5% loss"), one-day horizon.
* alpha is the confidence level, e.g. 0.99 means VaR is the 99th percentile of losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class RiskEstimate:
    method: str
    alpha: float
    var: float
    cvar: float


def _validate(returns, weights, alpha):
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be 2-D: (n_days, n_assets)")
    if w.shape != (r.shape[1],):
        raise ValueError("weights must have one entry per asset")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be in (0.5, 1), e.g. 0.99")
    return r, w


def _cov(r: np.ndarray) -> np.ndarray:
    """Sample covariance matrix, always 2-D (np.cov returns a scalar for one asset)."""
    return np.atleast_2d(np.cov(r, rowvar=False))


def tail_metrics(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR (alpha-quantile of losses) and CVaR (mean loss beyond VaR)."""
    var = float(np.quantile(losses, alpha))
    cvar = float(losses[losses >= var].mean())
    return var, cvar


def historical_var(returns, weights, alpha: float = 0.99) -> RiskEstimate:
    """Historical simulation: replay past returns through today's weights."""
    r, w = _validate(returns, weights, alpha)
    losses = -(r @ w)
    var, cvar = tail_metrics(losses, alpha)
    return RiskEstimate("Historical", alpha, var, cvar)


def parametric_var(returns, weights, alpha: float = 0.99) -> RiskEstimate:
    """Variance-covariance VaR assuming normally distributed portfolio returns."""
    r, w = _validate(returns, weights, alpha)
    mu = float(r.mean(axis=0) @ w)
    sigma = float(np.sqrt(w @ _cov(r) @ w))
    z = norm.ppf(alpha)
    var = -mu + sigma * z
    cvar = -mu + sigma * norm.pdf(z) / (1.0 - alpha)
    return RiskEstimate("Parametric (Normal)", alpha, float(var), float(cvar))


def component_var(returns, weights, alpha: float = 0.99) -> np.ndarray:
    """Per-asset contribution to parametric VaR (Euler allocation, ignoring the mean).

    The contributions sum exactly to sigma_p * z.
    """
    r, w = _validate(returns, weights, alpha)
    cov = _cov(r)
    sigma = np.sqrt(w @ cov @ w)
    marginal = (cov @ w) / sigma
    return w * marginal * norm.ppf(alpha)


def simulate_returns(mu, cov, n_sims: int, dist: str = "normal", df: float = 5.0,
                     antithetic: bool = True, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw correlated return scenarios.

    dist="normal": multivariate normal via Cholesky.
    dist="t":      multivariate Student-t, rescaled so its covariance equals `cov`
                   (fat tails with the same variance as the normal model).
    antithetic:    for every draw z also use -z (variance reduction). n_sims is
                   rounded up to an even number when this is on.
    """
    if rng is None:
        rng = np.random.default_rng()
    mu = np.asarray(mu, dtype=float)
    k = mu.size
    chol = np.linalg.cholesky(cov)
    half = (n_sims + 1) // 2 if antithetic else n_sims
    z = rng.standard_normal((half, k))
    if dist == "t":
        if df <= 2:
            raise ValueError("df must be > 2 for finite variance")
        chi = rng.chisquare(df, size=half)
        z = z * np.sqrt((df - 2.0) / chi)[:, None]
    elif dist != "normal":
        raise ValueError("dist must be 'normal' or 't'")
    if antithetic:
        z = np.vstack([z, -z])
    return mu + z @ chol.T


def monte_carlo_var(returns, weights, alpha: float = 0.99, n_sims: int = 100_000,
                    dist: str = "normal", df: float = 5.0, antithetic: bool = True,
                    seed: int | None = None) -> RiskEstimate:
    """Monte Carlo VaR/CVaR: estimate mean & covariance, simulate, take the tail."""
    r, w = _validate(returns, weights, alpha)
    mu = r.mean(axis=0)
    cov = _cov(r)
    rng = np.random.default_rng(seed)
    sims = simulate_returns(mu, cov, n_sims, dist, df, antithetic, rng)
    var, cvar = tail_metrics(-(sims @ w), alpha)
    name = "Monte Carlo (Normal)" if dist == "normal" else f"Monte Carlo (Student-t, df={df:g})"
    return RiskEstimate(name, alpha, var, cvar)
