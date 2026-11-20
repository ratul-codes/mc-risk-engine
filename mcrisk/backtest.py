"""Rolling VaR backtest with the Kupiec and Christoffersen tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2

from .models import RiskEstimate

Estimator = Callable[[np.ndarray, np.ndarray, float], RiskEstimate]


def kupiec_pof(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test: is the exception rate equal to 1-alpha?

    Returns (LR statistic, p-value). Small p-value => the model's exception
    frequency is statistically inconsistent with its stated confidence level.
    """
    hits = np.asarray(hits, dtype=bool)
    n, x = hits.size, int(hits.sum())
    p = 1.0 - alpha
    p_hat = x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence test: do exceptions cluster in time?

    Returns (LR statistic, p-value). Small p-value => exceptions are clustered
    (the model reacts too slowly to changing volatility).
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return float(lr), float(chi2.sf(lr, 1))


@dataclass
class BacktestResult:
    method: str
    alpha: float
    losses: np.ndarray
    var_forecasts: np.ndarray
    hits: np.ndarray
    kupiec_p: float
    christoffersen_p: float

    @property
    def n_obs(self) -> int:
        return int(self.hits.size)

    @property
    def n_exceptions(self) -> int:
        return int(self.hits.sum())

    @property
    def expected_exceptions(self) -> float:
        return self.n_obs * (1.0 - self.alpha)

    def verdict(self, significance: float = 0.05) -> str:
        ok = self.kupiec_p >= significance and self.christoffersen_p >= significance
        return "PASS" if ok else "FAIL"


def rolling_backtest(returns, weights, estimator: Estimator, window: int = 250,
                     alpha: float = 0.99) -> BacktestResult:
    """Forecast VaR each day from the previous `window` days, compare to the realised loss.

    The forecast for day t only uses data from days t-window .. t-1 (no lookahead).
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    forecasts, name = [], None
    for t in range(window, r.shape[0]):
        est = estimator(r[t - window:t], w, alpha)
        name = est.method
        forecasts.append(est.var)
    forecasts = np.asarray(forecasts)
    losses = -(r[window:] @ w)
    hits = losses > forecasts
    _, kp = kupiec_pof(hits, alpha)
    _, cp = christoffersen_independence(hits)
    return BacktestResult(name, alpha, losses, forecasts, hits, kp, cp)
