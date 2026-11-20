"""Data helpers: a synthetic generator and a CSV loader for real prices."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_returns(n_days: int = 2500, n_assets: int = 8, seed: int = 7):
    """Simulate daily returns with fat tails and volatility clustering.

    A market factor follows GARCH(1,1) with Student-t(5) shocks (so volatility
    clusters and extreme days are more common than a normal model expects).
    Each asset = beta * market + fat-tailed idiosyncratic noise.
    Returns (array of shape (n_days, n_assets), list of asset names).
    """
    rng = np.random.default_rng(seed)
    omega, a, b = 2e-6, 0.09, 0.89          # long-run daily vol = sqrt(omega/(1-a-b)) = 1%
    h = omega / (1 - a - b)
    scale_t5 = np.sqrt(3.0 / 5.0)           # makes t(5) shocks have unit variance
    mkt = np.empty(n_days)
    for t in range(n_days):
        mkt[t] = np.sqrt(h) * rng.standard_t(5) * scale_t5
        h = omega + a * mkt[t] ** 2 + b * h
    betas = rng.uniform(0.6, 1.4, n_assets)
    idio_vol = rng.uniform(0.006, 0.012, n_assets)
    idio = rng.standard_t(5, (n_days, n_assets)) * scale_t5 * idio_vol
    returns = 0.0003 + mkt[:, None] * betas + idio
    return returns, [f"ASSET_{i + 1}" for i in range(n_assets)]


def load_prices_csv(path: str):
    """Load a CSV of prices (first column = date, other columns = one asset each).

    Returns (simple daily returns array, list of asset names).
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df.select_dtypes("number").ffill().dropna()
    if df.shape[1] < 2:
        raise ValueError("need at least two numeric price columns")
    rets = df.pct_change().dropna()
    return rets.to_numpy(), list(rets.columns)
