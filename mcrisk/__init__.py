"""mcrisk: a small Monte Carlo portfolio risk engine with VaR backtesting."""
from .models import (
    RiskEstimate,
    historical_var,
    parametric_var,
    monte_carlo_var,
    simulate_returns,
    component_var,
)
from .backtest import BacktestResult, rolling_backtest, kupiec_pof, christoffersen_independence
from .ewma import ewma_cov, ewma_parametric_var, ewma_monte_carlo_var
from .data import synthetic_returns, load_prices_csv

__all__ = [
    "RiskEstimate", "historical_var", "parametric_var", "monte_carlo_var",
    "simulate_returns", "component_var", "BacktestResult", "rolling_backtest",
    "kupiec_pof", "christoffersen_independence", "synthetic_returns",
    "load_prices_csv", "ewma_cov", "ewma_parametric_var",
    "ewma_monte_carlo_var",
]
