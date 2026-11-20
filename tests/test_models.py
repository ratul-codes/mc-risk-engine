import numpy as np
import pytest
from scipy.stats import norm

from mcrisk.models import (component_var, historical_var, monte_carlo_var,
                           parametric_var, simulate_returns)

SIGMA = 0.01


def normal_returns(n=200_000, k=3, seed=0):
    return np.random.default_rng(seed).normal(0.0, SIGMA, (n, k))


def eq(k):
    return np.full(k, 1.0 / k)


def test_parametric_var_matches_theory():
    r = normal_returns(k=1)
    est = parametric_var(r, [1.0], 0.99)
    assert est.var == pytest.approx(SIGMA * norm.ppf(0.99), rel=0.03)
    assert est.cvar == pytest.approx(SIGMA * norm.pdf(norm.ppf(0.99)) / 0.01, rel=0.03)


def test_historical_close_to_parametric_on_normal_data():
    r = normal_returns()
    w = eq(3)
    assert historical_var(r, w).var == pytest.approx(parametric_var(r, w).var, rel=0.05)


def test_monte_carlo_normal_close_to_parametric():
    r = normal_returns()
    w = eq(3)
    mc = monte_carlo_var(r, w, n_sims=200_000, seed=1)
    assert mc.var == pytest.approx(parametric_var(r, w).var, rel=0.03)


def test_cvar_exceeds_var():
    r = normal_returns(n=5000)
    w = eq(3)
    for est in (historical_var(r, w), parametric_var(r, w),
                monte_carlo_var(r, w, n_sims=20_000, seed=3)):
        assert est.cvar > est.var > 0


def test_student_t_has_fatter_tail_than_normal_at_extreme_confidence():
    r = normal_returns(n=5000)
    w = eq(3)
    n = monte_carlo_var(r, w, alpha=0.999, n_sims=400_000, dist="normal", seed=2)
    t = monte_carlo_var(r, w, alpha=0.999, n_sims=400_000, dist="t", df=4, seed=2)
    assert t.var > n.var


def test_simulated_covariance_matches_input_for_both_distributions():
    cov = np.array([[1.0, 0.5], [0.5, 2.0]]) * 1e-4
    for dist in ("normal", "t"):
        sims = simulate_returns(np.zeros(2), cov, 400_000, dist=dist, df=8,
                                rng=np.random.default_rng(5))
        assert np.cov(sims, rowvar=False) == pytest.approx(cov, rel=0.06)


def test_seed_gives_reproducible_results():
    r = normal_returns(n=2000)
    w = eq(3)
    a = monte_carlo_var(r, w, n_sims=10_000, seed=11)
    b = monte_carlo_var(r, w, n_sims=10_000, seed=11)
    assert a == b


def test_component_var_sums_to_portfolio_sigma_times_z():
    r = normal_returns(n=5000)
    r[:, 1] += 0.5 * r[:, 0]                  # add correlation
    w = np.array([0.5, 0.3, 0.2])
    cov = np.cov(r, rowvar=False)
    expected = np.sqrt(w @ cov @ w) * norm.ppf(0.99)
    assert component_var(r, w, 0.99).sum() == pytest.approx(expected)


def test_input_validation():
    r = normal_returns(n=100)
    with pytest.raises(ValueError):
        historical_var(r, [0.5, 0.5, 0.5])    # weights do not sum to 1
    with pytest.raises(ValueError):
        historical_var(r, eq(2))              # wrong number of weights
    with pytest.raises(ValueError):
        historical_var(r, eq(3), alpha=1.2)   # bad confidence level
