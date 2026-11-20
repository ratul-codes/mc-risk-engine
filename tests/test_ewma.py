import numpy as np
import pytest
from scipy.stats import norm

from mcrisk.ewma import ewma_cov, ewma_monte_carlo_var, ewma_parametric_var
from mcrisk.models import parametric_var


def test_lambda_one_is_equal_weighted_second_moment():
    r = np.random.default_rng(0).normal(0, 0.01, (500, 3))
    assert ewma_cov(r, lam=1.0) == pytest.approx(r.T @ r / 500)


def test_newest_observation_gets_the_largest_weight():
    r = np.zeros((10, 1))
    r[-1] = 0.05                       # shock on the newest day
    r2 = np.zeros((10, 1))
    r2[0] = 0.05                       # same shock on the oldest day
    assert ewma_cov(r, 0.9)[0, 0] > ewma_cov(r2, 0.9)[0, 0]


def test_ewma_close_to_theory_on_stationary_normal_data():
    r = np.random.default_rng(1).normal(0, 0.01, (50_000, 1))
    est = ewma_parametric_var(r, [1.0], 0.99, lam=0.9995)
    assert est.var == pytest.approx(0.01 * norm.ppf(0.99), rel=0.08)


def test_ewma_reacts_faster_than_equal_weight_when_volatility_jumps():
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, (230, 2))
    wild = rng.normal(0, 0.03, (20, 2))
    r = np.vstack([calm, wild])
    w = np.array([0.5, 0.5])
    assert ewma_parametric_var(r, w).var > 1.5 * parametric_var(r, w).var


def test_ewma_monte_carlo_student_t_has_fatter_tail_than_normal_at_999():
    r = np.random.default_rng(3).normal(0, 0.01, (1000, 3))
    w = np.full(3, 1 / 3)
    n = ewma_monte_carlo_var(r, w, 0.999, n_sims=400_000, dist="normal", seed=4)
    t = ewma_monte_carlo_var(r, w, 0.999, n_sims=400_000, dist="t", df=4, seed=4)
    assert t.var > n.var


def test_invalid_lambda_rejected():
    r = np.random.default_rng(5).normal(0, 0.01, (50, 2))
    with pytest.raises(ValueError):
        ewma_cov(r, lam=1.5)
