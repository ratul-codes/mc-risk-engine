import numpy as np

from mcrisk.backtest import christoffersen_independence, kupiec_pof, rolling_backtest
from mcrisk.models import historical_var, parametric_var


def test_kupiec_accepts_expected_rate():
    hits = np.zeros(1000, dtype=bool)
    hits[::100] = True                        # exactly 10 of 1000 = 1%
    lr, p = kupiec_pof(hits, 0.99)
    assert lr < 1e-9 and p > 0.99


def test_kupiec_rejects_too_many_exceptions():
    hits = np.zeros(1000, dtype=bool)
    hits[::25] = True                         # 4% exception rate vs 1% expected
    assert kupiec_pof(hits, 0.99)[1] < 0.001


def test_kupiec_handles_zero_exceptions():
    lr, p = kupiec_pof(np.zeros(500, dtype=bool), 0.99)
    assert np.isfinite(lr) and 0.0 <= p <= 1.0


def test_christoffersen_flags_clustering():
    spread = np.zeros(1000, dtype=bool)
    spread[::100] = True
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True                 # same count, all in a row
    assert christoffersen_independence(spread)[1] > 0.5
    assert christoffersen_independence(clustered)[1] < 0.001


def test_backtest_has_no_lookahead_and_correct_shape():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, (600, 3))
    w = np.full(3, 1 / 3)
    res = rolling_backtest(r, w, historical_var, window=250, alpha=0.99)
    assert res.n_obs == 350
    # Changing the final day's return must not change the forecast made for that day.
    r2 = r.copy()
    r2[-1] += 0.5
    res2 = rolling_backtest(r2, w, historical_var, window=250, alpha=0.99)
    assert res.var_forecasts[-1] == res2.var_forecasts[-1]


def test_backtest_on_gaussian_data_is_calibrated():
    rng = np.random.default_rng(123)
    r = rng.normal(0, 0.01, (3000, 4))
    w = np.full(4, 0.25)
    res = rolling_backtest(r, w, parametric_var, window=250, alpha=0.99)
    assert res.kupiec_p > 0.01                # a correct model should not be rejected
