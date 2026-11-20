# Monte Carlo Portfolio Risk Engine

A small, tested Python library that estimates **Value-at-Risk (VaR)** and **Expected Shortfall (CVaR)** for a portfolio using three methods, then **backtests each one with formal statistical tests** to see which models can actually be trusted.

![Backtest](docs/backtest.png)

## What it does

| Component | Details |
|---|---|
| **Estimators** | Historical simulation, parametric (variance-covariance), Monte Carlo with multivariate Normal, Monte Carlo with multivariate Student-t |
| **Monte Carlo** | Correlated scenarios via Cholesky decomposition; Student-t rescaled so its covariance matches the Normal model (fair comparison); optional antithetic variates |
| **Risk attribution** | Component VaR (Euler allocation): which asset contributes how much of the portfolio risk |
| **Backtesting** | Rolling-window VaR forecasts with no lookahead, **Kupiec** (exception count) and **Christoffersen** (exception clustering) tests |
| **Data** | Synthetic generator (GARCH volatility clustering + fat-tailed shocks) or your own price CSV |

## Quick start

```bash
git clone https://github.com/your-github-username/mc-risk-engine.git
cd mc-risk-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m mcrisk            # full demo (about 30-60 seconds)
pytest                      # test suite
```

Use real data: `python -m mcrisk --csv prices.csv` (first column = date, one column per asset).

## Results (synthetic data, 2,500 days, 8 assets, equal weights, 99% one-day)

**Full-sample estimates**

| Method | VaR | CVaR |
|---|---|---|
| Historical | 2.88% | 4.28% |
| Parametric (Normal) | 2.33% | 2.67% |
| Monte Carlo (Normal) | 2.31% | 2.64% |
| Monte Carlo (Student-t, df=5) | 2.61% | 3.42% |

**Rolling backtest** (250-day window, 2,250 forecasts, 22.5 exceptions expected)

| Method | Exceptions | Kupiec p | Christoffersen p | Verdict |
|---|---|---|---|---|
| Historical | 31 | 0.088 | 0.074 | PASS |
| Parametric (Normal) | 43 | 0.000 | 0.260 | FAIL |
| Monte Carlo (Normal) | 44 | 0.000 | 0.283 | FAIL |
| Monte Carlo (Student-t) | 24 | 0.753 | 0.025 | FAIL |

### What the results say

1. **Normal-based models underestimate tail risk.** They breach their 99% VaR almost twice as often as promised (43-44 vs 22.5 expected), and Kupiec rejects them decisively. The two Normal models agree with each other, so the problem is the *distribution assumption*, not the Monte Carlo machinery.
2. **Fat tails fix the exception count but not the timing.** Student-t gets the number of exceptions right (24 vs 22.5, Kupiec p = 0.75) but fails Christoffersen: its exceptions cluster during the volatility spike (see chart). A 250-day window reacts too slowly when volatility jumps.
3. **Historical simulation passes narrowly** on this sample, but it is noisy and has no model of its own tail, so its result is not a strong guarantee.
4. **Conclusion:** fixing the distribution is necessary but not sufficient; the next improvement is a *volatility model* (EWMA or GARCH) so the forecast adapts quickly.

### Monte Carlo convergence

Standard error of the VaR estimate (100 seeds each) shrinks at roughly the expected 1/sqrt(N) rate:

| Paths | Plain SE | Antithetic SE |
|---|---|---|
| 1,000 | 0.110% | 0.110% |
| 10,000 | 0.044% | 0.041% |
| 100,000 | 0.012% | 0.013% |

Antithetic variates give **no meaningful gain here**. That makes sense: they cancel noise in *averages* of symmetric functions, but VaR is a *tail quantile*, which they do not help much. I kept the option in and reported the null result rather than claiming a benefit.

## Design decisions

- **Losses are positive numbers** (0.025 = a 2.5% loss) so "VaR is the alpha-quantile of losses".
- **No lookahead:** the forecast for day *t* uses only days *t-window ... t-1* (covered by a unit test).
- **Student-t is rescaled** by sqrt((df-2)/df) so its covariance equals the Normal model's. Differences in results come from tail shape only.
- **Estimators share one signature** `f(returns, weights, alpha) -> RiskEstimate`, so new methods plug into the backtester without changes.
- **Tests compare against closed-form answers** (e.g. Normal VaR = sigma * z), check seed reproducibility, covariance recovery, and input validation.

## Limitations

- Results use **synthetic data**; run `--csv` on real prices before drawing conclusions about real markets.
- Student-t degrees of freedom is fixed at 5, not estimated.
- Constant volatility within each window (no EWMA/GARCH), which is exactly what the backtest exposes.
- One-day horizon, linear equity-like portfolio (no options or non-linear payoffs), simple returns.
- Statistical power is limited: 2,250 observations at 99% gives only about 22 expected exceptions.

## What I'd do next

1. Add an **EWMA / GARCH** volatility forecast and show it fixing the Christoffersen failure.
2. **Filtered historical simulation** (standardize by volatility, resample, rescale).
3. Estimate Student-t degrees of freedom by maximum likelihood.
4. Add **stress scenarios** (correlation to 1, 2x volatility).
5. Run on real multi-asset data (e.g. 2008 and 2020 crises) and compare.

## Project layout

```
mcrisk/
  models.py     VaR/CVaR estimators, Monte Carlo simulation, component VaR
  backtest.py   rolling backtest, Kupiec and Christoffersen tests
  data.py       synthetic data generator and CSV loader
  plots.py      charts
  __main__.py   command-line demo
tests/          pytest suite (15 tests)
docs/           generated charts and results.txt
```

## License

MIT
