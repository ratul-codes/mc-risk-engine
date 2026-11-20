from mcrisk.study import run_study


def test_study_returns_valid_rates_for_every_method():
    rows = run_study(seeds=[1, 2], n_days=500, sims=200)
    assert len(rows) == 6
    for r in rows:
        for key in ("kupiec_pass", "christoffersen_pass", "both_pass"):
            assert 0.0 <= r[key] <= 1.0
        assert r["both_pass"] <= min(r["kupiec_pass"], r["christoffersen_pass"])
        assert r["avg_exceptions"] >= 0
