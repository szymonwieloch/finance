"""Unit tests for toolkit.other — GBM simulation functions."""

import numpy as np
import pandas as pd
import pytest

from toolkit.other import gbm_returns, gbm_prices


# ---------------------------------------------------------------------------
# gbm_returns
# ---------------------------------------------------------------------------

class TestGbmReturns:
    """Tests for :func:`gbm_returns`."""

    def test_default_shape(self):
        """Default arguments produce (10*12+1, 1000) = (121, 1000)."""
        result = gbm_returns()
        assert result.shape == (121, 1000)

    def test_custom_shape(self):
        """Shape matches (years * steps_per_year + 1, scenarios)."""
        result = gbm_returns(years=5, scenarios=50, steps_per_year=4)
        assert result.shape == (21, 50)

    def test_return_type(self):
        """Output is a plain numpy ndarray."""
        result = gbm_returns()
        assert isinstance(result, np.ndarray)

    def test_first_row_all_zeros(self):
        """Row 0 is always zero (t=0 has no return)."""
        result = gbm_returns(scenarios=20)
        np.testing.assert_array_equal(result[0, :], np.zeros(20))

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        np.random.seed(42)
        a = gbm_returns(years=1, scenarios=10, steps_per_year=4)
        np.random.seed(42)
        b = gbm_returns(years=1, scenarios=10, steps_per_year=4)
        np.testing.assert_array_equal(a, b)

    def test_reasonable_annualized_stats(self):
        """With large scenarios the annualized mean ≈ mu and vol ≈ sigma."""
        np.random.seed(123)
        rets = gbm_returns(years=30, scenarios=5000, mu=0.08, sigma=0.20, steps_per_year=12)
        # Annualize: sum log returns
        annual_log_rets = np.log(rets[1:] + 1).sum(axis=0)  # skip row 0
        ann_mean = annual_log_rets.mean() / 30
        ann_vol = annual_log_rets.std() / np.sqrt(30)
        # Wide bounds account for discretisation and sampling noise
        assert 0.03 < ann_mean < 0.13
        assert 0.15 < ann_vol < 0.25

    def test_zero_sigma_deterministic(self):
        """With sigma=0 every scenario is identical; each step return is (1+mu)^dt - 1."""
        np.random.seed(0)
        rets = gbm_returns(years=2, scenarios=5, mu=0.05, sigma=0.0, steps_per_year=1)
        # After row 0 (always 0), every row should be identical across columns
        for col in range(1, 5):
            np.testing.assert_array_almost_equal(rets[:, 0], rets[:, col])
        # Every step draws the same constant: (1+mu)^dt - 1 = 0.05
        expected = (1 + 0.05) ** 1 - 1
        np.testing.assert_array_almost_equal(rets[1:, 0], np.full(rets.shape[0] - 1, expected))

    def test_single_scenario(self):
        """Works with only one scenario."""
        result = gbm_returns(years=2, scenarios=1, steps_per_year=6)
        assert result.shape == (13, 1)

    def test_single_step(self):
        """Works with one step per year, one year."""
        result = gbm_returns(years=1, scenarios=30, steps_per_year=1)
        assert result.shape == (2, 30)


# ---------------------------------------------------------------------------
# gbm_prices
# ---------------------------------------------------------------------------

class TestGbmPrices:
    """Tests for :func:`gbm_prices`."""

    def test_default_shape(self):
        """Default arguments produce (121, 1000) DataFrame."""
        result = gbm_prices()
        assert result.shape == (121, 1000)

    def test_custom_shape(self):
        """Shape matches (years * steps_per_year + 1, scenarios)."""
        result = gbm_prices(years=3, scenarios=80, steps_per_year=2)
        assert result.shape == (7, 80)

    def test_return_type(self):
        """Output is a pandas DataFrame."""
        result = gbm_prices()
        assert isinstance(result, pd.DataFrame)

    def test_first_row_is_s0(self):
        """All prices at t=0 equal s_0."""
        result = gbm_prices(s_0=42.0, scenarios=30)
        np.testing.assert_array_equal(result.iloc[0, :].values, np.full(30, 42.0))

    def test_non_negative_prices(self):
        """GBM prices are always strictly positive."""
        np.random.seed(7)
        result = gbm_prices(years=10, scenarios=200, sigma=0.25)
        assert (result > 0).all().all()

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        np.random.seed(99)
        a = gbm_prices(years=1, scenarios=10, steps_per_year=4)
        np.random.seed(99)
        b = gbm_prices(years=1, scenarios=10, steps_per_year=4)
        pd.testing.assert_frame_equal(a, b)

    def test_zero_sigma_deterministic(self):
        """With sigma=0 every scenario is identical: s_0 * (1+mu)^(t·dt)."""
        np.random.seed(0)
        prices = gbm_prices(years=3, scenarios=5, mu=0.04, sigma=0.0, steps_per_year=1, s_0=200.0)
        for col in range(1, 5):
            pd.testing.assert_series_equal(
                prices.iloc[:, 0], prices.iloc[:, col], check_names=False
            )
        for t in range(prices.shape[0]):
            expected = 200.0 * (1 + 0.04) ** t
            assert np.isclose(prices.iloc[t, 0], expected)

    def test_different_s0(self):
        """All paths start from the supplied s_0."""
        for s0 in [1.0, 50.0, 1000.0]:
            result = gbm_prices(years=1, scenarios=5, s_0=s0, steps_per_year=1)
            np.testing.assert_array_equal(result.iloc[0, :].values, np.full(5, s0))

    def test_single_scenario(self):
        """Works with only one scenario."""
        result = gbm_prices(years=2, scenarios=1, steps_per_year=6)
        assert result.shape == (13, 1)

    def test_consistency_with_gbm_returns(self):
        """gbm_prices called with seed X ≈ cumprod of gbm_returns with seed X."""
        np.random.seed(555)
        prices = gbm_prices(years=4, scenarios=10, mu=0.06, sigma=0.18,
                            steps_per_year=4, s_0=100.0)
        np.random.seed(555)
        rets = gbm_returns(years=4, scenarios=10, mu=0.06, sigma=0.18,
                           steps_per_year=4)
        expected_prices = 100.0 * pd.DataFrame(rets + 1.0).cumprod()
        pd.testing.assert_frame_equal(prices, expected_prices)

    def test_positive_drift_upward_trend(self):
        """With many scenarios and mu>0 the average terminal price > s_0."""
        np.random.seed(42)
        prices = gbm_prices(years=20, scenarios=5000, mu=0.06, sigma=0.15,
                            steps_per_year=12, s_0=100.0)
        terminal_mean = prices.iloc[-1, :].mean()
        assert terminal_mean > 100.0
