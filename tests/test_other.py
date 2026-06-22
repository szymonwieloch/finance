"""Unit tests for toolkit.other — GBM simulation functions."""

import numpy as np
import pandas as pd
import pytest

from toolkit.other import discount, funding_ratio, gbm_prices, gbm_returns, present_value


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


# ---------------------------------------------------------------------------
# discount
# ---------------------------------------------------------------------------

class TestDiscount:
    """Tests for :func:`discount`."""

    # -- scalar input -------------------------------------------------------

    def test_scalar_returns_float(self):
        """Scalar t should return a plain float."""
        result = discount(5.0, 0.05)
        assert isinstance(result, float)

    def test_scalar_t_zero(self):
        """At t=0 the discount factor is always 1, regardless of r."""
        assert discount(0.0, 0.00) == 1.0
        assert discount(0.0, 0.05) == 1.0
        assert discount(0.0, 0.20) == 1.0

    def test_scalar_r_zero(self):
        """With r=0 the discount factor is always 1."""
        assert discount(0.0, 0.0) == 1.0
        assert discount(10.0, 0.0) == 1.0
        assert discount(100.0, 0.0) == 1.0

    def test_scalar_known_values(self):
        """Spot-check against hand-computed values."""
        # (1.05)^{-1} ≈ 0.95238
        assert discount(1.0, 0.05) == pytest.approx(0.9523809523809523)
        # (1.05)^{-5} ≈ 0.783526
        assert discount(5.0, 0.05) == pytest.approx(0.7835261664684595)
        # (1.10)^{-3} ≈ 0.7513148
        assert discount(3.0, 0.10) == pytest.approx(0.7513148009015775)

    def test_scalar_fractional_t(self):
        """Works with fractional years (e.g. half-year)."""
        # (1.05)^{-0.5} ≈ 0.9759
        assert discount(0.5, 0.05) == pytest.approx(0.9759000729485332)

    def test_scalar_large_t_near_zero(self):
        """Large t makes the discount factor approach zero."""
        result = discount(1000.0, 0.05)
        assert 0.0 < result < 1e-20

    # -- numpy array input --------------------------------------------------

    def test_array_returns_ndarray(self):
        """Array t should return a numpy ndarray."""
        result = discount(np.array([1.0, 2.0, 3.0]), 0.05)
        assert isinstance(result, np.ndarray)

    def test_array_shape_preserved(self):
        """Output shape matches input shape."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = discount(t, 0.05)
        assert result.shape == t.shape

    def test_array_known_values(self):
        """Every entry matches (1+r)^{-t_i}."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        r = 0.05
        expected = (1 + r) ** (-t)
        np.testing.assert_array_almost_equal(discount(t, r), expected)

    def test_array_r_zero(self):
        """With r=0 all discount factors are 1."""
        t = np.array([0.0, 5.0, 10.0, 100.0])
        np.testing.assert_array_equal(discount(t, 0.0), np.ones_like(t))

    def test_array_empty(self):
        """Empty array returns empty array."""
        result = discount(np.array([]), 0.05)
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_array_2d(self):
        """Works with 2-D arrays."""
        t = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = discount(t, 0.05)
        expected = (1 + 0.05) ** (-t)
        np.testing.assert_array_almost_equal(result, expected)

    # -- pandas Index input -------------------------------------------------

    def test_index_returns_index(self):
        """Pandas Index t returns a pandas Index (preserves input type)."""
        idx = pd.Index([1.0, 2.0, 3.0])
        result = discount(idx, 0.05)
        assert isinstance(result, pd.Index)

    def test_index_len_and_values(self):
        """Series length matches index, values match (1+r)^{-t}."""
        idx = pd.Index([0.0, 1.0, 2.0, 3.0, 4.0])
        r = 0.05
        result = discount(idx, r)
        expected = (1 + r) ** (-idx)
        assert len(result) == len(idx)
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_index_preserves_name(self):
        """Index name is carried through to the result."""
        idx = pd.Index([1.0, 2.0, 3.0], name="maturity")
        result = discount(idx, 0.05)
        assert result.name == "maturity"

    def test_index_zero_rate(self):
        """With r=0 all discount factors are 1."""
        idx = pd.Index([0.0, 2.5, 7.0])
        result = discount(idx, 0.0)
        expected = pd.Index([1.0, 1.0, 1.0])
        pd.testing.assert_index_equal(result, expected)


# ---------------------------------------------------------------------------
# present_value
# ---------------------------------------------------------------------------

class TestPresentValue:
    """Tests for :func:`present_value`."""

    def test_returns_float(self):
        """Output is a scalar float."""
        s = pd.Series([100.0], index=[1.0])
        result = present_value(s, 0.05)
        assert isinstance(result, float)

    def test_single_liability(self):
        """PV of a single cash flow = CF * (1+r)^{-t}."""
        s = pd.Series([100.0], index=[2.0])
        r = 0.05
        expected = 100.0 * (1 + r) ** (-2.0)
        assert present_value(s, r) == pytest.approx(expected)

    def test_multiple_liabilities(self):
        """PV of multiple cash flows = sum of individually discounted values."""
        liabilities = pd.Series(
            [100.0, 200.0, 300.0],
            index=[1.0, 2.0, 3.0],
        )
        r = 0.05
        expected = (
            100.0 * (1 + r) ** (-1.0)
            + 200.0 * (1 + r) ** (-2.0)
            + 300.0 * (1 + r) ** (-3.0)
        )
        assert present_value(liabilities, r) == pytest.approx(expected)

    def test_zero_rate(self):
        """With r=0 PV = sum of all cash flows."""
        liabilities = pd.Series([10.0, 20.0, 30.0], index=[1.0, 5.0, 10.0])
        assert present_value(liabilities, 0.0) == 60.0

    def test_all_at_t_zero(self):
        """When all liabilities are at t=0, PV = sum of cash flows."""
        liabilities = pd.Series([50.0, 75.0], index=[0.0, 0.0])
        r = 0.08
        assert present_value(liabilities, r) == pytest.approx(125.0)

    def test_empty_series(self):
        """An empty Series has PV = 0.0."""
        s = pd.Series([], dtype=float, index=pd.Index([], dtype=float))
        assert present_value(s, 0.05) == 0.0

    def test_mixed_times(self):
        """Liabilities at different times discount correctly."""
        liabilities = pd.Series([100.0, 100.0], index=[0.0, 10.0])
        r = 0.10
        expected = 100.0 + 100.0 * (1.10) ** (-10.0)
        assert present_value(liabilities, r) == pytest.approx(expected)

    def test_consistency_with_discount(self):
        """present_value should match manual discount + sum."""
        liabilities = pd.Series(
            [500.0, 800.0, 200.0],
            index=[0.5, 2.0, 7.0],
        )
        r = 0.04
        # Manual computation using discount()
        pv = 0.0
        for t, cf in liabilities.items():
            pv += cf * discount(t, r)
        assert present_value(liabilities, r) == pytest.approx(pv)

    def test_negative_cashflows(self):
        """Negative cash flows (e.g. costs) are handled correctly."""
        liabilities = pd.Series([100.0, -50.0], index=[1.0, 2.0])
        r = 0.05
        expected = 100.0 * (1.05) ** (-1.0) - 50.0 * (1.05) ** (-2.0)
        assert present_value(liabilities, r) == pytest.approx(expected)

    def test_non_integer_times(self):
        """Works with non-integer payment times."""
        liabilities = pd.Series([100.0], index=[1.5])
        r = 0.06
        expected = 100.0 * (1.06) ** (-1.5)
        assert present_value(liabilities, r) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# funding_ratio
# ---------------------------------------------------------------------------

class TestFundingRatio:
    """Tests for :func:`funding_ratio`."""

    def test_returns_float(self):
        """Output is a scalar float."""
        liabilities = pd.Series([100.0], index=[1.0])
        result = funding_ratio(200.0, liabilities, 0.05)
        assert isinstance(result, float)

    def test_fully_funded(self):
        """When assets == PV of liabilities, ratio is 1."""
        liabilities = pd.Series([100.0], index=[0.0])
        r = 0.05
        pv = present_value(liabilities, r)
        assert funding_ratio(pv, liabilities, r) == pytest.approx(1.0)

    def test_surplus(self):
        """Assets > PV → ratio > 1."""
        liabilities = pd.Series([100.0], index=[1.0])
        r = 0.05
        pv = present_value(liabilities, r)
        result = funding_ratio(pv * 1.5, liabilities, r)
        assert result == pytest.approx(1.5)

    def test_shortfall(self):
        """Assets < PV → ratio < 1."""
        liabilities = pd.Series([100.0], index=[1.0])
        r = 0.05
        pv = present_value(liabilities, r)
        result = funding_ratio(pv * 0.5, liabilities, r)
        assert result == pytest.approx(0.5)

    def test_zero_assets(self):
        """Zero assets → ratio is 0."""
        liabilities = pd.Series([100.0, 200.0], index=[1.0, 2.0])
        assert funding_ratio(0.0, liabilities, 0.05) == 0.0

    def test_zero_rate(self):
        """With r=0, PV = sum of cash flows."""
        liabilities = pd.Series([100.0, 200.0, 300.0], index=[1.0, 2.0, 3.0])
        result = funding_ratio(300.0, liabilities, 0.0)
        assert result == pytest.approx(300.0 / 600.0)

    def test_multiple_liabilities(self):
        """Ratio = assets / Σ CF_i * (1+r)^{-t_i}."""
        liabilities = pd.Series([100.0, 200.0], index=[2.0, 5.0])
        r = 0.04
        expected_ratio = 250.0 / present_value(liabilities, r)
        assert funding_ratio(250.0, liabilities, r) == pytest.approx(expected_ratio)

    def test_consistency_with_present_value(self):
        """funding_ratio ≡ assets / present_value."""
        liabilities = pd.Series([500.0, 300.0, 200.0], index=[0.5, 3.0, 8.0])
        r = 0.06
        assets = 1200.0
        expected = assets / present_value(liabilities, r)
        assert funding_ratio(assets, liabilities, r) == pytest.approx(expected)
