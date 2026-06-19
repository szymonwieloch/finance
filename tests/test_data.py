import datetime
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from toolkit.data import (
    _ticker_key,
    _period_to_dates,
    _to_date,
    cached_download,
    get_stock_data,
    DEFAULT_CACHE_DIR,
)


class TestTickerKey:
    def test_single_ticker(self):
        assert _ticker_key("SPY") == "SPY"

    def test_multiple_tickers_sorted(self):
        assert _ticker_key(["VOO", "SPY"]) == "SPY_VOO"


class TestPeriodToDates:
    def test_years(self):
        start, end = _period_to_dates("10y")
        today = datetime.date.today()
        assert end == today
        assert start == today.replace(year=today.year - 10)

    def test_months(self):
        start, end = _period_to_dates("6mo")
        today = datetime.date.today()
        assert end == today
        expected_month = today.month - 6
        expected_year = today.year
        if expected_month <= 0:
            expected_month += 12
            expected_year -= 1
        assert start.month == expected_month
        assert start.year == expected_year

    def test_days(self):
        start, end = _period_to_dates("5d")
        today = datetime.date.today()
        assert end == today
        assert (today - start).days == 5

    def test_ytd(self):
        start, end = _period_to_dates("ytd")
        today = datetime.date.today()
        assert start == datetime.date(today.year, 1, 1)
        assert end == today

    def test_max(self):
        start, end = _period_to_dates("max")
        today = datetime.date.today()
        assert end == today
        assert (today - start).days > 365 * 49  # roughly 50 years

    def test_invalid(self):
        with pytest.raises(ValueError):
            _period_to_dates("nonsense")


class TestToDate:
    def test_date_unchanged(self):
        d = datetime.date(2024, 1, 15)
        assert _to_date(d) == d

    def test_datetime_converted(self):
        dt = datetime.datetime(2024, 1, 15, 12, 30)
        assert _to_date(dt) == datetime.date(2024, 1, 15)

    def test_string_converted(self):
        assert _to_date("2024-01-15") == datetime.date(2024, 1, 15)

    def test_timestamp_converted(self):
        ts = pd.Timestamp("2024-01-15")
        assert _to_date(ts) == datetime.date(2024, 1, 15)


class TestCachedDownload:
    """Tests for cached_download with mocked yf.download."""

    @pytest.fixture
    def tmp_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def mock_yf_download(self):
        with patch("toolkit.data.yf.download") as mock:
            yield mock

    def _make_df(self, dates, tickers, price=100.0):
        """Build a DataFrame mimicking yf.download output for one or more tickers."""
        if isinstance(tickers, str):
            tickers = [tickers]
        data = {}
        for t in tickers:
            data[("Close", t)] = [float(price + i) for i in range(len(dates))]
            data[("Open", t)] = [float(price + i) for i in range(len(dates))]
            data[("Volume", t)] = [1000] * len(dates)
        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Price", "Ticker"])
        return df

    def test_first_download_saves_cache(self, tmp_cache, mock_yf_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_yf_download.return_value = self._make_df(dates, "SPY")

        result = cached_download("SPY", start="2024-01-01", end="2024-01-07", cache_dir=tmp_cache)

        mock_yf_download.assert_called_once()
        assert not result.empty
        # Cache file should exist
        cache_file = tmp_cache / "SPY_1d.parquet"
        assert cache_file.exists()

    def test_second_call_reads_cache(self, tmp_cache, mock_yf_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_yf_download.return_value = self._make_df(dates, "SPY")

        # First call
        cached_download("SPY", start="2024-01-01", end="2024-01-07", cache_dir=tmp_cache)
        assert mock_yf_download.call_count == 1

        # Second call — same range, should NOT call yf.download again
        result = cached_download("SPY", start="2024-01-01", end="2024-01-07", cache_dir=tmp_cache)
        assert mock_yf_download.call_count == 1  # no additional call
        assert not result.empty

    def test_extend_forward_downloads_only_new(self, tmp_cache, mock_yf_download):
        old_dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_yf_download.return_value = self._make_df(old_dates, "SPY")

        # First: download old range
        cached_download("SPY", start="2024-01-01", end="2024-01-07", cache_dir=tmp_cache)
        assert mock_yf_download.call_count == 1

        # Second: extend forward — only the new tail should be downloaded
        new_dates = pd.date_range("2024-01-08", periods=3, freq="B")
        # The second call will first be for the "later" slice only
        mock_yf_download.reset_mock()
        mock_yf_download.return_value = self._make_df(new_dates, "SPY")

        result = cached_download("SPY", start="2024-01-01", end="2024-01-10", cache_dir=tmp_cache)
        assert mock_yf_download.call_count == 1  # only the new tail
        # Result should include all dates
        assert len(result) == 8  # 5 old + 3 new

    def test_extend_backward_downloads_only_old(self, tmp_cache, mock_yf_download):
        old_dates = pd.date_range("2024-01-08", periods=5, freq="B")
        mock_yf_download.return_value = self._make_df(old_dates, "SPY")

        # First: download old range
        cached_download("SPY", start="2024-01-08", end="2024-01-12", cache_dir=tmp_cache)
        assert mock_yf_download.call_count == 1

        # Second: extend backward
        new_dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_yf_download.reset_mock()
        mock_yf_download.return_value = self._make_df(new_dates, "SPY")

        result = cached_download("SPY", start="2024-01-01", end="2024-01-12", cache_dir=tmp_cache)
        assert mock_yf_download.call_count == 1  # only the older tail
        assert len(result) == 10

    def test_period_parameter(self, tmp_cache, mock_yf_download):
        dates = pd.date_range("2026-06-01", periods=10, freq="B")
        mock_yf_download.return_value = self._make_df(dates, "SPY")

        result = cached_download("SPY", period="5d", cache_dir=tmp_cache)

        mock_yf_download.assert_called_once()
        # Verify start/end were resolved from period and passed through
        call_kwargs = mock_yf_download.call_args.kwargs
        assert "start" in call_kwargs
        assert "end" in call_kwargs

    def test_multi_ticker(self, tmp_cache, mock_yf_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_yf_download.return_value = self._make_df(dates, ["SPY", "VOO"])

        result = cached_download(["SPY", "VOO"], start="2024-01-01", end="2024-01-07", cache_dir=tmp_cache)

        assert not result.empty
        cache_file = tmp_cache / "SPY_VOO_1d.parquet"
        assert cache_file.exists()

    def test_kwargs_passthrough(self, tmp_cache, mock_yf_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_yf_download.return_value = self._make_df(dates, "SPY")

        cached_download("SPY", start="2024-01-01", end="2024-01-07",
                        cache_dir=tmp_cache, auto_adjust=False)

        call_kwargs = mock_yf_download.call_args.kwargs
        assert call_kwargs.get("auto_adjust") is False


class TestGetStockData:
    @patch("toolkit.data.cached_download")
    def test_converts_index_to_period(self, mock_cached):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        # Build a simple single-ticker dataframe (non-multiindex columns for simplicity)
        df = pd.DataFrame(
            {"Close": [100, 101, 102], "Open": [99, 100, 101]},
            index=dates,
        )
        mock_cached.return_value = df

        result = get_stock_data("SPY", years=10)

        assert isinstance(result.index, pd.PeriodIndex)
        assert result.index.freqstr == "D"
