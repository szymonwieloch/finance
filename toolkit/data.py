import yfinance as yf
from dateutil.relativedelta import relativedelta
import datetime
import pandas as pd
from pathlib import Path
import re

import tempfile

DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "finance-cache"


def _ticker_key(tickers) -> str:
    """Convert ticker(s) to a filesystem-safe string for cache filenames."""
    if isinstance(tickers, str):
        return tickers
    return "_".join(sorted(tickers))


def _period_to_dates(period: str) -> tuple[datetime.date, datetime.date]:
    """
    Convert a yfinance period string (e.g. '10y', '6mo', 'ytd', 'max')
    to absolute (start, end) dates.
    """
    today = datetime.date.today()
    if period == "ytd":
        return datetime.date(today.year, 1, 1), today
    if period == "max":
        return today - relativedelta(years=50), today
    m = re.match(r"^(\d+)(d|mo|y)$", period)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {"d": relativedelta(days=n), "mo": relativedelta(months=n), "y": relativedelta(years=n)}[unit]
        return today - delta, today
    raise ValueError(f"Unsupported period format: {period!r}")


def _to_date(d) -> datetime.date:
    """Normalize various date-like objects to datetime.date."""
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, str):
        return datetime.date.fromisoformat(d)
    return d


def cached_download(
    tickers,
    start=None,
    end=None,
    period=None,
    interval="1d",
    cache_dir=None,
    **kwargs,
) -> pd.DataFrame:
    """
    A caching wrapper around ``yf.download``.

    Data is stored as parquet files under *cache_dir* (default:
    OS temporary directory, e.g. ``/tmp/finance-cache/`` on Linux).
    hasn't been fetched yet — previously downloaded ranges are
    read from the local cache.

    Parameters
    ----------
    tickers : str or list of str
        Ticker symbol(s), exactly as passed to ``yf.download``.
    start : str or date, optional
        Start date (ignored when *period* is given).
    end : str or date, optional
        End date (defaults to today).
    period : str, optional
        yfinance period string (``'10y'``, ``'6mo'``, ``'ytd'``, ``'max'``, …).
    interval : str
        Data interval (``'1d'``, ``'1wk'``, ``'1mo'``, …).
    cache_dir : str or Path, optional
        Directory for cached parquet files.
    **kwargs
        Passed through to ``yf.download`` (e.g. ``auto_adjust``).

    Returns
    -------
    pd.DataFrame
        The same format that ``yf.download`` returns.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- resolve the absolute date range -----
    today = datetime.date.today()
    if period is not None:
        req_start, req_end = _period_to_dates(period)
    else:
        req_start = _to_date(start) if start is not None else None
        req_end = _to_date(end) if end is not None else today

    # ----- locate cache file -----
    key = _ticker_key(tickers)
    cache_path = cache_dir / f"{key}_{interval}.parquet"

    cached = pd.read_parquet(cache_path) if cache_path.exists() else None

    # ----- fast path: no cache yet -----
    if cached is None or cached.empty:
        new = yf.download(tickers, start=req_start, end=req_end, interval=interval, **kwargs)
        if not new.empty:
            new.to_parquet(cache_path)
        return new

    # ----- determine which edges need downloading -----
    cache_min = _to_date(cached.index.min())
    cache_max = _to_date(cached.index.max())
    pieces = []

    # earlier data needed?  (skip trivial gaps like weekends)
    if req_start is not None and req_start < cache_min - datetime.timedelta(days=3):
        dl_end = cache_min - datetime.timedelta(days=1)
        earlier = yf.download(tickers, start=req_start, end=dl_end, interval=interval, **kwargs)
        if not earlier.empty:
            pieces.append(earlier)

    # later data needed?  (skip trivial gaps like weekends)
    if req_end > cache_max + datetime.timedelta(days=3):
        dl_start = cache_max + datetime.timedelta(days=1)
        later = yf.download(tickers, start=dl_start, end=req_end, interval=interval, **kwargs)
        if not later.empty:
            pieces.append(later)

    # ----- if nothing new, just slice from cache -----
    if not pieces:
        s = pd.Timestamp(req_start) if req_start is not None else None
        e = pd.Timestamp(req_end) if req_end is not None else None
        if s is not None and e is not None:
            return cached.loc[s:e]
        elif s is not None:
            return cached.loc[s:]
        elif e is not None:
            return cached.loc[:e]
        return cached

    # ----- merge new data with cache -----
    merged = pd.concat([cached] + pieces).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.to_parquet(cache_path)

    # ----- return the requested slice -----
    s = pd.Timestamp(req_start) if req_start is not None else None
    e = pd.Timestamp(req_end) if req_end is not None else None
    if s is not None and e is not None:
        return merged.loc[s:e]
    elif s is not None:
        return merged.loc[s:]
    elif e is not None:
        return merged.loc[:e]
    return merged


def get_stock_data(ticker: str | list[str], years=10) -> pd.DataFrame:
    """Convenience wrapper: download *years* of daily data with caching."""
    today = datetime.date.today()
    start = today - relativedelta(years=years)
    data = cached_download(ticker, start=start.isoformat(), end=today.isoformat())
    data.index = pd.to_datetime(data.index).to_period("D")
    return data
