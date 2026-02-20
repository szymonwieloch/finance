import pandas as pd
import numpy as np
from .general import returns_to_prices
import scipy.stats as stats
import typing


def drawdown(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculates drawdown series from a time series of asset returns.

    Args:
        returns (pd.Series or pd.DataFrame): Time series of asset returns.

    Returns:
        pd.Series or pd.DataFrame: Drawdown series.
    """
    wealth_index = returns_to_prices(returns)
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return drawdowns


def semideviation(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    """
    Calculates the semideviation (negative semideviation) of returns.

    The semideviation is the standard deviation of returns that are less than
    zero and is a measure of downside risk.

    Args:
        returns (pd.Series or pd.DataFrame): Returns series or DataFrame.

    Returns:
        float or pd.Series: Semideviation value for a Series, or a Series of
            semideviations when a DataFrame is provided.

    Raises:
        TypeError: If `returns` is not a Series or DataFrame.
    """
    if isinstance(returns, pd.Series):
        is_negative = returns < 0
        return returns[is_negative].std(ddof=0)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(semideviation)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def var_historic(
    returns: pd.Series | pd.DataFrame, level: float = 0.05
) -> pd.Series | float:
    """
    Calculates historical Value at Risk (VaR) for returns.

    Args:
        returns (pd.Series or pd.DataFrame): Returns series or DataFrame.
        level (float): VaR level (e.g., 0.05 for 5%).

    Returns:
        float or pd.Series: VaR value(s) at the specified level.

    Raises:
        TypeError: If `returns` is not a Series or DataFrame.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
    elif isinstance(returns, pd.Series):
        return -np.percentile(returns, level*100)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def var_gaussian(
    returns: pd.Series | pd.DataFrame | None = None,
    level: float = 0.05,
    mean: float = None,
    std: float = None,
) -> pd.Series | float:
    """
    Calculates Gaussian (parametric) Value at Risk (VaR).

    This method assumes returns are normally distributed and computes VaR
    using the provided or calculated mean and standard deviation.

    Args:
        returns (pd.Series or pd.DataFrame, optional): Returns to compute
            statistics from if `mean` and `std` are not provided.
        level (float): VaR level (e.g., 0.05 for 5%).
        mean (float, optional): Precomputed mean of returns. If None, it's
            calculated from `returns`.
        std (float, optional): Precomputed standard deviation of returns. If
            None, it's calculated from `returns`.

    Returns:
        float or pd.Series: Gaussian VaR value(s).
    """
    mean, std = _mean_std(returns, mean, std)
    z = stats.norm.ppf(level)
    return -(mean + z * std)


def var_cornish_fisher(returns: pd.Series, level: float = 0.05) -> float:
    """
    Calculates Value at Risk (VaR) using the Cornish-Fisher expansion.

    The Cornish-Fisher expansion adjusts the Z-score to account for
    skewness and kurtosis in the returns distribution.

    Args:
        returns (pd.Series): Returns series.
        level (float): VaR level (e.g., 0.05 for 5%).

    Returns:
        float: Cornish-Fisher adjusted VaR (as a positive loss value).
    """
    # 1. Basic Stats
    mu = np.mean(returns)
    sigma = np.std(returns)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)  # Scipy returns excess kurtosis by default

    # 2. Get the standard Z-score (e.g., -1.645 for 5%)
    z = stats.norm.ppf(level)

    # 3. Cornish-Fisher Expansion
    z_cf = _adjusted_z(z, s, k)

    # 4. Calculate VaR
    # We return the absolute loss amount
    var_result = -(mu + z_cf * sigma)
    return var_result


def es_historic(
    returns: pd.Series | pd.DataFrame, level: float = 0.05
) -> pd.Series | float:
    """
    Computes historical Expected Shortfall (ES / CVaR).

    Expected Shortfall is the expected return conditional on returns being
    beyond the VaR threshold (i.e., the average of the worst `level` percent
    of returns).

    Args:
        returns (pd.Series or pd.DataFrame): Returns series or DataFrame.
        level (float): Tail level (e.g., 0.05 for the worst 5%).

    Returns:
        float or pd.Series: Expected Shortfall value(s).

    Raises:
        TypeError: If `returns` is not a Series or DataFrame.
    """
    if isinstance(returns, pd.Series):
        is_beyond = returns <= -var_historic(returns, level=level)
        return -returns[is_beyond].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(es_historic, level=level)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def es_gaussian(
    returns: pd.Series | pd.DataFrame,
    level: float = 0.05,
    mean: float = None,
    std: float = None,
) -> pd.Series | float:
    """
    Calculates Gaussian Expected Shortfall (ES / CVaR).

    This computes ES under the assumption that returns are normally
    distributed, using the provided or computed mean and standard
    deviation.

    Args:
        returns (pd.Series or pd.DataFrame): Returns to compute statistics
            from if `mean` and `std` are not provided.
        level (float): Tail level (e.g., 0.05 for the worst 5%).
        mean (float, optional): Precomputed mean of returns.
        std (float, optional): Precomputed standard deviation of returns.

    Returns:
        float or pd.Series: Expected Shortfall value(s).
    """
    # Calculate Mean and Standard Deviation
    mean, std = _mean_std(returns, mean, std)
    z = stats.norm.ppf(level)

    return -mean + std * (stats.norm.pdf(z) / level)


def es_cornish_fisher(returns: pd.Series, level: float = 0.05) -> float:
    """
    Calculates Expected Shortfall using the Cornish-Fisher expansion.

    This method modifies the Gaussian ES estimate by adjusting the Z-score
    for skewness and kurtosis using the Cornish-Fisher expansion.

    Args:
        returns (pd.Series): Returns series.
        level (float): Tail level (e.g., 0.05 for the worst 5%).

    Returns:
        float: Cornish-Fisher adjusted Expected Shortfall.
    """
    mu = returns.mean()
    sigma = returns.std()
    s = stats.skew(returns)
    k = stats.kurtosis(returns)  # scipy returns 'excess' kurtosis by default
    z = stats.norm.ppf(level)

    # 1. Calculate the Cornish-Fisher Adjusted Z-score
    z_cf = _adjusted_z(z, s, k)

    # 3. Estimate ES
    # Note: Unlike Gaussian, there isn't a simple closed-form ES for CF.
    # We typically use the modified Z-score to find the tail expectations.
    # A common robust approach is to use the modified VaR in a Cornish-Fisher
    # specific density integration or simpler: use it to scale the Gaussian ES.

    gauss_es_z = -(stats.norm.pdf(z) / level)
    # Adjustment factor based on the ratio of CF-VaR to Gaussian-VaR
    cf_adjustment = z_cf / z

    return -(mu + (sigma * gauss_es_z * cf_adjustment))


def _adjusted_z(z: float, skew: float, kurtosis: float) -> float:
    """
    Adjusts the Z-score using the Cornish-Fisher expansion.

    Args:
        z (float): Base Z-score.
        skew (float): Skewness of the distribution.
        kurtosis (float): Excess kurtosis of the distribution.

    Returns:
        float: Adjusted Z-score.
    """
    return (
        z
        + (1 / 6) * (z**2 - 1) * skew
        + (1 / 24) * (z**3 - 3 * z) * kurtosis
        - (1 / 36) * (2 * z**3 - 5 * z) * skew**2
    )

def _all_defined(**kwargs) -> bool:
    """
    Helper to check if all provided keyword arguments are not None.

    Args:
        **kwargs: Arbitrary keyword arguments to check.

    Returns:
        bool: True if all values are not None, False  when all of them are None.

    Raises:
        ValueError: If mixture of None and non-None values is provided.
    """
    all_none = all(value is None for value in kwargs.values())
    all_not_none = all(value is not None for value in kwargs.values())
    if not all_none and not all_not_none:
        names = ", ".join(kwargs.keys())
        raise ValueError(f"All arguments {names} must be either None or non-None")
    return all_not_none


def _mean_std(returns, mean, std) -> typing.Tuple[float, float]:
    """
    Helper to obtain or validate mean and std values.

    Args:
        returns (pd.Series or pd.DataFrame): Source of statistics if mean/std
            are not provided.
        mean (float or None): Precomputed mean.
        std (float or None): Precomputed standard deviation.

    Returns:
        tuple[float, float]: (mean, std)

    Raises:
        ValueError: If only one of `mean` or `std` is provided or if neither
            statistics nor returns are available.
    """
    returns_provided = returns is not None
    mean_std_provided = _all_defined(mean=mean, std=std)
    if returns_provided == mean_std_provided:
        raise ValueError(
            "Must provide either both mean and std, or returns"
        )
    if mean is None:
        mean = returns.mean()
    if std is None:
        std = returns.std(axis=0)
    return mean, std
