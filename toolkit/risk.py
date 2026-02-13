import pandas as pd
import numpy as np
from .general import returns_to_prices
import scipy.stats as stats

def drawdown(returns: pd.Series|pd.DataFrame) -> pd.Series|pd.DataFrame:
    """Calculates drawdown series from a time series of asset returns.
    """
    wealth_index = returns_to_prices(returns)
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks

    return drawdowns


def semideviation(returns: pd.Series|pd.DataFrame) -> pd.Series|float:
    """
    Calculates the semideviation aka negative semideviation of returns series. 
    The semideviation is the standard deviation of returns that are less than zero. It is a measure of downside risk.
    returns must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(returns, pd.Series):
        is_negative = returns < 0
        return returns[is_negative].std(ddof=0)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(semideviation)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")
    


def var_historic(returns: pd.Series|pd.DataFrame, level:float=0.05) -> pd.Series|float:
    """
    Calculates value at risk (VaR) of a Series or DataFrame using the historical method

    Parameters:
    returns (pd.Series|pd.DataFrame): The returns Series or DataFrame.
    level (float): The level of VaR to calculate (e.g., 0.05 for 5%).

    Returns:
    float|pd.Series: The VaR value(s) at the specified level.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
    elif isinstance(returns, pd.Series):
        return -np.percentile(returns, level/100)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")
    

def var_gaussian(returns: pd.Series|pd.DataFrame, level=0.05):
    """
   Calculates value at risk (VaR) of a Series or DataFrame using the Gaussian method. 
    This method assumes that returns are normally distributed and calculates VaR based on the mean and standard deviation of the returns.

    Parameters:
    returns (pd.Series|pd.DataFrame): The returns Series or DataFrame.
    level (float): The level of VaR to calculate (e.g., 0.05 for 5%). Specifies the confidence level for VaR (e.g., 0.05 for 5% VaR).
    
    Returns:
    float|pd.Series: The VaR value(s) at the specified level.
    """
    # compute the Z score assuming it was Gaussian
    z = stats.norm.ppf(level)
    return -(returns.mean() + z*returns.std())


def var_cornish_fisher(returns: pd.Series, level:float=0.05)-> float:
    # 1. Basic Stats
    mu = np.mean(returns)
    sigma = np.std(returns)
    s = stats.skew(returns)
    k = stats.kurtosis(returns) # Scipy returns excess kurtosis by default
    
    # 2. Get the standard Z-score (e.g., -1.645 for 5%)
    zp = stats.norm.ppf(level)
    
    # 3. Cornish-Fisher Expansion
    z_cf = (zp + 
            (1/6) * (zp**2 - 1) * s + 
            (1/24) * (zp**3 - 3*zp) * k - 
            (1/36) * (2*zp**3 - 5*zp) * s**2)
    
    # 4. Calculate VaR
    # We return the absolute loss amount
    var_result = -(mu + z_cf * sigma)
    return var_result


def es_historic(returns: pd.Series|pd.DataFrame, level:float=0.05)-> pd.Series|float:
    """
    Computes the estimated shortfall (CVaR) of a Series or DataFrame using the historical method. 
    Estimated shortfall (CVaR) is the expected return in the worst "level" percent of cases. 
    For example, if level=0.05, then CVaR is the expected return in the worst 5% of cases.
    """
    if isinstance(returns, pd.Series):
        is_beyond = returns <= -var_historic(returns, level=level)
        return -returns[is_beyond].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(es_historic, level=level)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def es_gaussian(returns: pd.Series|pd.DataFrame, level:float=0.05)-> pd.Series|float:
    """
    Calculates Expected Shortfall (CVaR) assuming a Normal Distribution.
    """
    # Calculate Mean and Standard Deviation
    mu = returns.mean()
    sigma = returns.std()
    zp = stats.norm.ppf(level)
    
    return mu - sigma * (stats.norm.pdf(zp) / level)

def es_cornish_fisher(returns: pd.Series, level:float=0.05) -> float:
    """
    Calculates Expected Shortfall using the Cornish-Fisher expansion
    to account for Skewness and Kurtosis.
    """
    mu = returns.mean()
    sigma = returns.std()
    s = stats.skew(returns)
    k = stats.kurtosis(returns) # scipy returns 'excess' kurtosis by default
    z = stats.norm.ppf(level)
    
    # 1. Calculate the Cornish-Fisher Adjusted Z-score
    z_cf = (z + 
            (1/6) * (z**2 - 1) * s + 
            (1/24) * (z**3 - 3*z) * k - 
            (1/36) * (2*z**3 - 5*z) * s**2)
    
    # 3. Estimate ES
    # Note: Unlike Gaussian, there isn't a simple closed-form ES for CF.
    # We typically use the modified Z-score to find the tail expectations.
    # A common robust approach is to use the modified VaR in a Cornish-Fisher
    # specific density integration or simpler: use it to scale the Gaussian ES.
    
    gauss_es_z = -(stats.norm.pdf(z) / level)
    # Adjustment factor based on the ratio of CF-VaR to Gaussian-VaR
    cf_adjustment = z_cf / z
    
    return  mu + (sigma * gauss_es_z * cf_adjustment)