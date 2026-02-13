import pandas as pd
import numpy as np
from .general import returns_to_prices

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