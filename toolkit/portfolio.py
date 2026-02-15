import pandas as pd
import numpy as np


def portfolio_return(weights:np.ndarray, returns:np.ndarray) -> np.ndarray:
    """Computes the return on a portfolio from constituent returns and weights

    Args:
        weights (np.ndarray): Portfolio weights as a numpy array or Nx1 matrix.
        returns (np.ndarray): Asset returns as a numpy array or Nx1 matrix.

    Returns:        np.ndarray: The return on the portfolio as a numpy array or 1x1 matrix.
    """
    return weights.T @ returns


def portfolio_std(weights, covmat):
    """Computes the volatility of a portfolio from a covariance matrix and constituent weights

    Args:
        weights (np.ndarray): Portfolio weights as a numpy array or Nx1 matrix.
        covmat (np.ndarray): Covariance matrix of asset returns as an NxN matrix.
    
    Returns:        np.ndarray: The volatility of the portfolio as a numpy array or 1x1 matrix.
    """
    return (weights.T @ covmat @ weights)**0.5


def risk_contribution(weights:pd.Series,cov: pd.DataFrame) -> pd.Series:
    """Computes the contributions to risk of the constituents of a portfolio

    Args:
        weights (pd.Series): Portfolio weights.
        cov (pd.DataFrame): Covariance matrix of asset returns.

    Returns:        pd.Series: Risk contributions of each asset in the portfolio.
    """
    total_portfolio_var = portfolio_std(weights,cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@weights
    return  np.multiply(marginal_contrib,weights.T)/total_portfolio_var