import pandas as pd
import numpy as np
from scipy.optimize import minimize


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


def msr(riskfree_rate, er, cov) -> pd.Series:
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_std(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    
    weights = minimize(portfolio_std, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_std, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x