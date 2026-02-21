import numpy as np
import pandas as pd
import pytest

from toolkit.risk import var_gaussian, var_cornish_fisher, es_gaussian, es_historic, var_historic, semideviation, drawdown


def test_semideviation():
    # Test with a Series
    returns_series = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5])
    assert np.isclose(semideviation(returns_series), 0.02**0.5)

    # Test with a DataFrame
    returns_df = pd.DataFrame({
        'Asset1': [0.1, -0.2, 0.3, -0.4, 0.5],
        'Asset2': [-0.1, -0.3, 0.2, -0.5, 0.4]
    })
    expected_semideviation_df = pd.Series({
        'Asset1': 0.02**0.5,
        'Asset2': (0.08/2)**0.5
    })

    print (expected_semideviation_df)
    print (semideviation(returns_df))
    assert np.allclose(semideviation(returns_df), expected_semideviation_df, equal_nan=True)

    # Test with invalid input
    with pytest.raises(TypeError):
        semideviation([0.1, -0.2, 0.3])  # Not a Series or DataFrame

def test_drawdown():
    returns = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5])
    expected_drawdown = pd.Series([0.0, 0.0, -0.2, 0.0, -0.4, -0.1])
    assert np.isclose(drawdown(returns), expected_drawdown).all()

    returns_df = pd.DataFrame({
        'Asset1': [0.1, -0.2, 0.3, -0.4, 0.5],
        'Asset2': [-0.1, -0.3, 0.0, 0.6, -0.3]
    })
    expected_drawdown_df = pd.DataFrame({
        'Asset1': [0.0, 0.0, -0.2, 0.0, -0.4, -0.1],
        'Asset2': [0.0, -0.1, -0.37, -0.37, -0.0, -0.3]
    })

    assert np.isclose(drawdown(returns_df), expected_drawdown_df).all().all()


def test_es_gaussian():
    from toolkit.risk import es_gaussian

    np.random.seed(42)
    mu = 0.2
    sigma = 0.01
    level = 0.05
    test_returns = np.random.normal(mu, sigma, 1000000)
    
    calculated_es = es_gaussian(test_returns, level=level)
    
    # calculate empirical ES for comparison
    var_threshold = np.percentile(test_returns, level * 100)
    empirical_es = -test_returns[test_returns <= var_threshold].mean()
    assert np.isclose(calculated_es, empirical_es, rtol=1e-2)

def test_var_gaussian():
    from toolkit.risk import var_gaussian

    np.random.seed(42)
    mu = 0.2
    sigma = 0.01
    level = 0.05
    test_returns = np.random.normal(mu, sigma, 1000000)
    
    calculated_var = var_gaussian(test_returns, level=level)
    
    # calculate empirical VaR for comparison
    empirical_var = -np.percentile(test_returns, level * 100)
    assert np.isclose(calculated_var, empirical_var, rtol=1e-2)

def test_var_historic():
    from toolkit.risk import var_historic

    np.random.seed(42)
    test_returns = pd.Series(np.random.normal(0.2, 0.01, 1000000))
    level = 0.05
    
    calculated_var = var_historic(test_returns, level=level)
    
    empirical_var = -np.percentile(test_returns, level * 100)
    assert np.isclose(calculated_var, empirical_var, rtol=1e-2)

def test_es_historic():
    from toolkit.risk import es_historic

    np.random.seed(42)
    test_returns = pd.Series(np.random.normal(0.2, 0.01, 1000000))
    level = 0.05
    
    calculated_es = es_historic(test_returns, level=level)
    
    var_threshold = np.percentile(test_returns, level * 100)
    empirical_es = -test_returns[test_returns <= var_threshold].mean()
    assert np.isclose(calculated_es, empirical_es, rtol=1e-2)


from toolkit.risk import var_cornish_fisher


def test_var_cornish_fisher_normal_approximation():
    """
    With near-normal data, Cornish-Fisher should be close to 
    standard Gaussian VaR. 1.645 is the 95% Z-score.
    """
    
    np.random.seed(42)
    normal_returns = np.random.normal(0, 1, 100000)
    calculated_var = var_cornish_fisher(normal_returns, 0.05)
    # Expected is roughly 1.645
    assert np.isclose(calculated_var, 1.645, rtol=1e-2)

def test_confidence_levels():
    """Lower level should result in a higher VaR."""
    np.random.seed(42)
    normal_returns = np.random.normal(0, 1, 100000)
    var_95 = var_cornish_fisher(normal_returns, 0.05)
    var_99 = var_cornish_fisher(normal_returns, 0.01)
    assert var_99 > var_95

def test_skewness_impact():
    """Negatively skewed data should increase the VaR (higher risk)."""
    skewed_returns = np.array([-0.05, -0.04, -0.03, 0.01, 0.02, 0.02])
    # Standard VaR would be lower than CF VaR here due to negative skew
    cf_var = var_cornish_fisher(skewed_returns, 0.05)
    var = var_gaussian(skewed_returns, 0.05)
    assert cf_var > var