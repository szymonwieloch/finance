from toolkit.general import returns_to_prices
import pandas as pd
import numpy as np

def test_returns_to_prices():
    # Test with a simple series of returns
    returns = pd.Series([0.05, 0.02, -0.01, 0.03], index=[7, 8, 9, 10])
    prices = returns_to_prices(returns, initial_price=100)
    expected_prices = pd.Series([100, 105, 107.1, 106.029, 109.20987], index=[6, 7, 8, 9, 10])
    assert np.allclose(prices.values, expected_prices.values, rtol=1e-3)
    assert prices.index.equals(expected_prices.index)

    returns_df = pd.DataFrame({
        'Asset1': returns,
        'Asset2': [0.1, 0.1, 0.1, 0.1]
    }, index=returns.index)
    expected_prices_df = pd.DataFrame({
        'Asset1': expected_prices,
        'Asset2': [100, 110, 121, 133.1, 146.41]
    }, index=expected_prices.index)
    prices_df = returns_to_prices(returns_df, initial_price=100)
    assert np.allclose(prices_df.values, expected_prices_df.values, rtol=1e-3)
    assert prices_df.index.equals(expected_prices_df.index)