import numpy as np
import pandas as pd
import pytest


def test_semideviation():
    
    from toolkit.risk import semideviation

    # Test with a Series
    returns_series = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5])
    assert np.isclose(semideviation(returns_series), 0.1)

    # Test with a DataFrame
    returns_df = pd.DataFrame({
        'Asset1': [0.1, -0.2, 0.3, -0.4, 0.5],
        'Asset2': [-0.1, -0.3, 0.2, -0.5, 0.4]
    })
    expected_semideviation_df = pd.Series({
        'Asset1': 0.1,
        'Asset2': (0.08/3)**0.5
    })
    assert semideviation(returns_df).equals(expected_semideviation_df)

    # Test with invalid input
    with pytest.raises(TypeError):
        semideviation([0.1, -0.2, 0.3])  # Not a Series or DataFrame

def test_drawdown():
    from toolkit.risk import drawdown

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