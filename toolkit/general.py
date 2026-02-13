import pandas as pd

def returns_to_prices(returns: pd.Series | pd.DataFrame, initial_price=1.0) -> pd.Series | pd.DataFrame:
    """
    Convert a series of returns to a series of prices.

    The function attempts to recreate a valid index for the resulting prices
    and prepends a zero-return row so that cumulative product produces the
    correct price series starting from `initial_price`.

    Args:
        returns (pd.Series or pd.DataFrame): A series or DataFrame of returns.
        initial_price (float): The initial price to start from.

    Returns:
        pd.Series or pd.DataFrame: A series or DataFrame of prices corresponding
            to the provided returns.
    """
    # try to recreate valid index in the result
    index_fill = None
    try:
        index_fill = [returns.index[0] - (returns.index[1] - returns.index[0])]
    except:
        pass
    zeros = pd.DataFrame({col: [0] for col in returns.columns}, index=index_fill) if isinstance(returns, pd.DataFrame) else pd.Series([0], index=index_fill)

    # 2. Stack the new row on top
    extended = pd.concat([zeros, returns], ignore_index=(index_fill is None))
    extended +=1
    result = extended.cumprod()
    result *= initial_price
    return result