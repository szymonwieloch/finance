import pandas as pd

def returns_to_prices(returns:pd.Series|pd.DataFrame, initial_price=1.0) -> pd.Series|pd.DataFrame:
    """
    Convert a series of returns to a series of prices.

    Parameters:
    returns (pd.Series): A series of returns.
    initial_price (float): The initial price to start from.

    Returns:
    pd.Series|pd.DataFrame: A series of prices corresponding to the returns.
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