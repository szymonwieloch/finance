import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import pandas as pd
    import yfinance as yf
    import datetime as dt
    import matplotlib.pyplot as plt
    import numpy as np
    from toolkit import general

    return dt, general, mo, np, plt, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Annualization

    The two most commonly use attributes of stocks are return and volatility (standard deviation).
    Typically exchanges provide daily OHLC prices, which then need to annualized in order to be able to compare them with other stocks or other assets.

    There are several methods for calculating annalyzed values, ranging from a quick and dirty estimates to very precise, but also mathematically complicated equations.
    """)
    return


@app.cell
def _(dt, yf):
    today = dt.date.today()
    data = yf.download("SPY", period='10y', interval='1d')
    data
    return (data,)


@app.cell
def _(data, plt):
    prices = data[("Close", "SPY")]
    prices.plot(figsize=(12, 6))
    plt.gca()
    return (prices,)


@app.cell
def _(general, np, prices):
    returns = general.prices_to_returns(prices)
    log_returns = np.log(1+returns)
    total_return = (1+returns).prod() - 1
    days_in_year = 252 # business days

    # simplified annualization
    annual_ret_simple = returns.mean() * days_in_year
    annual_vol_simple = returns.std() * (days_in_year**0.5)

    # logarithmic versions
    annual_ret_log = log_returns.sum() / 10
    annual_vol_log = log_returns.std() * ((len(returns)/10) ** 0.5)

    # more precise version that takes into account non-linear character of returns.
    annual_ret_prec = ((1+returns).prod() ** 0.1) -1

    # convert variance from the logarithmic scale back to the linear scale
    # this generally causes "widening" of volatility
    term1 = np.exp(annual_vol_log**2) - 1
    term2 = np.exp(2 * annual_ret_log + annual_vol_log**2)
    annual_vol_prec = (term1 * term2)**0.5
    return (
        annual_ret_log,
        annual_ret_prec,
        annual_ret_simple,
        annual_vol_log,
        annual_vol_prec,
        annual_vol_simple,
    )


@app.cell(hide_code=True)
def _(
    annual_ret_log,
    annual_ret_prec,
    annual_ret_simple,
    annual_vol_log,
    annual_vol_prec,
    annual_vol_simple,
    mo,
):
    mo.md(rf"""
    ## Results

    |                   | Simplified          | Logarithmic      | Precise           |
    | ---------------   | ------------------- | ---------------- | ----------------- |
    | Annual return     | {annual_ret_simple} | {annual_ret_log} | {annual_ret_prec} |
    | Annual volatility | {annual_vol_simple} | {annual_vol_log} | {annual_vol_prec}  |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
