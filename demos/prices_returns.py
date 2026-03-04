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
    from toolkit import general

    return dt, general, mo, plt, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From Prices To Returns

    Traditionally exchanges report daily OHLC prices, which can later be translated back into daily returns. Typically returns are expressed as percentage change or as a fraction of change. Prices can be convereted into returns and returns into prices.
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
def _(general, prices):
    returns = general.prices_to_returns(prices)
    return (returns,)


@app.cell
def _(returns):
    returns.plot(figsize=(12,6))
    return


@app.cell
def _(general, returns):
    recreated_prices = general.returns_to_prices(returns)
    return (recreated_prices,)


@app.cell
def _(recreated_prices):
    recreated_prices.plot(figsize=(12, 6))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
