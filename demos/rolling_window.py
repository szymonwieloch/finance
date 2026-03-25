import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import seaborn as sns

    return mo, pd, plt, sns, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rolling Window
    Rolling window is a common way to present price changes, but without short term volatility. Typically it's an average calculated over a short period of time. Thanks to eliminating short term volatility the general trends become move visible.
    """)
    return


@app.cell
def _(yf):
    ticker = "^GSPC" # Define the ticker for the S&P 500 Index

    df = yf.download([ticker], period='10y')
    df
    return df, ticker


@app.cell
def _(df, pd, ticker):
    window_size = 50
    daily = df['Close']
    avg = daily.rolling(window=window_size).mean().shift(-window_size//2)
    data = pd.DataFrame({
        'daily': daily[ticker],
        'window': avg[ticker]
    })
    daily_r = daily.pct_change(fill_method=None)
    avg_r = daily.pct_change(window_size, fill_method=None).shift(-window_size//2)/(window_size**0.5) #scale
    data_r = pd.DataFrame({
        'daily': daily_r[ticker],
        'window': avg_r[ticker]
    })
    return data, data_r


@app.cell
def _(data, plt, sns):
    plt.figure(figsize=(14,7))
    plt.title('Prices')
    sns.lineplot(data, palette="tab10", dashes=False)
    return


@app.cell
def _(data_r, plt, sns):
    plt.figure(figsize=(14,7))
    plt.title('Returns')
    sns.lineplot(data_r, palette="tab10", dashes=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
