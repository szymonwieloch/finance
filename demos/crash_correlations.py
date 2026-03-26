import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import yfinance as yf

    return mo, plt, sns, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlations During Market Crash

    Investor use diversification in hope of reducing their risk. However during crashes, when investors need derising their investments the most, divesification start failing because of increased of correlation between all assets.

    The global market is too big to present the problem on all stocks, so instead a complete set of American sector ETFs is used as a reasonable approximation of the market behaviour.
    """)
    return


@app.cell
def _(yf):
    tickers = [ # all started in 1998
        "XLK", "XLF", "XLV", "XLE", "XLY", 
        "XLP", "XLI", "XLU", "XLB"
    ]

    # "XLC" "XLRE" were created later, we don't have enough data for them

    sp500_ticker = "^GSPC"

    df = yf.download(tickers, period='40y', interval='1wk')
    sp500 = yf.download(sp500_ticker, start=df.index[0], interval='1wk')
    return df, sp500


@app.cell
def _(df, sp500):
    sp500_rets = sp500['Close'].pct_change().dropna()
    rets = df['Close'].pct_change().dropna()
    rets
    return rets, sp500_rets


@app.cell
def _(rets, sp500_rets):
    rolling_window = 20 # 20 weeks
    avg = sp500_rets.rolling(window=rolling_window).mean().dropna()
    corr = rets.rolling(window=rolling_window).corr().groupby(level='Date').apply(lambda cormat: cormat.values.mean()).dropna()
    return avg, corr


@app.cell
def _(avg, corr, plt, sns):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    color1 = '#FF5733'
    sns.lineplot(avg, ax=ax1, palette=[color1], legend=False)
    ax1.set_ylabel('Market returns', color=color1)
    ax2 = ax1.twinx()
    color2='green'
    sns.lineplot(corr, ax=ax2, color=color2, legend=False)
    ax2.set_ylabel('Average correlation', color=color2)
    return


@app.cell
def _(avg, corr):
    crash_corr = avg.iloc[:,0].corr(corr)
    return (crash_corr,)


@app.cell(hide_code=True)
def _(crash_corr, mo):
    mo.md(rf"""
    # Summary

    During market crashes (negative returns) there is an abvious negative correlation around {crash_corr: .4f}, which means that when the market is goind down, all stocks in the market tend to go down together, while the growth is much more random.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
