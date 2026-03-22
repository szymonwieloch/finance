import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import pandas as pd
    import yfinance as yf
    import toolkit.risk as risk
    import matplotlib.pyplot as plt
    import seaborn as sns

    return mo, pd, plt, risk, sns, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Drawdown

    Drawdown is a measure of risk, which is very intuitive, but extreamly diffucult to calculate reliably as it suffers from sample selection. Maximum drawdown is commony used for historical analysis to present maximum amount of money you could have lost in the absolutely worst case while investing in the given period.
    """)
    return


@app.cell
def _(yf):

    ticker = "^GSPC" # Define the ticker for the S&P 500 Index

    df = yf.download([ticker], period='10y')
    df
    return df, ticker


@app.cell
def _(df, risk):
    rets = df['Close'].pct_change().dropna()
    drawdown = risk.drawdown(rets)
    max_drawdown = float(-drawdown.min().iloc[0])
    max_drawdown_date = drawdown.idxmin().iloc[0]
    return drawdown, max_drawdown, max_drawdown_date


@app.cell
def _(drawdown, plt, sns):
    plt.figure(figsize=(14,8))
    plt.title('Drawdown')
    sns.lineplot(drawdown, legend=None)
    return


@app.cell
def _(df):
    cummax = df['Close'].cummax()
    return (cummax,)


@app.cell
def _(cummax, df, pd, ticker):
    data = pd.DataFrame({
        'Previous max': cummax[ticker],
        'S&P 500': df['Close'][ticker],
    
    }, index = cummax.index)
    return (data,)


@app.cell
def _(data, plt, sns):
    plt.figure(figsize=(14,8))
    plt.title('Difference between S&P 500 value and the previous greatest value')
    sns.lineplot(data)
    plt.fill_between(data.index, data['Previous max'], data['S&P 500'], color='red', alpha=0.4)
    return


@app.cell(hide_code=True)
def _(max_drawdown, max_drawdown_date, mo):
    mo.md(rf"""
    ## Summary

    In the last 10 years the maximum observed drawdown was {100*max_drawdown: .2f}% that took place on {max_drawdown_date.date()}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
