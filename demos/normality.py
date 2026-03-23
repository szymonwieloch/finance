import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import import_fix as _
    import pandas as pd
    import scipy.stats as st
    import seaborn as sns
    import marimo as mo
    import yfinance as yf
    import matplotlib.pyplot as plt

    return mo, pd, plt, sns, st, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deviations From Normality

    Market returns are often said to follow normal distribution, but upon a deeper inspection it can be noticed, that they tend to be significantly skewed and have strong kurtosis. This demo shows basic stats related to measuring a realistic distribution of stocks.
    """)
    return


@app.cell
def _(yf):
    ticker = "^GSPC" # Define the ticker for the S&P 500 Index

    df = yf.download([ticker], period='10y')
    df
    return df, ticker


@app.cell
def _(df, ticker):
    rets = df['Close'][ticker].pct_change().dropna()
    return (rets,)


@app.cell
def _(plt, rets, sns):
    plt.figure(figsize=(14,6))
    plt.axvline(x=rets.mean(), color='red', linestyle='--', linewidth=2)
    sns.histplot(rets)
    return


@app.cell
def _(pd, rets, st):
    s = st.skew(rets)
    k = st.kurtosis(rets)
    _, p = st.jarque_bera(rets)
    pd.Series([s, k, p], index=['Skewness', 'Excessive Kurtosis', 'P-value'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summary

    S&P 500 daily returs are significantly skewed and have huge kurtosis. Also they do not pass the normality test with p-value being effectively zero (so 0% chance that this is a normal distribution). Those are all strong indicators, that models using Gaussian distribution may be making significant calculation errors.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
