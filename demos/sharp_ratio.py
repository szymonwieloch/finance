import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import yfinance as yf

    return mo, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Risk Adjusted Returns

    The Sharp ratio is known as the main method for comparing two investments or judging effectiveness of the given portfolio manager. In this demo two ETFs reflecting S&P 500 market are going to be compared in order to calculate their relative performance. This calculation also takes into account different fees and costs of transactions.
    """)
    return


@app.cell
def _(yf):
    market_cap = "VOO"  # Vanguard S&P 500 ETF
    eq_weighted = "RSP"  # Invesco S&P 500 Equal Weight ETF
    risk_free = "^IRX"  # 13-week Treasury Bill
    # last 10 years of data
    start_date = "2016-03-22"
    end_date = "2026-03-22"
    df = yf.download(
        [market_cap, eq_weighted, risk_free], start=start_date, end=end_date
    )
    df
    return df, eq_weighted, market_cap, risk_free


@app.cell
def _(df, risk_free):
    daily_rf = (df["Close"][risk_free] / (100 * 252)).ffill()
    daily_excess_rets = (
        df[("Close")]
        .drop(columns=[risk_free])
        .pct_change()
        .dropna()
        .sub(daily_rf, axis=0)
    )
    return (daily_excess_rets,)


@app.cell
def _(daily_excess_rets):
    annualysed_exc_rets = ((1 + daily_excess_rets).prod() ** 0.1) - 1
    annualysed_vol = daily_excess_rets.std() * (252**0.5)
    sharp_ratio = annualysed_exc_rets / annualysed_vol
    sharp_ratio
    return (sharp_ratio,)


@app.cell(hide_code=True)
def _(eq_weighted, market_cap, mo, sharp_ratio):
    mo.md(rf"""
    ## Summary
    During the last 10 years investment in the cap-weighted ETF was a significantly better option with sharp ratio {sharp_ratio[market_cap]:.2f} compared to the equaly weighted ETF {sharp_ratio[eq_weighted]:.2f}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
