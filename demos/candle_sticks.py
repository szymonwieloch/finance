import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from toolkit import general
    import mplfinance as mpf
    import matplotlib.pyplot as plt

    return mo, mpf, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resampling And Candle Sticks Chart

    This demo present creation a basic candle stick chart characteristic to finance with data resampled from 1 day input to a weekly interval.
    """)
    return


@app.cell
def _(yf):
    df = yf.download('TSLA', period='2y', interval='1d')
    df.columns = df.columns.droplevel(1)
    df
    return (df,)


@app.cell
def _(df):
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return (weekly_df,)


@app.cell
def _(weekly_df):
    weekly_df
    return


@app.cell
def _(mpf, weekly_df):
    mpf.plot(weekly_df, 
             type='candle', 
             style='charles', 
             title=f"TSLA Price Action",
             ylabel='Price ($)',
             volume=True,
             tight_layout=True,
             figsize=(12,6))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
