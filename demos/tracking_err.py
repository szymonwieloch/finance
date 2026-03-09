import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tracking Error
    While professional ETF companies try to mimic indexes using full replication (bying exactly the whole market) it is also possible to aproximate market using partial replication. The first option provides tracking error around 0.01 - 0.005%, the second really depends on the selection of stocks and their number. In this basic example a small set of stocks belonging to S&P 500 arranged in an equally weighted portfolio is going to be compared to the S&P index itself to calculate tracking error.
    """)
    return


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import yfinance as yf
    import pandas as pd
    import seaborn as sns
    from toolkit import general
    import matplotlib.pyplot as plt

    return general, mo, plt, sns, yf


@app.cell
def _(yf):
    STOCKS = [
        "TXN",
        "ADBE",
        "COST",
        "PLTR",
        "PGR",
        "ES",
        "MAR",
        "AMAT",
        "ZTS",
        "GIS",
    ]
    SP500 = "^GSPC"
    portfolio = yf.download(STOCKS, period="1y")
    sp500 = yf.download(SP500, period="1y")
    return portfolio, sp500


@app.cell
def _(general, portfolio, sp500):
    port_rets = general.prices_to_returns(portfolio["Close"]).mean(axis=1)
    idx_rets = general.prices_to_returns(sp500["Close"]).iloc[:, 0]
    return idx_rets, port_rets


@app.cell
def _(general, idx_rets, port_rets):
    general.tracking_error(port_rets, idx_rets) * (252**0.5)
    return


@app.cell
def _(idx_rets, plt, port_rets, sns):
    plt.figure(figsize=(14, 8))
    sns.lineplot(port_rets, label="portfolio")
    sns.lineplot(idx_rets, label="S&P 500")
    plt.title("Comparison Of Returns")
    plt.ylabel("Returns")

    plt.gca()
    return


@app.cell
def _(general, idx_rets, port_rets):
    tracking_err = general.tracking_error(port_rets, idx_rets) * (252**0.5)
    return (tracking_err,)


@app.cell(hide_code=True)
def _(mo, tracking_err):
    mo.md(rf"""
    ## Result
    The final tracking error is {tracking_err * 100:.2f}%
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
