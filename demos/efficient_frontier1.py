import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from toolkit import portfolio

    return mo, np, pd, plt, portfolio, sns, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Efficient Frontier I

    Data from two stocks can be combined together to present a simple fact that combining two weakly correlated stocks in a portfolio reduces total volatility.
    """)
    return


@app.cell
def _(yf):
    # Choose two stocks from different sectors (often weakly correlated)
    tickers = ["XOM", "IBM"]  # IBM (tech) and Exxon Mobil Corp (energy)

    # Download historical data
    rets = (
        yf.download(tickers, start="2020-01-01", end="2025-01-01")["Close"]
        .pct_change()
        .dropna()
    )
    return (rets,)


@app.cell
def _(rets):
    rets
    return


@app.cell
def _(np, pd, plt, portfolio, rets, sns):
    # this is cheating: it's difficult to obtain real expected returns
    # so the historical returns are used as a replacement
    exp_rets = (1 + rets).prod() ** 0.2 - 1
    cov = rets.cov()

    points = 20
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, points)]


    prets = [portfolio.portfolio_return(w, exp_rets) for w in weights]
    pvols = [portfolio.portfolio_std(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": prets, "Volatility": pvols})

    gmv = portfolio.gmv(cov)
    gmv_r, gmv_v = (
        portfolio.portfolio_return(gmv, exp_rets),
        portfolio.portfolio_std(gmv, cov),
    )

    ew = np.array([0.5, 0.5])
    ew_r, ew_v = (
        portfolio.portfolio_return(ew, exp_rets),
        portfolio.portfolio_std(ew, cov),
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(ef, x="Volatility", y="Returns", orient="y")
    plt.plot(gmv_v, gmv_r, "ro", markersize=10, label="Minimum variance portfolio")
    plt.plot(ew_v, ew_r, "go", markersize=10, label="Equally weighted portfolio")
    plt.legend()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
