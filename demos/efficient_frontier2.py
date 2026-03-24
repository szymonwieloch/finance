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


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Efficient Frontier II

    Data from multiple stocks can be combined together to achieve optimal portfolio. Typically "optimal" is defined as having the greatest Sharp ratio.

    While this sounds great in theory, in practice the so-called Markowitz portfolio relies on knowing the covariance matrix and the expected returns. While covariance matrix is possible to calculate for small number of stocks, calculation of expected returns is a much harder task. At the same time the result of calculations is very sensitive to the input meaning that even small mistake in the estimation of expected returns results in a completely different portfolio. Therefore the Markowitz portfolio remains a very important theoretical but not practical model. Global minimum variance portfolio is another solution that offers a more practical outcome.

    This notebook presents application of the Markovits concepts to a portfolio that consists of Vanguard sector ETFs. Those typically have small correlation, but together cover the whole market. Because there are no easy ways to calculate expected returns, historical returns of ETFs were use with mean reversion to make them more realistic.
    """)
    return


@app.cell
def _(yf):
    # There are too many stocks for practical calculations, so instead of stocks use Vanguard sector ETFs
    vanguard_sectors = {
        "VOX": "Communication Services",
        "VCR": "Consumer Discretionary",
        "VDC": "Consumer Staples",
        "VDE": "Energy",
        "VFH": "Financials",
        "VHT": "Health Care",
        "VIS": "Industrials",
        "VGT": "Information Technology",
        "VAW": "Materials",
        "VNQ": "Real Estate",
        "VPU": "Utilities",
    }
    tickers = list(vanguard_sectors.keys())

    # Download historical data
    rets = (
        yf.download(tickers, start="2020-01-01", end="2025-01-01")["Close"]
        .pct_change()
        .dropna()
    )
    return rets, tickers


@app.cell
def _(rets):
    rets
    return


@app.cell
def _(np, pd, plt, portfolio, rets, sns, tickers):
    # this is cheating: it's difficult to obtain real expected returns
    # so the historical returns are used as a replacement
    exp_rets = (1 + rets).prod() ** 0.2 - 1
    # to make the values a bit more realistic, revert them to the mean by 50%
    exp_rets = 0.5 * exp_rets + 0.5 * exp_rets.mean()
    cov = rets.cov()

    points = 20
    target_rs = np.linspace(exp_rets.min(), exp_rets.max(), points)
    weights = [
        portfolio.minimize_vol(target_return, exp_rets, cov)
        for target_return in target_rs
    ]


    prets = [portfolio.portfolio_return(w, exp_rets) for w in weights]
    pvols = [portfolio.portfolio_std(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": prets, "Volatility": pvols})

    gmv = portfolio.gmv(cov)
    gmv_r, gmv_v = (
        portfolio.portfolio_return(gmv, exp_rets),
        portfolio.portfolio_std(gmv, cov),
    )

    ew = np.repeat(1 / len(tickers), len(tickers))
    ew_r, ew_v = (
        portfolio.portfolio_return(ew, exp_rets),
        portfolio.portfolio_std(ew, cov),
    )

    risk_free_rate = 0.0265  # averate in the period
    msr = portfolio.msr(risk_free_rate, exp_rets, cov)
    msr_r, msr_v = (
        portfolio.portfolio_return(msr, exp_rets),
        portfolio.portfolio_std(msr, cov),
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(ef, x="Volatility", y="Returns", orient="y")
    plt.plot(gmv_v, gmv_r, "ro", markersize=10, label="Minimum variance portfolio")
    plt.plot(ew_v, ew_r, "go", markersize=10, label="Equally weighted portfolio")
    plt.plot(msr_v, msr_r, "bo", markersize=10, label="Markowitz portfolio")
    plt.plot(0, risk_free_rate, "yo", markersize=10, label="Risk free portfolio")
    plt.plot([0, msr_v], [risk_free_rate, msr_r], linestyle="--", color="orange")
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
