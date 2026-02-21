# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Value At Risk Comparison

    There are several different methods for calculating Value At Risk and related metric Estimated Shortfall:

    - Historical (difficult for long periods as you typically don't have enough samples, suffers from sample selection error)
    - Gaussian (assumes normal distribution of losses)
    - Cornish-Fisher (like gaussian, but correct the calculation by skewnes and kurtosis)
    - Monte Carlo simulation - the industry standard. Depending on simulation parameters, either Students-T or GARCH-T. This demo concentrates on a longer simulation period where Students-T is the appropriate choice.

    Different methods provide different precision, with Cornish-Fisher and Monte Carlo simulation being the two most commonly used in professional finance. This demonstration is supposed to compare different methods and estimate roughly methodological error caused by using the similified models. S&P index is going to be analyzed using differnt models in the period of last 20 years.
    """)
    return


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    from toolkit import risk
    from toolkit import data
    from toolkit import general
    import pandas as pd


    return data, general, mo, pd, plt, risk, sns


@app.cell
def _():
    # Parametrize the experiment
    ticker = "^GSPC" # S&P 500
    years = 20
    level=0.05
    FIGSIZE=(16, 10)
    return level, ticker, years


@app.cell(hide_code=True)
def _(level, mo, ticker, years):
    mo.md(rf"""
    ## Input Data

    The last {years} years of returns for the {ticker} ticker is use, we are looking at {level*100}% of the worst cases.
    """)
    return


@app.cell
def _(data, ticker, years):
    # Get and present data
    sp500 = data.get_stock_data(ticker, years=years)
    sp500
    return (sp500,)


@app.cell
def _(sp500):
    plot_data = sp500['Close']
    plot_data.plot.line(title="S&P 500", figsize=(16, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Daily VaR
    """)
    return


@app.cell
def _(general, sp500):
    returns = general.prices_to_returns(sp500['Close'])
    return (returns,)


@app.cell
def _(level, returns, risk):
    var_h = risk.var_historic(returns, level).iloc[0]
    var_g = risk.var_gaussian(returns, level).iloc[0]
    var_cf = risk.var_cornish_fisher(returns, level).iloc[0]

    es_h = risk.es_historic(returns).iloc[0]
    es_g = risk.es_gaussian(returns).iloc[0]
    es_cf = risk.es_cornish_fisher(returns).iloc[0]
    return es_cf, es_g, es_h, var_cf, var_g, var_h


@app.cell
def _(pd, plt, sns, var_cf, var_g, var_h):
    var_df = pd.DataFrame({
        'Method': ['Historical', 'Gaussian', 'Cornish-Fisher'],
        'Daily VaR [%]': [var_h*100, var_g*100, var_cf*100]
    })
    sns.barplot(data=var_df, x='Method', y='Daily VaR [%]')
    plt.title("Daily VaR using differnt methods of calculation")
    #plt.figure(figsize=FIGSIZE)
    plt.gca()
    return


@app.cell
def _(es_cf, es_g, es_h, pd, plt, sns):
    es_df = pd.DataFrame({
        'Method': ['Historical', 'Gaussian', 'Cornish-Fisher'],
        'Daily ES [%]': [es_h*100, es_g*100, es_cf*100]
    })
    sns.barplot(data=es_df, x='Method', y='Daily ES [%]')
    plt.title("Daily ES using differnt methods of calculation")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Yearly VaR And ES

    And Monte Carlo simulation using Students-T distribution.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
