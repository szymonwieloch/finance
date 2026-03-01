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
    from scipy.stats import t
    import numpy as np


    return data, general, mo, np, pd, plt, risk, sns, t


@app.cell
def _():
    # Parametrize the experiment
    ticker = "^GSPC" # S&P 500
    years = 20
    level=0.01
    FIGSIZE=(16, 10)
    return FIGSIZE, level, ticker, years


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

    es_h = risk.es_historic(returns, level).iloc[0]
    es_g = risk.es_gaussian(returns, level).iloc[0]
    es_cf = risk.es_cornish_fisher(returns, level).iloc[0]
    return es_cf, es_g, es_h, var_cf, var_g, var_h


@app.cell
def _(pd, plt, sns, var_cf, var_g, var_h):
    var_df = pd.DataFrame({
        'Method': ['Historical', 'Gaussian', 'Cornish-Fisher'],
        'Daily VaR [%]': [var_h*100, var_g*100, var_cf*100]
    })
    ax_vd = sns.barplot(data=var_df, x='Method', y='Daily VaR [%]')
    for container_vd in ax_vd.containers:
        ax_vd.bar_label(container_vd, fmt='%.2f')
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
    ax = sns.barplot(data=es_df, x='Method', y='Daily ES [%]')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.title("Daily ES using differnt methods of calculation")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis
    Clearly the Cornish-Fisher provides greater estimates of VaR and ES compared to the Gaussian method. Value of the historical method cannot be fully trusted as it is known to depend on the selected sample, but it roughly matches calculated values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Yearly VaR And ES

    We start by using Monte Carlo simulation with student-t distribution to estimate VaR and ES and draw few first generated returns as a sanity check.
    """)
    return


@app.cell
def _(FIGSIZE, general, level, np, pd, returns, t):
    # Monte Carlo simulation using student-t distribution with mean and variance scaled to S&P 500
    #, but with thicker tails of the distribution
    days = 252
    nu = 4                      # Degrees of freedom (fat tails)
    simulations=10000
    mu_daily = (returns+1).prod()**(1/len(returns))-1
    sigma_daily = returns.std()
    # Adjust scale because Var(T) = scale^2 * (nu / (nu - 2))
    scale_daily = sigma_daily / np.sqrt(nu / (nu - 2))

    # 3. Generate Returns using Scipy
    # stats.t.rvs returns random variables from the Student-t distribution
    daily_returns = t.rvs(df=nu, loc=mu_daily, scale=scale_daily, size=(simulations, days))
    dr = pd.DataFrame(daily_returns.T)
    rets = (dr + 1).prod()-1
    var_ymc = -np.percentile(rets, level*100)
    es_ymc = -rets[rets<-var_ymc].mean()

    mc_chart = general.returns_to_prices(dr.iloc[:, :5])
    mc_chart.plot(title="Few generated annual flows using student-t distribution", figsize=FIGSIZE)
    return days, es_ymc, var_ymc


@app.cell
def _(days, level, np, returns, risk):
    var_yh = float(risk.var_historic(returns, level=level, samples_in_period=days).iloc[0])
    var_ycf = float(risk.var_cornish_fisher(returns, level=level, samples_in_period=days).iloc[0])
    var_yg= float(risk.var_gaussian(returns, level=level, samples_in_period=days).iloc[0])
    es_yh = float(risk.es_historic(returns, level=level, samples_in_period=days).iloc[0])
    es_ycf = float(risk.es_cornish_fisher(returns, level=level, samples_in_period=days).iloc[0])
    es_yg = float(risk.es_gaussian(returns, level=level, samples_in_period=days).iloc[0])
    var_ycfe = 1-np.exp(-var_ycf)
    var_yge = 1-np.exp(-var_yg)
    return es_ycf, es_yg, es_yh, var_ycf, var_ycfe, var_yg, var_yge, var_yh


@app.cell
def _(
    FIGSIZE,
    pd,
    plt,
    sns,
    var_ycf,
    var_ycfe,
    var_yg,
    var_yge,
    var_yh,
    var_ymc,
):
    var_ydf = pd.DataFrame({
        'Method': ['Historical', 'Gaussian', 'Gaussian (Exp)', 'Cornish-Fisher', 'Cornish-Fisher (Exp)', 'Monte Carlo'],
        'Annual VaR [%]': [var_yh*100, var_yg*100, var_yge*100, var_ycf*100, var_ycfe*100, var_ymc*100]
    })
    plt.figure(figsize=FIGSIZE)
    ax_vy = sns.barplot(data=var_ydf, x='Method', y='Annual VaR [%]')
    for container_vy in ax_vy.containers:
        ax_vy.bar_label(container_vy, fmt='%.2f')
    plt.title("Annual VaR using differnt methods of calculation")

    plt.gca()
    return


@app.cell
def _(es_ycf, es_yg, es_yh, es_ymc, pd, plt, sns):
    es_ydf = pd.DataFrame({
        'Method': ['Historical', 'Gaussian', 'Cornish-Fisher', 'Monte Carlo'],
        'Annual ES [%]': [es_yh*100, es_yg*100, es_ycf*100, es_ymc*100]
    })
    ax_ey = sns.barplot(data=es_ydf, x='Method', y='Annual ES [%]')
    for container_ey in ax_ey.containers:
        ax_ey.bar_label(container_ey, fmt='%.2f')
    plt.title("Annual ES using differnt methods of calculation")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis

    Results contain few interesting points:

    - Cornish-Fisher method predicts almost 100% VaR and more than 100% ES. That is obviously not possible.
      The reason behind it is the exponential nature of returns, while the standard C-F method returns values in a linear fashion. After correcting VaR, results better match expectations and historical value.
      For ES unfortunately there is no good method.
    - Gaussian method predicts greater VaR than Monte Carlo simulation with student-t distribution, which does make much sense, as student-t is known to have "fatter" tails. This is again caused by the exponential nature of returns. After rescaling to the exponential value, M-C method gives slightly greater value.
    - Because ES Gaussian and Cornish-Fisher methods do no provide easy method to switch from a linear scale to the exponential scale, they work very well with daily returns, where the relative changes are small, but result in significant estimation errors for longer periods.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
