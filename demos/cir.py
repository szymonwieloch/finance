import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from toolkit.other import cir

    return cir, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CIR — Cox-Ingersoll-Ross Interest Rate Model

    The **Cox-Ingersoll-Ross (CIR)** model is a classic stochastic model for interest rates.
    It describes the evolution of the short rate $r_t$ via the stochastic differential equation:

    $$dr_t = a (b - r_t) dt + \sigma \sqrt{r_t} dW_t$$

    where:

    - **$a$** — Mean-reversion speed. How quickly the rate reverts to the long-run mean.
    - **$b$** — Long-run mean rate. The level the rate tends toward over time.
    - **$\sigma$** — Volatility. Amplitude of random fluctuations.
    - **$r_0$** — Initial (starting) short rate.

    Key properties:

    - **Mean reversion**: Rates are pulled back toward $b$ — when $r_t$ is above $b$,
      the drift is negative, and vice versa.
    - **Non-negative rates**: The $\sqrt{r_t}$ term ensures rates stay non-negative
      (and strictly positive if $2ab > \sigma^2$, the Feller condition).
    - **Stationary distribution**: As $t \to \infty$, the rate follows a Gamma distribution.

    Use the sliders below to explore how different parameter values affect the simulated paths.
    """)
    return


@app.cell
def _(mo):
    # --- Parameter sliders ---
    n_years = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=10,
        label="Horizon (years)",
    )
    a = mo.ui.slider(
        start=0.01,
        stop=0.50,
        step=0.01,
        value=0.05,
        label="Mean-reversion speed (a)",
    )
    b = mo.ui.slider(
        start=0.01,
        stop=0.10,
        step=0.005,
        value=0.03,
        label="Long-run mean (b)",
    )
    sigma = mo.ui.slider(
        start=0.01,
        stop=0.20,
        step=0.01,
        value=0.05,
        label="Volatility (σ)",
    )
    r_0_slider = mo.ui.slider(
        start=0.01,
        stop=0.10,
        step=0.005,
        value=0.03,
        label="Initial rate (r₀)",
    )
    steps_per_year = mo.ui.slider(
        start=4,
        stop=52,
        step=4,
        value=12,
        label="Steps per year",
    )
    n_scenarios = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=50,
        label="Number of scenarios",
    )

    mo.md(f"""
    ## Parameters

    {a}  
    {b}  
    {sigma}  
    {r_0_slider}  
    {steps_per_year}  
    {n_years}  
    {n_scenarios}  
    """)
    return a, b, n_scenarios, n_years, r_0_slider, sigma, steps_per_year


@app.cell
def _(a, b, cir, n_scenarios, n_years, np, r_0_slider, sigma, steps_per_year):
    # --- Run CIR simulation ---
    rates_df = cir(
        n_years=n_years.value,
        n_scenarios=n_scenarios.value,
        a=a.value,
        b=b.value,
        sigma=sigma.value,
        steps_per_year=steps_per_year.value,
        r_0=r_0_slider.value,
    )

    # Build a time axis in years
    num_steps = int(n_years.value * steps_per_year.value) + 1
    t_axis = np.linspace(0, n_years.value, num_steps)

    # --- Terminal statistics ---
    terminal_rates = rates_df.iloc[-1].values
    mean_terminal = np.mean(terminal_rates)
    median_terminal = np.median(terminal_rates)
    std_terminal = np.std(terminal_rates)

    # Feller condition check
    feller_holds = 2 * a.value * b.value > sigma.value ** 2

    return (
        feller_holds,
        mean_terminal,
        median_terminal,
        rates_df,
        std_terminal,
        t_axis,
        terminal_rates,
    )


@app.cell
def _(
    a,
    b,
    feller_holds,
    mean_terminal,
    median_terminal,
    mo,
    sigma,
    std_terminal,
):
    feller_icon = "✅" if feller_holds else "⚠️"
    mo.md(rf"""
    ## Summary Statistics

    | Metric | Value |
    |---|---|
    | Mean terminal rate | {mean_terminal:.4f} ({mean_terminal*100:.2f}%) |
    | Median terminal rate | {median_terminal:.4f} ({median_terminal*100:.2f}%) |
    | Std of terminal rates | {std_terminal:.4f} ({std_terminal*100:.2f}%) |
    | Long-run mean (b) | {b.value:.3f} ({b.value*100:.1f}%) |
    | Feller condition ($2ab > \sigma^2$) | {feller_icon} ($2ab = {2 * a.value * b.value:.4f}$, $\sigma^2 = {sigma.value ** 2:.4f}$) |

    The **Feller condition** ensures the process stays strictly positive.
    When it holds ($2ab > \sigma^2$), zero is never reached.
    """)
    return


@app.cell
def _(a, b, n_scenarios, np, plt, rates_df, sigma, t_axis):
    # --- Plot: Sample trajectories ---
    n_plot = min(15, n_scenarios.value)
    sample_cols = np.random.choice(n_scenarios.value, size=n_plot, replace=False)

    fig_paths, ax_paths = plt.subplots(figsize=(14, 6))

    # Plot individual paths with low alpha
    for i in sample_cols:
        ax_paths.plot(
            t_axis,
            rates_df.iloc[:, i] * 100,
            alpha=0.35,
            linewidth=0.7,
            color="steelblue",
        )

    # Plot mean path
    mean_path = rates_df.mean(axis=1) * 100
    ax_paths.plot(
        t_axis,
        mean_path,
        color="darkred",
        linewidth=2.5,
        label="Mean across all scenarios",
    )

    # Long-run mean line
    ax_paths.axhline(
        y=b.value * 100,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Long-run mean b = {b.value * 100:.1f}%",
    )

    ax_paths.set_title(
        f"CIR — Sample Interest Rate Paths (a={a.value:.2f}, b={b.value*100:.1f}%, σ={sigma.value:.2f})",
        fontsize=14,
    )
    ax_paths.set_xlabel("Years")
    ax_paths.set_ylabel("Annual Interest Rate (%)")
    ax_paths.legend(loc="upper right")
    ax_paths.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(b, mean_terminal, median_terminal, n_years, plt, terminal_rates):
    # --- Plot: Histogram of terminal rates ---
    fig_hist, ax_hist = plt.subplots(figsize=(14, 6))

    ax_hist.hist(
        terminal_rates * 100,
        bins=40,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
        density=True,
    )
    ax_hist.axvline(
        x=b.value * 100,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Long-run mean b = {b.value * 100:.1f}%",
    )
    ax_hist.axvline(
        x=mean_terminal * 100,
        color="darkred",
        linestyle="-",
        linewidth=2,
        label=f"Mean terminal = {mean_terminal * 100:.2f}%",
    )
    ax_hist.axvline(
        x=median_terminal * 100,
        color="darkgreen",
        linestyle="-.",
        linewidth=2,
        label=f"Median terminal = {median_terminal * 100:.2f}%",
    )
    ax_hist.set_title(
        f"Distribution of Terminal Interest Rates (at t={n_years.value} years)",
        fontsize=14,
    )
    ax_hist.set_xlabel("Annual Interest Rate (%)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
