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
    from toolkit.other import gbm_returns

    return gbm_returns, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # CPPI — Constant Proportion Portfolio Insurance

    CPPI is a dynamic portfolio insurance strategy that allocates capital between
    a **risky asset** (e.g., stocks) and a **safe asset** (e.g., bonds or cash).
    The goal is to protect a predefined **floor** (minimum acceptable portfolio value)
    while still participating in the upside of the risky asset.

    **How it works:**

    - **Cushion** = Portfolio Value − Floor
    - **Exposure to risky asset** = Multiplier × Cushion (capped at total portfolio value)
    - **Safe allocation** = Portfolio Value − Exposure

    When markets rise, the cushion grows and the strategy increases exposure to the risky asset
    (pro-cyclical). When markets fall, it cuts exposure to protect the floor.
    A higher **multiplier** means more aggressive participation but also higher risk of
    "gap risk" — when the portfolio crashes through the floor in a single period.

    Use the sliders below to explore how different parameters affect the strategy's behavior.
    """)
    return


@app.cell
def _(mo):
    # --- Parameter sliders ---
    multiplier = mo.ui.slider(
        start=1,
        stop=15,
        step=1,
        value=5,
    )
    floor_pct = mo.ui.slider(
        start=50,
        stop=95,
        step=5,
        value=80,
    )
    mu = mo.ui.slider(
        start=0.01,
        stop=0.20,
        step=0.01,
        value=0.07,
    )
    sigma = mo.ui.slider(
        start=0.05,
        stop=0.50,
        step=0.05,
        value=0.15,
    )
    rf = mo.ui.slider(
        start=0.0,
        stop=0.10,
        step=0.005,
        value=0.02,
    )
    years = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=10,
    )
    scenarios = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1000,
    )

    mo.md(f"""
    ## Parameters

    Multiplier (m)        {multiplier}  
    Floor (% of initial)  {floor_pct}  
    Risky return (μ)      {mu}  
    Risky volatility (σ)  {sigma}  
    Risk-free rate        {rf}  
    Horizon (years)       {years}  
    Scenarios             {scenarios}  
    """)
    return floor_pct, mu, multiplier, rf, scenarios, sigma, years


@app.cell
def _(
    floor_pct,
    gbm_returns,
    mu,
    multiplier,
    np,
    pd,
    rf,
    scenarios,
    sigma,
    years,
):
    # --- CPPI Simulation ---
    steps_per_year = 12
    steps = int(years.value * steps_per_year) + 1
    dt = 1.0 / steps_per_year

    # Generate risky asset returns
    risky_rets = gbm_returns(
        years=years.value,
        scenarios=scenarios.value,
        mu=mu.value,
        sigma=sigma.value,
        steps_per_year=steps_per_year,
    )  # shape: (steps, scenarios), first row is zero

    # Risk-free rate per period
    rf_period = (1 + rf.value) ** dt - 1

    # Floor: constant in real terms (no growth), just the initial floor
    floor_initial = floor_pct.value / 100.0
    initial_value = 1.0

    # Run CPPI for all scenarios at once
    portfolio = np.ones((steps, scenarios.value)) * initial_value
    floor = np.ones((steps, scenarios.value)) * floor_initial
    risky_exposure = np.zeros((steps, scenarios.value))

    for t in range(steps - 1):
        cushion = np.maximum(portfolio[t] - floor[t], 0.0)
        # Exposure to risky asset = m * cushion, capped at portfolio value
        risky_exposure[t] = np.minimum(multiplier.value * cushion, portfolio[t])
        safe_alloc = portfolio[t] - risky_exposure[t]

        # Next period portfolio value
        portfolio[t + 1] = safe_alloc * (1 + rf_period) + risky_exposure[t] * (
            1 + risky_rets[t + 1]
        )
        floor[t + 1] = floor_initial

    # Last period allocation
    cushion = np.maximum(portfolio[-1] - floor[-1], 0.0)
    risky_exposure[-1] = np.minimum(multiplier.value * cushion, portfolio[-1])
    safe_alloc = portfolio[-1] - risky_exposure[-1]

    # Build DataFrames for easier plotting
    portfolio_df = pd.DataFrame(portfolio)
    risky_pct_df = pd.DataFrame(risky_exposure / portfolio * 100)

    # --- Statistics ---
    final_vals = portfolio[-1]
    below_floor = (final_vals < floor_initial).mean() * 100
    mean_final = final_vals.mean()
    median_final = np.median(final_vals)
    risk_free_final = (1 + rf.value) ** years.value
    return (
        below_floor,
        final_vals,
        floor_initial,
        mean_final,
        median_final,
        portfolio,
        risk_free_final,
        risky_pct_df,
        steps,
    )


@app.cell
def _(
    below_floor,
    floor_pct,
    mean_final,
    median_final,
    mo,
    multiplier,
    risk_free_final,
):
    mo.md(rf"""
    ## Results

    With multiplier **{multiplier.value}×** and floor at **{floor_pct.value}%** of initial capital:

    | Metric | Value |
    |---|---|
    | Mean final portfolio | {mean_final:.3f}× initial |
    | Median final portfolio | {median_final:.3f}× initial |
    | Risk-free accumulation | {risk_free_final:.3f}× initial |
    | Scenarios below floor | **{below_floor:.1f}%** |
    """)
    return


@app.cell
def _(floor_initial, floor_pct, np, plt, portfolio, scenarios, steps, years):
    # --- Plot: Sample trajectories ---
    sample_cols = np.random.choice(
        min(scenarios.value, scenarios.value),
        size=min(10, scenarios.value),
        replace=False,
    )
    sample_paths = portfolio[:, sample_cols]

    fig_paths, ax_paths = plt.subplots(figsize=(14, 6))
    t_axis = np.linspace(0, years.value, steps)

    # Plot individual paths with low alpha
    for i in range(sample_paths.shape[1]):
        ax_paths.plot(
            t_axis,
            sample_paths[:, i],
            alpha=0.4,
            linewidth=0.8,
            color="steelblue",
        )

    # Plot mean path
    mean_path = portfolio.mean(axis=1)
    ax_paths.plot(
        t_axis,
        mean_path,
        color="darkred",
        linewidth=2.5,
        label="Mean across all scenarios",
    )

    # Floor line
    ax_paths.axhline(
        y=floor_initial,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Floor ({floor_pct.value}%)",
    )

    ax_paths.set_title("CPPI — Sample Portfolio Trajectories", fontsize=14)
    ax_paths.set_xlabel("Years")
    ax_paths.set_ylabel("Portfolio Value (× initial)")
    ax_paths.legend(loc="upper left")
    ax_paths.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(final_vals, floor_initial, plt, risk_free_final):
    # --- Plot: Histogram of final outcomes ---
    fig_hist, ax_hist = plt.subplots(figsize=(14, 6))

    ax_hist.hist(
        final_vals,
        bins=80,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
        density=True,
    )
    ax_hist.axvline(
        x=floor_initial,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Floor ({floor_initial:.2f})",
    )
    ax_hist.axvline(
        x=risk_free_final,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Risk-free ({risk_free_final:.2f})",
    )
    ax_hist.axvline(
        x=final_vals.mean(),
        color="darkred",
        linestyle="-",
        linewidth=2,
        label=f"Mean ({final_vals.mean():.2f})",
    )
    ax_hist.set_title("Distribution of Final Portfolio Values", fontsize=14)
    ax_hist.set_xlabel("Final Value (× initial)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(below_floor, final_vals, floor_initial, np, plt):
    # --- Plot: Cumulative distribution of final outcomes ---
    fig_cum, ax_cum = plt.subplots(figsize=(14, 6))

    sorted_vals = np.sort(final_vals)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax_cum.plot(sorted_vals, cumulative * 100, color="steelblue", linewidth=2)
    ax_cum.axvline(
        x=floor_initial,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Floor ({floor_initial:.2f})",
    )
    ax_cum.axhline(
        y=below_floor,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"P(below floor) = {below_floor:.1f}%",
    )
    ax_cum.set_title("Cumulative Distribution of Final Outcomes", fontsize=14)
    ax_cum.set_xlabel("Final Value (× initial)")
    ax_cum.set_ylabel("Cumulative Probability (%)")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(np, plt, portfolio, risky_pct_df, steps, years):
    # --- Plot: Allocation over time for a single scenario ---
    scenario_idx = 0
    t_alloc = np.linspace(0, years.value, steps)

    fig_alloc, ax_alloc = plt.subplots(figsize=(14, 6))

    risky_pct = risky_pct_df.iloc[:, scenario_idx]
    safe_pct = 100 - risky_pct

    ax_alloc.fill_between(
        t_alloc, 0, safe_pct, alpha=0.6, color="lightgreen", label="Safe Asset"
    )
    ax_alloc.fill_between(
        t_alloc, safe_pct, 100, alpha=0.6, color="steelblue", label="Risky Asset"
    )

    # Portfolio value on secondary axis
    ax_twin = ax_alloc.twinx()
    ax_twin.plot(
        t_alloc,
        portfolio[:, scenario_idx],
        color="darkred",
        linewidth=2,
        label="Portfolio Value",
    )
    ax_twin.set_ylabel("Portfolio Value (× initial)", color="darkred")

    ax_alloc.set_title(
        f"Asset Allocation Over Time (Scenario #{scenario_idx})", fontsize=14
    )
    ax_alloc.set_xlabel("Years")
    ax_alloc.set_ylabel("Allocation (%)")
    ax_alloc.set_ylim(0, 100)
    ax_alloc.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax_alloc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(below_floor, mo, multiplier):
    mo.md(rf"""
    ## Interpretation

    - With multiplier **{multiplier.value}×**, the strategy allocates up to
      **{multiplier.value}×** the cushion to the risky asset.
    - In **{below_floor:.1f}%** of scenarios the portfolio ended **below the floor**
      (gap risk materialized).
    - A lower multiplier reduces gap risk but also limits upside participation.
    - A higher floor provides stronger protection but leaves less room for growth.

    > **Tip:** Try setting the multiplier very high (e.g. 12–15) to see how
    > gap risk increases dramatically in volatile markets.
    """)
    return


if __name__ == "__main__":
    app.run()
