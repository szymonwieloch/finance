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
    from toolkit.other import cir, discount

    return cir, discount, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zero-Coupon Bond Risk Analysis

    A **zero-coupon bond** (also called a pure discount bond) pays a fixed face value $F$ at
    maturity $T$ with no intermediate coupon payments. Its price today is simply the present
    value of that future payment:

    $$P(t, T) = F \cdot (1 + r_t)^{-(T-t)}$$

    where $r_t$ is the prevailing interest rate (short rate) at time $t$.

    ## Why Does Bond Price Change?

    Even though the final payoff is certain (absent default), the **market value** of the bond
    fluctuates before maturity because interest rates are stochastic. When rates rise, the bond
    becomes less valuable (you could earn more elsewhere), and when rates fall, the bond gains value.

    ## The CIR Model for Interest Rates

    We model the short rate $r_t$ using the **Cox-Ingersoll-Ross (CIR)** process:

    $$dr_t = a (b - r_t) dt + \sigma \sqrt{r_t} dW_t$$

    | Parameter | Meaning |
    |---|---|
    | $a$ | Mean-reversion speed — how fast rates return to the long-run mean |
    | $b$ | Long-run mean rate |
    | $\sigma$ | Volatility of rate changes |
    | $r_0$ | Initial (starting) short rate |

    Use the sliders below to explore how bond prices behave under different interest rate scenarios.
    """)
    return


@app.cell
def _(mo):
    # --- Bond parameters ---
    face_value = mo.ui.slider(
        start=100,
        stop=10000,
        step=100,
        value=1000,
        label="Face Value ($)",
    )

    maturity = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=10,
        label="Maturity (years)",
    )

    # --- CIR model parameters ---
    a = mo.ui.slider(
        start=0.01,
        stop=0.50,
        step=0.01,
        value=0.10,
        label="Mean-reversion speed (a)",
    )

    b = mo.ui.slider(
        start=0.01,
        stop=0.10,
        step=0.005,
        value=0.04,
        label="Long-run mean rate (b)",
    )

    sigma = mo.ui.slider(
        start=0.01,
        stop=0.20,
        step=0.01,
        value=0.06,
        label="Volatility (σ)",
    )

    r_0_slider = mo.ui.slider(
        start=0.01,
        stop=0.10,
        step=0.005,
        value=0.04,
        label="Initial short rate (r₀)",
    )

    # --- Simulation settings ---
    steps_per_year = mo.ui.slider(
        start=4,
        stop=52,
        step=4,
        value=12,
        label="Steps per year",
    )

    n_scenarios = mo.ui.slider(
        start=20,
        stop=500,
        step=10,
        value=100,
        label="Number of scenarios",
    )

    mo.md(f"""
    ## Parameters

    ### Bond
    {face_value}  
    {maturity}  

    ### CIR Model
    {a}  
    {b}  
    {sigma}  
    {r_0_slider}  

    ### Simulation
    {steps_per_year}  
    {n_scenarios}  
    """)
    return (
        a,
        b,
        face_value,
        maturity,
        n_scenarios,
        r_0_slider,
        sigma,
        steps_per_year,
    )


@app.cell
def _(
    a,
    b,
    cir,
    discount,
    face_value,
    maturity,
    n_scenarios,
    np,
    pd,
    r_0_slider,
    sigma,
    steps_per_year,
):
    # --- Simulate interest rate paths using CIR ---
    rates_df = cir(
        n_years=maturity.value,
        n_scenarios=n_scenarios.value,
        a=a.value,
        b=b.value,
        sigma=sigma.value,
        steps_per_year=steps_per_year.value,
        r_0=r_0_slider.value,
    )

    # Time axis in years
    num_steps = int(maturity.value * steps_per_year.value) + 1
    t_axis = np.linspace(0, maturity.value, num_steps)

    # --- Compute zero-coupon bond price along each path ---
    # Bond price at time t: P(t) = F * (1 + r_t)^{-(T - t)}
    # Build a matrix of remaining times (maturity - t) for each step
    remaining = maturity.value - t_axis  # shape (steps,)

    # For each scenario, compute bond price at every time step
    bond_prices = face_value.value * (1 + rates_df.values) ** (-remaining[:, np.newaxis])

    bond_prices_df = pd.DataFrame(bond_prices, index=t_axis)

    # --- Key statistics ---
    # Initial bond price (deterministic)
    initial_price = face_value.value * discount(maturity.value, r_0_slider.value)
    initial_price_val = float(initial_price)

    # Terminal bond prices (at maturity they all equal face value — that's deterministic)
    # More interesting: bond prices at an intermediate horizon, e.g. halfway
    mid_idx = num_steps // 2
    mid_horizon = t_axis[mid_idx]
    mid_prices = bond_prices_df.iloc[mid_idx].values

    mean_mid = np.mean(mid_prices)
    median_mid = np.median(mid_prices)
    std_mid = np.std(mid_prices)
    min_mid = np.min(mid_prices)
    max_mid = np.max(mid_prices)

    # VaR at 5% level — worst-case bond price at mid-horizon
    var_95_mid = np.percentile(mid_prices, 5)

    # Feller condition
    feller_holds = 2 * a.value * b.value > sigma.value ** 2

    return (
        bond_prices_df,
        feller_holds,
        initial_price_val,
        max_mid,
        mean_mid,
        median_mid,
        mid_horizon,
        mid_prices,
        min_mid,
        rates_df,
        std_mid,
        t_axis,
        var_95_mid,
    )


@app.cell
def _(
    a,
    b,
    face_value,
    feller_holds,
    initial_price_val,
    maturity,
    max_mid,
    mean_mid,
    median_mid,
    mid_horizon,
    min_mid,
    mo,
    sigma,
    std_mid,
    var_95_mid,
):
    feller_icon = "✅" if feller_holds else "⚠️"
    gain_loss_icon = "📈" if mean_mid > initial_price_val else "📉"

    mo.md(rf"""
    ## Summary Statistics

    | Metric | Value |
    |---|---|
    | Face Value | ${face_value.value:,.0f} |
    | Maturity | {maturity.value} years |
    | Initial Bond Price (deterministic) | **${initial_price_val:,.2f}** |
    | | |
    | **At t = {mid_horizon:.1f} years (mid-horizon)** | |
    | Mean bond price | ${mean_mid:,.2f} {gain_loss_icon} |
    | Median bond price | ${median_mid:,.2f} |
    | Std of bond prices | ${std_mid:,.2f} |
    | Min bond price | ${min_mid:,.2f} |
    | Max bond price | ${max_mid:,.2f} |
    | 5% VaR (worst-case) | **${var_95_mid:,.2f}** |
    | | |
    | Feller condition ($2ab > \sigma^2$) | {feller_icon} ($2ab = {2 * a.value * b.value:.4f}$, $\sigma^2 = {sigma.value ** 2:.4f}$) |

    ### Interpretation

    - **Initial price** of ${initial_price_val:,.2f} is what you pay today for a bond that will pay ${face_value:,.0f} in {maturity.value} years.
    - At the **mid-horizon** (t = {mid_horizon:.1f} years), the bond price ranges from ${min_mid:,.2f} to ${max_mid:,.2f} across scenarios.
    - The **5% VaR** of ${var_95_mid:,.2f} means there is a 5% chance the bond will be worth less than this amount at the mid-horizon.
    - At **maturity**, the bond converges to its face value of ${face_value:,.0f} in all scenarios (the "pull-to-par" effect).
    """)
    return


@app.cell
def _(a, b, bond_prices_df, face_value, n_scenarios, np, plt, sigma, t_axis):
    # --- Plot 1: Bond price trajectories ---
    n_plot = min(20, n_scenarios.value)
    sample_cols = np.random.choice(n_scenarios.value, size=n_plot, replace=False)

    fig1, ax1 = plt.subplots(figsize=(14, 6))

    for bp_col in sample_cols:
        ax1.plot(
            t_axis,
            bond_prices_df.iloc[:, bp_col],
            alpha=0.3,
            linewidth=0.6,
            color="steelblue",
        )

    # Mean path
    mean_bond = bond_prices_df.mean(axis=1)
    ax1.plot(t_axis, mean_bond, color="darkred", linewidth=2.5, label="Mean bond price")

    # Face value line
    ax1.axhline(
        y=face_value.value,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Face Value = ${face_value.value:,.0f}",
    )

    ax1.set_title(
        f"Zero-Coupon Bond Price Paths (a={a.value:.2f}, b={b.value*100:.1f}%, σ={sigma.value:.2f})",
        fontsize=14,
    )
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Bond Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return (sample_cols,)


@app.cell
def _(b, plt, rates_df, sample_cols, t_axis):
    # --- Plot 2: Interest rate paths (CIR) ---
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    for rate_col in sample_cols:
        ax2.plot(
            t_axis,
            rates_df.iloc[:, rate_col] * 100,
            alpha=0.3,
            linewidth=0.6,
            color="darkgreen",
        )

    # Mean rate
    mean_rate = rates_df.mean(axis=1) * 100
    ax2.plot(t_axis, mean_rate, color="darkred", linewidth=2.5, label="Mean rate")

    ax2.axhline(
        y=b.value * 100,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Long-run mean b = {b.value * 100:.1f}%",
    )

    ax2.set_title("CIR Interest Rate Paths", fontsize=14)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Annual Interest Rate (%)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(
    initial_price_val,
    mean_mid,
    median_mid,
    mid_horizon,
    mid_prices,
    plt,
    var_95_mid,
):
    # --- Plot 3: Distribution of bond prices at mid-horizon ---
    fig3, ax3 = plt.subplots(figsize=(14, 6))

    ax3.hist(
        mid_prices,
        bins=50,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
        density=True,
    )

    ax3.axvline(
        x=initial_price_val,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Initial price = ${initial_price_val:,.2f}",
    )
    ax3.axvline(
        x=mean_mid,
        color="darkred",
        linestyle="-",
        linewidth=2,
        label=f"Mean = ${mean_mid:,.2f}",
    )
    ax3.axvline(
        x=median_mid,
        color="darkgreen",
        linestyle="-.",
        linewidth=2,
        label=f"Median = ${median_mid:,.2f}",
    )
    ax3.axvline(
        x=var_95_mid,
        color="darkorange",
        linestyle=":",
        linewidth=2,
        label=f"5% VaR = ${var_95_mid:,.2f}",
    )

    # Shade the worst 5% tail
    ax3.axvspan(mid_prices.min(), var_95_mid, alpha=0.1, color="red", label="Worst 5% tail")

    ax3.set_title(
        f"Distribution of Bond Prices at t = {mid_horizon:.1f} years",
        fontsize=14,
    )
    ax3.set_xlabel("Bond Price ($)")
    ax3.set_ylabel("Density")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(bond_prices_df, face_value, np, plt, t_axis):
    # --- Plot 4: Fan chart — percentile bands over time ---
    fig4, ax4 = plt.subplots(figsize=(14, 6))

    # Compute percentiles at each time step
    pcts = [5, 10, 25, 50, 75, 90, 95]
    percentiles = np.percentile(bond_prices_df.values, pcts, axis=1)

    # Fill bands
    colors = ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4"]
    for band in range(len(pcts) // 2):
        ax4.fill_between(
            t_axis,
            percentiles[band],
            percentiles[-(band + 1)],
            alpha=0.2,
            color=colors[band],
            label=f"{pcts[band]}%–{pcts[-(band+1)]}%",
        )

    # Median line
    ax4.plot(t_axis, percentiles[len(pcts) // 2], color="black", linewidth=2, label="Median (50%)")

    # Face value
    ax4.axhline(y=face_value.value, color="gray", linestyle="--", linewidth=1.5, label=f"Face Value")

    ax4.set_title("Bond Price Fan Chart — Percentile Bands Over Time", fontsize=14)
    ax4.set_xlabel("Time (years)")
    ax4.set_ylabel("Bond Price ($)")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(bond_prices_df, np, plt, t_axis):
    # --- Plot 5: Bond price volatility over time (pull-to-par) ---
    fig5, ax5 = plt.subplots(figsize=(14, 6))

    std_over_time = bond_prices_df.std(axis=1)
    pct_range_90 = np.percentile(bond_prices_df.values, 95, axis=1) - np.percentile(bond_prices_df.values, 5, axis=1)

    ax5.fill_between(t_axis, 0, pct_range_90, alpha=0.3, color="steelblue", label="90% range width")
    ax5.plot(t_axis, std_over_time, color="darkred", linewidth=2, label="Standard deviation")
    ax5.plot(t_axis, pct_range_90, color="darkblue", linewidth=2, linestyle="--", label="5%–95% range")

    ax5.set_title("Bond Price Uncertainty Over Time (Pull-to-Par Effect)", fontsize=14)
    ax5.set_xlabel("Time (years)")
    ax5.set_ylabel("Price Dispersion ($)")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
