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
    # Bond Duration Matching (Immunization)

    **Duration matching** (also called **immunization**) is a strategy that structures a bond
    portfolio so its interest rate sensitivity cancels out at a specific future date — the
    **target horizon**. A portfolio immunized at time $T$ will have approximately the same
    terminal value regardless of parallel shifts in interest rates.

    ## Macaulay Duration

    For a zero-coupon bond, **Macaulay duration equals its maturity**. The duration of a
    portfolio is the present-value-weighted average of individual bond durations:

    $$D_\text{portfolio} = w_1 \cdot T_1 + w_2 \cdot T_2$$

    where $w_1, w_2$ are the fractions of **present value** invested in each bond.

    ## The Matching Condition

    Given a target horizon $T$ and two zero-coupon bonds with maturities $T_1 < T < T_2$:

    $$w_1 = \frac{T_2 - T}{T_2 - T_1}, \qquad w_2 = \frac{T - T_1}{T_2 - T_1}$$

    This ensures $D_\text{portfolio} = T$ — the portfolio is immunized against small
    parallel rate shifts at the target date.

    ## Three Strategies Compared

    | Strategy | Description | Terminal value at $T$ |
    |---|---|---|
    | **Short Bond Only** | Rolling $T_1$-maturity bonds — reinvest face value at each maturity | Path-dependent |
    | **Long Bond Only** | Buy bond maturing at $T_2$, sell at market price at $T$ | $F \cdot (1 + r_T)^{-(T_2 - T)}$ |
    | **Duration-Matched** | Weighted portfolio of both bonds | $w_1 \cdot$ Short $+ \; w_2 \cdot$ Long |

    The **duration-matched portfolio** should exhibit the **lowest dispersion** of terminal
    values at $T$ — this is the immunization effect.
    """)
    return


@app.cell
def _(mo):
    # --- Target horizon ---
    T = mo.ui.slider(
        start=2,
        stop=15,
        step=0.5,
        value=7.0,
        label="Target Horizon T (years)",
    )

    # --- Bond maturities ---
    T1 = mo.ui.slider(
        start=1,
        stop=14,
        step=0.5,
        value=3.0,
        label="Short Bond Maturity T₁ (years)",
    )

    T2 = mo.ui.slider(
        start=3,
        stop=30,
        step=0.5,
        value=10.0,
        label="Long Bond Maturity T₂ (years)",
    )

    face_value = mo.ui.slider(
        start=100,
        stop=10000,
        step=100,
        value=1000,
        label="Face Value ($)",
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

    ### Target &amp; Bonds
    {T}  
    {T1}  
    {T2}  
    {face_value}  

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
        T,
        T1,
        T2,
        a,
        b,
        face_value,
        n_scenarios,
        r_0_slider,
        sigma,
        steps_per_year,
    )


@app.cell
def _(T, T1, T2, face_value, mo):
    # --- Validate maturities ---
    if T1.value >= T.value:
        mo.md("⚠️ **Adjustment**: Short bond maturity T₁ must be less than target horizon T. Using T₁ = T − 0.5.")
        T1_eff = T.value - 0.5
    else:
        T1_eff = T1.value

    if T2.value <= T.value:
        mo.md("⚠️ **Adjustment**: Long bond maturity T₂ must be greater than target horizon T. Using T₂ = T + 0.5.")
        T2_eff = T.value + 0.5
    else:
        T2_eff = T2.value

    F = face_value.value
    return F, T1_eff, T2_eff


@app.cell
def _(
    F,
    T,
    T1_eff,
    T2_eff,
    a,
    b,
    cir,
    discount,
    n_scenarios,
    np,
    pd,
    r_0_slider,
    sigma,
    steps_per_year,
):
    # --- Duration-matching weights (present-value fractions) ---
    w1 = (T2_eff - T.value) / (T2_eff - T1_eff)  # fraction in short bond
    w2 = (T.value - T1_eff) / (T2_eff - T1_eff)  # fraction in long bond

    # --- Run CIR simulation (need rates up to max maturity) ---
    max_T = max(T2_eff, T.value)
    rates_df = cir(
        n_years=max_T,
        n_scenarios=n_scenarios.value,
        a=a.value,
        b=b.value,
        sigma=sigma.value,
        steps_per_year=steps_per_year.value,
        r_0=r_0_slider.value,
    )

    num_steps = int(max_T * steps_per_year.value) + 1
    t_axis_full = np.linspace(0, max_T, num_steps)

    # --- Compute zero-coupon bond prices along each path ---
    def bond_price_paths(maturity, rates, t_axis, face):
        """Compute bond price paths for all scenarios."""
        remaining = np.maximum(maturity - t_axis, 0.0)
        prices = face * (1 + rates) ** (-remaining[:, np.newaxis])
        return pd.DataFrame(prices, index=t_axis)

    P1_df = bond_price_paths(T1_eff, rates_df.values, t_axis_full, F)  # short bond
    P2_df = bond_price_paths(T2_eff, rates_df.values, t_axis_full, F)  # long bond

    # Initial prices (deterministic — use r_0)
    P1_0 = float(F * discount(T1_eff, r_0_slider.value))
    P2_0 = float(F * discount(T2_eff, r_0_slider.value))

    # --- Number of bonds for each strategy (invest $1 initially) ---
    n1_all = 1.0 / P1_0  # all-in short
    n2_all = 1.0 / P2_0  # all-in long
    n1_p = w1 / P1_0      # portfolio short allocation
    n2_p = w2 / P2_0      # portfolio long allocation

    # --- Index lookups for key time points ---
    idx_T1 = int(T1_eff * steps_per_year.value)
    idx_T = int(T.value * steps_per_year.value)

    # --- Path setup ---
    idx_range = slice(0, idx_T + 1)
    t_axis = t_axis_full[idx_range]

    # --- Rolling short bond: at each maturity, reinvest into a new T₁-duration bond ---
    roll_step_size = int(T1_eff * steps_per_year.value)
    roll_indices = set(range(roll_step_size, idx_T + 1, roll_step_size))

    V1_path = np.ones((idx_T + 1, n_scenarios.value))
    n_short_bonds = np.full(n_scenarios.value, n1_all)  # number of bonds held
    short_next_mat = T1_eff  # next maturity date (years)

    for step in range(1, idx_T + 1):
        r_s = rates_df.iloc[step].values  # rates at this step (all scenarios)
        t_s = t_axis[step]

        if step in roll_indices:
            # Bonds matured — receive face value, buy new T₁-duration bonds
            cash = n_short_bonds * F
            bond_price = F * (1 + r_s) ** (-T1_eff)
            n_short_bonds = cash / bond_price
            short_next_mat = t_s + T1_eff

        rem = max(short_next_mat - t_s, 0.0)
        V1_path[step] = n_short_bonds * F * (1 + r_s) ** (-rem)

    V1_terminal = V1_path[-1]  # terminal value per scenario

    # --- Long bond: sell at T at market price (remaining maturity T2 - T) ---
    r_T_all = rates_df.iloc[idx_T].values
    V2_terminal = n2_all * F * (1 + r_T_all) ** (-(T2_eff - T.value))

    # --- Duration-matched portfolio (combines rolling short + long) ---
    Vp_terminal = w1 * V1_terminal + w2 * V2_terminal

    # --- Path values (normalized to start at 1) ---
    V2_path = n2_all * P2_df.values[idx_range]
    Vp_path = w1 * V1_path + w2 * V2_path

    V1_path_df = pd.DataFrame(V1_path, index=t_axis)
    V2_path_df = pd.DataFrame(V2_path, index=t_axis)
    Vp_path_df = pd.DataFrame(Vp_path, index=t_axis)

    # --- Statistics for terminal values ---
    stats = {}
    for name, term_vals in [("Short Only", V1_terminal), ("Long Only", V2_terminal), ("Matched Portfolio", Vp_terminal)]:
        stats[name] = {
            "mean": np.mean(term_vals),
            "median": np.median(term_vals),
            "std": np.std(term_vals),
            "min": np.min(term_vals),
            "max": np.max(term_vals),
            "var_95": np.percentile(term_vals, 5),
            "var_99": np.percentile(term_vals, 1),
        }

    # Feller condition
    feller_holds = 2 * a.value * b.value > sigma.value ** 2

    return (
        P1_0,
        P2_0,
        V1_path_df,
        V1_terminal,
        V2_path_df,
        V2_terminal,
        Vp_path_df,
        Vp_terminal,
        feller_holds,
        rates_df,
        stats,
        t_axis,
        t_axis_full,
        w1,
        w2,
    )


@app.cell
def _(P1_0, P2_0, T, T1_eff, T2_eff, a, b, feller_holds, mo, sigma, w1, w2):
    feller_icon = "✅" if feller_holds else "⚠️"

    mo.md(rf"""
    ## Duration-Matching Summary

    | Metric | Value |
    |---|---|
    | Target horizon (T) | **{T.value:.1f} years** |
    | Short bond maturity (T₁) | {T1_eff:.1f} years |
    | Long bond maturity (T₂) | {T2_eff:.1f} years |
    | | |
    | **Portfolio weights (PV fractions)** | |
    | Short bond weight (w₁) | **{w1:.1%}** |
    | Long bond weight (w₂) | **{w2:.1%}** |
    | Portfolio Macaulay duration | {w1 * T1_eff + w2 * T2_eff:.2f} years |
    | | |
    | Initial short bond price | ${P1_0:,.2f} |
    | Initial long bond price | ${P2_0:,.2f} |
    | | |
    | Feller condition ($2ab > \sigma^2$) | {feller_icon} ($2ab = {2 * a.value * b.value:.4f}$, $\sigma^2 = {sigma.value ** 2:.4f}$) |

    ### Interpretation

    - To immunize a **{T.value:.1f}-year** liability, invest **{w1:.1%}** in the {T1_eff:.1f}-year bond and **{w2:.1%}** in the {T2_eff:.1f}-year bond.
    - The portfolio's Macaulay duration ({w1 * T1_eff + w2 * T2_eff:.2f} years) exactly matches the target horizon.
    - At $t = {T.value:.1f}$ years, the duration-matched portfolio should show **lower dispersion** than either bond held alone.
    """)
    return


@app.cell
def _(T, T1_eff, mo, stats):
    # --- Terminal Value Statistics Table ---
    mo.md(rf"""
    ## Terminal Value Statistics at T = {T.value:.1f} years

    (All strategies start with a $1.00 initial investment)

    | Metric | Short Only | Long Only | **Matched Portfolio** |
    |---|---|---|---|
    | Mean | ${stats['Short Only']['mean']:,.3f} | ${stats['Long Only']['mean']:,.3f} | **${stats['Matched Portfolio']['mean']:,.3f}** |
    | Median | ${stats['Short Only']['median']:,.3f} | ${stats['Long Only']['median']:,.3f} | **${stats['Matched Portfolio']['median']:,.3f}** |
    | **Std Dev** | ${stats['Short Only']['std']:,.4f} | ${stats['Long Only']['std']:,.4f} | **${stats['Matched Portfolio']['std']:,.4f}** |
    | Min | ${stats['Short Only']['min']:,.3f} | ${stats['Long Only']['min']:,.3f} | **${stats['Matched Portfolio']['min']:,.3f}** |
    | Max | ${stats['Short Only']['max']:,.3f} | ${stats['Long Only']['max']:,.3f} | **${stats['Matched Portfolio']['max']:,.3f}** |
    | 5% VaR | ${stats['Short Only']['var_95']:,.3f} | ${stats['Long Only']['var_95']:,.3f} | **${stats['Matched Portfolio']['var_95']:,.3f}** |
    | 1% VaR | ${stats['Short Only']['var_99']:,.3f} | ${stats['Long Only']['var_99']:,.3f} | **${stats['Matched Portfolio']['var_99']:,.3f}** |

    ### Key Observation

    The **duration-matched portfolio** has the **lowest standard deviation** —
    this is the immunization effect. By matching the portfolio duration to the
    target horizon, interest rate risk is largely neutralized:

    - **Short bond alone**: every {T1_eff:.1f} years the bond matures and face value is
      reinvested into a new bond. At each maturity the value is exactly known
      (the "pull-to-par" reset), but the amount available to reinvest depends on
      the rate path — this is **reinvestment risk**.
    - **Long bond alone**: exposed to **price risk** throughout — you must sell
      before maturity at an unknown market price.
    - **Matched portfolio**: these two risks partially **offset each other**,
      reducing overall dispersion at the target horizon.
    """)
    return


@app.cell
def _(
    T,
    T1_eff,
    T2_eff,
    a,
    b,
    n_scenarios,
    np,
    plt,
    rates_df,
    sigma,
    t_axis_full,
):
    # --- Plot 1: CIR Interest Rate Paths ---
    n_plot = min(15, n_scenarios.value)
    sample_cols = np.random.choice(n_scenarios.value, size=n_plot, replace=False)

    fig1, ax1 = plt.subplots(figsize=(14, 6))

    for rate_col in sample_cols:
        ax1.plot(
            t_axis_full,
            rates_df.iloc[:, rate_col] * 100,
            alpha=0.3,
            linewidth=0.6,
            color="#69F0AE",
        )

    mean_rate = rates_df.mean(axis=1) * 100
    ax1.plot(t_axis_full, mean_rate, color="#FFD740", linewidth=2.5, label="Mean rate")

    ax1.axhline(y=b.value * 100, color="#B0BEC5", linestyle="--", linewidth=1.5,
                label=f"Long-run mean b = {b.value*100:.1f}%")
    ax1.axvline(x=T.value, color="#FF6E40", linestyle=":", linewidth=2,
                label=f"Target horizon T = {T.value:.1f}y")
    ax1.axvline(x=T1_eff, color="#448AFF", linestyle=":", linewidth=1.5,
                label=f"T₁ = {T1_eff:.1f}y")
    ax1.axvline(x=T2_eff, color="#E040FB", linestyle=":", linewidth=1.5,
                label=f"T₂ = {T2_eff:.1f}y")

    ax1.set_title(
        f"CIR Interest Rate Paths (a={a.value:.2f}, b={b.value*100:.1f}%, σ={sigma.value:.2f})",
        fontsize=14,
    )
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Annual Interest Rate (%)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return n_plot, sample_cols


@app.cell
def _(T, T1_eff, V1_path_df, n_plot, np, plt, sample_cols, t_axis):
    # --- Plot 2a: Short Bond Only — Value Paths ---
    fig2a, ax2a = plt.subplots(figsize=(14, 6))

    for short_col in sample_cols:
        ax2a.plot(t_axis, V1_path_df.iloc[:, short_col], alpha=0.25, linewidth=0.5, color="#448AFF")

    mean_short = V1_path_df.mean(axis=1)
    ax2a.plot(t_axis, mean_short, color="#FFD740", linewidth=2.5, label="Mean")
    ax2a.axhline(y=1.0, color="#B0BEC5", linestyle="--", linewidth=1, label="Initial ($1)")
    ax2a.axvline(x=T.value, color="#FF6E40", linestyle=":", linewidth=1.5, label=f"T = {T.value:.1f}y")

    # Mark each roll maturity
    roll_years = np.arange(T1_eff, T.value + 1e-9, T1_eff)
    for i, ry in enumerate(roll_years):
        ax2a.axvline(x=ry, color="#448AFF", linestyle="--", linewidth=0.8, alpha=0.5,
                     label="Maturity (roll)" if i == 0 else "")

    ax2a.set_title(f"Short Bond Only (rolling every {T1_eff:.1f}y) — {n_plot} Sample Scenarios", fontsize=14)
    ax2a.set_xlabel("Time (years)")
    ax2a.set_ylabel("Portfolio Value (start = $1.00)")
    ax2a.legend(loc="upper left", fontsize=9)
    ax2a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(T, V2_path_df, n_plot, plt, sample_cols, t_axis):
    # --- Plot 2b: Long Bond Only — Value Paths ---
    fig2b, ax2b = plt.subplots(figsize=(14, 6))

    for long_col in sample_cols:
        ax2b.plot(t_axis, V2_path_df.iloc[:, long_col], alpha=0.25, linewidth=0.5, color="#69F0AE")

    mean_long = V2_path_df.mean(axis=1)
    ax2b.plot(t_axis, mean_long, color="#FFD740", linewidth=2.5, label="Mean")
    ax2b.axhline(y=1.0, color="#B0BEC5", linestyle="--", linewidth=1, label="Initial ($1)")
    ax2b.axvline(x=T.value, color="#FF6E40", linestyle=":", linewidth=1.5, label=f"T = {T.value:.1f}y")

    ax2b.set_title(f"Long Bond Only — {n_plot} Sample Scenarios", fontsize=14)
    ax2b.set_xlabel("Time (years)")
    ax2b.set_ylabel("Portfolio Value (start = $1.00)")
    ax2b.legend(loc="upper left", fontsize=9)
    ax2b.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(T, Vp_path_df, n_plot, plt, sample_cols, t_axis):
    # --- Plot 2c: Duration-Matched Portfolio — Value Paths ---
    fig2c, ax2c = plt.subplots(figsize=(14, 6))

    for portfolio_col in sample_cols:
        ax2c.plot(t_axis, Vp_path_df.iloc[:, portfolio_col], alpha=0.25, linewidth=0.5, color="#FF6E40")

    mean_portfolio = Vp_path_df.mean(axis=1)
    ax2c.plot(t_axis, mean_portfolio, color="#FFD740", linewidth=2.5, label="Mean")
    ax2c.axhline(y=1.0, color="#B0BEC5", linestyle="--", linewidth=1, label="Initial ($1)")
    ax2c.axvline(x=T.value, color="#FF6E40", linestyle=":", linewidth=1.5, label=f"T = {T.value:.1f}y")

    ax2c.set_title(f"Duration-Matched Portfolio — {n_plot} Sample Scenarios", fontsize=14)
    ax2c.set_xlabel("Time (years)")
    ax2c.set_ylabel("Portfolio Value (start = $1.00)")
    ax2c.legend(loc="upper left", fontsize=9)
    ax2c.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(T, V1_terminal, V2_terminal, Vp_terminal, np, plt):
    # --- Plot 3: Terminal Value Distribution Comparison ---
    fig3, ax3 = plt.subplots(figsize=(14, 6))

    hist_data = [
        (V1_terminal, "#448AFF", "Short Only"),
        (V2_terminal, "#69F0AE", "Long Only"),
        (Vp_terminal, "#FF6E40", "Matched Portfolio"),
    ]

    bins = 50
    for hist_vals, hist_color, label in hist_data:
        ax3.hist(hist_vals, bins=bins, color=hist_color, alpha=0.4, density=True, label=label)
        counts, edges = np.histogram(hist_vals, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax3.plot(centers, counts, color=hist_color, linewidth=2)

    # Vertical lines for means
    for hist_vals, hist_color, _ in hist_data:
        ax3.axvline(x=np.mean(hist_vals), color=hist_color, linestyle="--", linewidth=2, alpha=0.8)

    ax3.axvline(x=1.0, color="#B0BEC5", linestyle="-", linewidth=1.5,
                label="Initial investment ($1.00)")

    ax3.set_title(
        f"Distribution of Terminal Values at T = {T.value:.1f} years",
        fontsize=14,
    )
    ax3.set_xlabel("Terminal Value (per $1 invested)")
    ax3.set_ylabel("Density")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(T, Vp_path_df, np, plt, t_axis):
    # --- Plot 4: Fan Chart — Duration-Matched Portfolio ---
    fig4, ax4 = plt.subplots(figsize=(14, 6))

    pcts = [5, 10, 25, 50, 75, 90, 95]
    percentiles = np.percentile(Vp_path_df.values, pcts, axis=1)
    band_colors = ["#FF5252", "#FFAB40", "#FFD740", "#69F0AE", "#40C4FF", "#448AFF"]

    for band in range(len(pcts) // 2):
        ax4.fill_between(
            t_axis,
            percentiles[band],
            percentiles[-(band + 1)],
            alpha=0.2,
            color=band_colors[band],
            label=f"{pcts[band]}%–{pcts[-(band+1)]}%",
        )

    ax4.plot(t_axis, percentiles[len(pcts) // 2], color="#FFD740", linewidth=2,
             label="Median (50%)")
    ax4.axhline(y=1.0, color="#B0BEC5", linestyle="--", linewidth=1.5,
                label="Initial investment ($1.00)")
    ax4.axvline(x=T.value, color="#FF6E40", linestyle=":", linewidth=2,
                label=f"Target horizon T = {T.value:.1f}y")

    ax4.set_title(
        "Duration-Matched Portfolio — Fan Chart Over Time",
        fontsize=14,
    )
    ax4.set_xlabel("Time (years)")
    ax4.set_ylabel("Portfolio Value (start = $1.00)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(T, T1_eff, V1_path_df, V2_path_df, Vp_path_df, plt, t_axis):
    # --- Plot 5: Dispersion Comparison — Std Dev Over Time ---
    fig5, ax5 = plt.subplots(figsize=(14, 6))

    std_short = V1_path_df.std(axis=1)
    std_long = V2_path_df.std(axis=1)
    std_portfolio = Vp_path_df.std(axis=1)

    ax5.plot(t_axis, std_short, color="#448AFF", linewidth=2, label="Short Bond Only")
    ax5.plot(t_axis, std_long, color="#69F0AE", linewidth=2, label="Long Bond Only")
    ax5.plot(t_axis, std_portfolio, color="#FF6E40", linewidth=2.5, label="Duration-Matched Portfolio")

    ax5.axvline(x=T.value, color="#FF6E40", linestyle=":", linewidth=2,
                label=f"Target horizon T = {T.value:.1f}y")
    ax5.axvline(x=T1_eff, color="#B0BEC5", linestyle="--", linewidth=1,
                label=f"T₁ = {T1_eff:.1f}y", alpha=0.6)

    ax5.set_title(
        "Value Dispersion Over Time — Standard Deviation of Portfolio Value",
        fontsize=14,
    )
    ax5.set_xlabel("Time (years)")
    ax5.set_ylabel("Standard Deviation ($)")
    ax5.legend(loc="upper left", fontsize=9)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How It Works — The Intuition

    Duration matching exploits the **offsetting nature** of two interest rate risks:

    ### 1. Price Risk vs. Reinvestment Risk

    | Risk | When rates rise... | When rates fall... |
    |---|---|---|
    | **Price risk** (long bond) | Bond price falls 📉 | Bond price rises 📈 |
    | **Reinvestment risk** (short bond) | More bonds bought at roll 📈 | Fewer bonds bought at roll 📉 |

    When the portfolio duration equals the horizon, these two effects partially **cancel out**:

    - If rates **rise**: the long bond loses value, but the short bond's maturing
      proceeds buy more new bonds at the higher rate.
    - If rates **fall**: the long bond gains value, but the short bond's maturing
      proceeds buy fewer new bonds at the lower rate.

    ### 2. The Sawtooth Pattern

    The rolling short bond exhibits a distinctive **sawtooth** pattern in its
    dispersion: variance grows between maturities (price risk) and resets at each
    maturity when face value is received (pull-to-par). The first maturity ($t = T_1$)
    is fully deterministic — all scenarios receive the same face value. Subsequent
    maturities vary because different rate paths lead to different reinvestment amounts.

    ### 3. Why It's Imperfect

    Immunization is only exact for:
    - **Small, parallel** shifts in the yield curve
    - **Instantaneous** shifts occurring right after portfolio setup

    In practice, with stochastic CIR rates and a continuously evolving yield curve,
    the hedge is approximate — but still dramatically reduces dispersion compared
    to holding either bond alone, as the statistics above show.

    ### 4. Rebalancing

    True immunization requires **periodic rebalancing** because as time passes,
    the portfolio duration drifts away from the (shrinking) target horizon.
    The current weights $w_1, w_2$ are optimal only at $t=0$.
    """)
    return


if __name__ == "__main__":
    app.run()
