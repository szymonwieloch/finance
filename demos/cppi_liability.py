import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CPPI vs. Alternative Strategies for Liability Funding

    You have a **liability** $L$ due in $T$ years and **initial capital** to invest today.
    Your money can go into two assets:

    - A **risky asset** (e.g., stocks) with expected return $\mu$ and volatility $\sigma$,
      modeled via Geometric Brownian Motion.
    - A **safe asset** (e.g., bonds) earning a constant annual rate $r_f$.

    This notebook compares **three strategies** for meeting the liability:

    | # | Strategy | Description |
    |---|----------|-------------|
    | **A** | **All-In Risky** | Invest 100% in the risky asset. Maximum upside, maximum downside. |
    | **B** | **Liability-Matched** | At $t=0$, invest exactly $\frac{L}{(1+r_f)^T}$ (the PV of the liability) in the safe asset; the rest in the risky asset. No rebalancing — the safe portion compounds to exactly $L$ at maturity (assuming constant rates). |
    | **C** | **CPPI** | Dynamic strategy: at every step, the **floor** $F_t = \frac{L}{(1+r_f)^{T-t}}$ (the PV of the remaining liability). Cushion $= \max(\text{Portfolio}_t - F_t,\;0)$. Risky exposure $= \min(m \cdot \text{Cushion},\; \text{Portfolio}_t)$. Pro-cyclical — increases risk after gains, cuts risk after losses. |

    > **Key question:** Which strategy gives the best balance of upside participation
    > and downside protection?

    Use the sliders below to explore.
    """)
    return


@app.cell
def _():
    import import_fix as _
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from toolkit.other import gbm_returns, discount
    return discount, gbm_returns, mo, np, plt


@app.cell
def _(mo):
    # --- Parameter sliders ---

    # Liability
    liability_amount = mo.ui.slider(
        start=200, stop=2000, step=50, value=1000,
        label="Liability Amount ($)",
    )
    liability_maturity = mo.ui.slider(
        start=1, stop=20, step=1, value=10,
        label="Liability Maturity (years)",
    )

    # Capital & Safe Rate
    initial_capital = mo.ui.slider(
        start=200, stop=2000, step=50, value=1000,
        label="Initial Capital ($)",
    )
    safe_rate = mo.ui.slider(
        start=0.01, stop=0.10, step=0.005, value=0.04,
        label="Safe Rate (r_f)",
    )

    # CPPI
    multiplier = mo.ui.slider(
        start=1, stop=15, step=1, value=4,
        label="CPPI Multiplier (m)",
    )

    # Risky asset (GBM)
    mu = mo.ui.slider(
        start=0.01, stop=0.20, step=0.01, value=0.07,
        label="Risky Asset Return (μ)",
    )
    sigma_risky = mo.ui.slider(
        start=0.05, stop=0.50, step=0.05, value=0.18,
        label="Risky Asset Volatility (σ)",
    )

    # Simulation settings
    steps_per_year = 12
    scenarios = mo.ui.slider(
        start=100, stop=5000, step=100, value=1000,
        label="Number of Scenarios",
    )
    return (
        initial_capital,
        liability_amount,
        liability_maturity,
        mu,
        multiplier,
        safe_rate,
        scenarios,
        sigma_risky,
        steps_per_year,
    )


@app.cell
def _(
    initial_capital,
    liability_amount,
    liability_maturity,
    mo,
    mu,
    multiplier,
    safe_rate,
    scenarios,
    sigma_risky,
):
    mo.md(f"""
    ## Parameters

    | Category | Parameter | Value |
    |---|---|---|
    | **Liability** | Amount | ${liability_amount.value:,.0f} |
    | | Maturity | {liability_maturity.value} years |
    | **Capital & Rates** | Initial Capital | ${initial_capital.value:,.0f} |
    | | Safe Rate (r_f) | {safe_rate.value:.1%} |
    | **Risky Asset** | Expected Return (μ) | {mu.value:.0%} |
    | | Volatility (σ) | {sigma_risky.value:.0%} |
    | **CPPI** | Multiplier (m) | {multiplier.value}× |
    | **Simulation** | Scenarios | {scenarios.value} |
    """)
    return


@app.cell
def _(
    discount,
    gbm_returns,
    initial_capital,
    liability_amount,
    liability_maturity,
    mu,
    multiplier,
    np,
    safe_rate,
    scenarios,
    sigma_risky,
    steps_per_year,
):
    # ============================================================
    # Simulation — all three strategies in one pass
    # ============================================================
    T = liability_maturity.value
    L = liability_amount.value
    r = safe_rate.value
    steps = int(T * steps_per_year) + 1
    dt = 1.0 / steps_per_year
    n_scen = scenarios.value
    init_val = initial_capital.value

    # Time grid
    t_grid = np.linspace(0, T, steps)

    # --- Generate risky asset returns (shared across strategies) ---
    risky_rets = gbm_returns(
        years=T, scenarios=n_scen, mu=mu.value,
        sigma=sigma_risky.value, steps_per_year=steps_per_year,
    )  # (steps, n_scen), first row = 0
    risky_prices = (1 + risky_rets).cumprod(axis=0)  # cumulative growth of $1

    # --- Safe asset: deterministic compounding at constant rate ---
    rf_period = (1 + r) ** dt - 1
    safe_growth = (1 + rf_period) ** np.arange(steps).reshape(-1, 1)

    # ============================================================
    # Strategy A: All-In Risky
    # ============================================================
    port_a = init_val * risky_prices  # (steps, n_scen)

    # ============================================================
    # Strategy B: Liability-Matched (static allocation at t=0)
    # ============================================================
    pv_liability = L * discount(T, r)  # scalar: PV at constant safe rate
    safe_b = min(init_val, pv_liability)
    risky_b = init_val - safe_b
    safe_growth_2d = np.tile(safe_growth, (1, n_scen))
    port_b = safe_b * safe_growth_2d + risky_b * risky_prices

    # ============================================================
    # Strategy C: CPPI with dynamic floor
    # ============================================================
    portfolio_c = np.ones((steps, n_scen)) * init_val
    floor_c = np.zeros((steps, n_scen))
    risky_exposure_c = np.zeros((steps, n_scen))

    for t in range(steps):
        remaining = T - t_grid[t]
        floor_c[t, :] = L * discount(remaining, r)

        if t < steps - 1:
            cushion = np.maximum(portfolio_c[t] - floor_c[t], 0.0)
            risky_exposure_c[t] = np.minimum(multiplier.value * cushion, portfolio_c[t])
            safe_c = portfolio_c[t] - risky_exposure_c[t]
            portfolio_c[t + 1] = (
                safe_c * (1 + rf_period)
                + risky_exposure_c[t] * (1 + risky_rets[t + 1])
            )

    # Final period allocation
    cushion = np.maximum(portfolio_c[-1] - floor_c[-1], 0.0)
    risky_exposure_c[-1] = np.minimum(multiplier.value * cushion, portfolio_c[-1])

    # ============================================================
    # Final values for all strategies
    # ============================================================
    final_a = port_a[-1]
    final_b = port_b[-1]
    final_c = portfolio_c[-1]

    portfolios = {
        "A: All-In Risky": port_a,
        "B: Liability-Matched": port_b,
        "C: CPPI": portfolio_c,
    }
    finals = {
        "A: All-In Risky": final_a,
        "B: Liability-Matched": final_b,
        "C: CPPI": final_c,
    }

    # ============================================================
    # Summary statistics
    # ============================================================
    stats = {}
    for s_name, s_fv in finals.items():
        stats[s_name] = {
            "Mean": s_fv.mean(),
            "Median": np.median(s_fv),
            "Std Dev": s_fv.std(),
            "P5": np.percentile(s_fv, 5),
            "P95": np.percentile(s_fv, 95),
            "P(Shortfall)": (s_fv < L).mean() * 100,
            "Sharpe-like*": (s_fv.mean() - L) / s_fv.std() if s_fv.std() > 0 else 0,
        }

    # Funding ratios
    initial_pv_liability = L * discount(T, r)
    initial_funding_ratio = init_val / initial_pv_liability

    return (
        L,
        T,
        finals,
        floor_c,
        initial_funding_ratio,
        portfolio_c,
        portfolios,
        r,
        risky_exposure_c,
        stats,
        t_grid,
    )


@app.cell
def _(L, T, initial_funding_ratio, mo, r, stats):
    # --- Summary Comparison Table ---
    rows = ""
    for t_name, s in stats.items():
        rows += (
            f"| **{t_name}** | ${s['Mean']:,.0f} | ${s['Median']:,.0f} | "
            f"${s['Std Dev']:,.0f} | ${s['P5']:,.0f} | ${s['P95']:,.0f} | "
            f"**{s['P(Shortfall)']:.1f}%** | {s['Sharpe-like*']:.3f} |\n"
        )

    mo.md(f"""
    ## Results — Strategy Comparison

    **Liability:** ${L:,.0f} due in {T} years  
    **Initial Capital:** based on sliders (Funding Ratio = {initial_funding_ratio:.3f})  
    **Safe Rate:** {r:.1%}

    | Strategy | Mean ($) | Median ($) | Std Dev ($) | P5 ($) | P95 ($) | % Liability Not Met | Sharpe-like* |
    |---|---|---|---|---|---|---|---|
    {rows}
    > *Sharpe-like = (Mean Final Value − Liability) / Std Dev of Final Value.
    > Higher is better: more excess over the liability per unit of risk.
    > **% Liability Not Met** = percentage of scenarios where the final portfolio
    > value fell below the liability — the key downside risk measure.

    ### Key Takeaways

    - **All-In Risky** has the highest upside (P95) but also the highest shortfall
      probability — you could end up with very little.
    - **Liability-Matched** guarantees the safe portion covers the liability at
      maturity (0% shortfall from the safe piece), but the risky portion still
      adds variance — though typically much less than All-In.
    - **CPPI** dynamically manages the trade-off: it cuts risk when the portfolio
      approaches the floor and increases it when there's surplus. The multiplier
      controls how aggressively.
    """)
    return


@app.cell
def _(L, plt, portfolios, t_grid):
    # --- Plot 1: Mean trajectories of all three strategies ---
    fig_mean, ax_mean = plt.subplots(figsize=(14, 7))

    colors = {
        "A: All-In Risky": "darkred",
        "B: Liability-Matched": "darkgreen",
        "C: CPPI": "steelblue",
    }
    linestyles = {
        "A: All-In Risky": "-",
        "B: Liability-Matched": "--",
        "C: CPPI": "-",
    }

    for p_name, p_port in portfolios.items():
        mean_path = p_port.mean(axis=1)
        ax_mean.plot(
            t_grid, mean_path, color=colors[p_name],
            linestyle=linestyles[p_name], linewidth=2.5,
            label=f"{p_name} (mean)",
        )

    ax_mean.axhline(y=L, color="gray", linestyle=":", linewidth=1.5,
                    label=f"Liability (${L:,.0f})")
    ax_mean.set_title("Mean Portfolio Value Across All Scenarios", fontsize=14)
    ax_mean.set_xlabel("Years")
    ax_mean.set_ylabel("Portfolio Value ($)")
    ax_mean.legend(loc="upper left")
    ax_mean.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(L, finals, plt):
    # --- Plot 2: Histogram of final outcomes — all strategies overlaid ---
    fig_hist, ax_hist = plt.subplots(figsize=(14, 7))

    colors_hist = {
        "A: All-In Risky": "darkred",
        "B: Liability-Matched": "darkgreen",
        "C: CPPI": "steelblue",
    }

    for h_name, h_fv in finals.items():
        ax_hist.hist(h_fv, bins=70, color=colors_hist[h_name],
                    edgecolor="white", alpha=0.45, density=True,
                    label=f"{h_name}")

    ax_hist.axvline(x=L, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Liability (${L:,.0f})")
    ax_hist.set_title("Distribution of Final Portfolio Values — All Strategies", fontsize=14)
    ax_hist.set_xlabel("Final Value ($)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(L, finals, np, plt):
    # --- Plot 3: Cumulative distribution — all three strategies overlaid ---
    fig_cum, ax_cum = plt.subplots(figsize=(14, 7))

    colors_cum = {
        "A: All-In Risky": "darkred",
        "B: Liability-Matched": "darkgreen",
        "C: CPPI": "steelblue",
    }

    for c_name, c_fv in finals.items():
        sorted_vals = np.sort(c_fv)
        cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        ax_cum.plot(sorted_vals, cumulative, color=colors_cum[c_name],
                    linewidth=2, label=c_name)
        sf = (c_fv < L).mean() * 100
        ax_cum.axhline(y=sf, color=colors_cum[c_name], linestyle=":",
                       linewidth=1, alpha=0.6)

    ax_cum.axvline(x=L, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Liability (${L:,.0f})")
    ax_cum.set_title("Cumulative Distribution of Final Outcomes", fontsize=14)
    ax_cum.set_xlabel("Final Value ($)")
    ax_cum.set_ylabel("Cumulative Probability (%)")
    ax_cum.legend(loc="lower right")
    ax_cum.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(L, floor_c, plt, portfolio_c, risky_exposure_c, t_grid):
    # --- Plot 4: CPPI allocation & floor for a single scenario ---
    scenario_idx = 0

    fig_cppi, ax_cppi = plt.subplots(figsize=(14, 7))

    risky_pct = np.where(portfolio_c > 0, risky_exposure_c / portfolio_c * 100, 0.0)
    safe_pct = 100 - risky_pct[:, scenario_idx]

    ax_cppi.fill_between(t_grid, 0, safe_pct, alpha=0.5, color="lightgreen",
                         label="Safe Allocation (%)")
    ax_cppi.fill_between(t_grid, safe_pct, 100, alpha=0.5, color="steelblue",
                         label="Risky Allocation (%)")
    ax_cppi.set_ylabel("Allocation (%)")
    ax_cppi.set_ylim(0, 100)

    ax_twin = ax_cppi.twinx()
    ax_twin.plot(t_grid, portfolio_c[:, scenario_idx], color="darkred",
                 linewidth=2, label="Portfolio Value")
    ax_twin.plot(t_grid, floor_c[:, scenario_idx], color="darkorange",
                 linewidth=2, linestyle="--", label="Floor (PV of Liability)")
    ax_twin.axhline(y=L, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                    label=f"Liability (${L:,.0f})")
    ax_twin.set_ylabel("Value ($)")

    lines1, labels1 = ax_cppi.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_cppi.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax_cppi.set_title(
        f"CPPI: Asset Allocation & Dynamic Floor (Scenario #{scenario_idx + 1})",
        fontsize=14,
    )
    ax_cppi.set_xlabel("Years")
    ax_cppi.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(L, mo, multiplier, r, stats):
    mo.md(f"""
    ## Interpretation

    ### How the Three Strategies Differ

    | Strategy | Risk-Taking | Rebalancing | When to Use |
    |---|---|---|---|
    | **A: All-In Risky** | Always 100% risky | Never | Strong conviction in equity returns; can tolerate large losses |
    | **B: Liability-Matched** | Static split at $t=0$ | Never | Want a guaranteed floor with some upside; simple and transparent |
    | **C: CPPI** | Dynamic: $m \\times$ cushion | Every period | Want to protect the floor while maximizing upside participation |

    ### The Role of the CPPI Multiplier

    With multiplier **{multiplier.value}×**, each $1 of cushion above the floor translates
    to ${multiplier.value} of risky exposure. Key trade-offs:

    - **Low multiplier (m=1–3):** Conservative — behaves similarly to Strategy B but
      with some dynamic adjustment. Low gap risk.
    - **Medium multiplier (m=4–7):** Balanced — meaningful upside participation with
      manageable gap risk.
    - **High multiplier (m=8+):** Aggressive — approaching All-In Risky behavior when
      the portfolio is well above the floor, but with a hard floor. Higher gap risk
      (risk of crashing through the floor in a single period).

    ### Gap Risk in CPPI

    "Gap risk" occurs when a sharp market drop pushes the portfolio below the floor
    before the strategy can rebalance. With monthly rebalancing (steps_per_year=12),
    a single month's crash can breach the floor if the risky exposure is too high.
    Compare the **P(Shortfall)** row in the results table across strategies to see
    how well each protects against this.

    ### What to Explore

    | Experiment | Action |
    |---|---|
    | See CPPI converge to All-In Risky | Set multiplier very high (10–15) |
    | See CPPI converge to Liability-Matched | Set multiplier to 1 |
    | Underfunded scenario | Set Initial Capital below PV of Liability |
    | High volatility stress | Increase σ to 0.30–0.50 |
    | Long-dated liability | Set maturity to 15–20 years |
    """)
    return


if __name__ == "__main__":
    app.run()
