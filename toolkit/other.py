import numpy as np
import pandas as pd


def gbm_returns(
    years: int = 10,
    scenarios: int = 1000,
    mu: float = 0.07,
    sigma: float = 0.15,
    steps_per_year: int = 12,
) -> np.ndarray:
    """Generate periodic returns using Geometric Brownian Motion via Monte Carlo simulation.

    Simulates asset return trajectories by drawing from a normal distribution
    parameterized by annualized drift and volatility, discretized into
    ``steps_per_year`` intervals.

    Args:
        years: Number of years to simulate.
        scenarios: Number of independent trajectories (paths) to generate.
        mu: Annualized drift (expected return).
        sigma: Annualized volatility (standard deviation of returns).
        steps_per_year: Number of discrete time steps per year.

    Returns:
        A ``(steps, scenarios)`` numpy array of periodic returns. The first row
        is always zero (no return at t=0).
    """
    dt = 1.0 / steps_per_year
    steps = int(years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(
        loc=(1 + mu) ** dt,
        scale=sigma * np.sqrt(dt),
        size=(steps, scenarios),
    )
    rets_plus_1[0] = 1.0
    return rets_plus_1 - 1.0


def gbm_prices(
    years: int = 10,
    scenarios: int = 1000,
    mu: float = 0.07,
    sigma: float = 0.15,
    steps_per_year: int = 12,
    s_0: float = 100.0,
) -> pd.DataFrame:
    """Generate price trajectories using Geometric Brownian Motion via Monte Carlo simulation.

    Builds on :func:`gbm_returns` by converting periodic returns into
    cumulative price paths starting from an initial value ``s_0``.

    Args:
        years: Number of years to simulate.
        scenarios: Number of independent trajectories (paths) to generate.
        mu: Annualized drift (expected return).
        sigma: Annualized volatility (standard deviation of returns).
        steps_per_year: Number of discrete time steps per year.
        s_0: Initial asset price at t=0.

    Returns:
        A pandas DataFrame of shape ``(steps, scenarios)`` where each column
        is a simulated price trajectory and each row is a time step. All paths
        start at ``s_0``.
    """
    rets = gbm_returns(
        years=years,
        scenarios=scenarios,
        mu=mu,
        sigma=sigma,
        steps_per_year=steps_per_year,
    )
    return s_0 * pd.DataFrame(rets + 1.0).cumprod()


def discount(
    t: float | np.ndarray | pd.Index,
    r: float,
) -> float | np.ndarray | pd.Series:
    """Compute the price of a pure discount bond that pays $1 at time t.

    Computes :math:`(1 + r)^{-t}`, the present value of \$1 received after
    ``t`` years discounted at the annual rate ``r`` with annual compounding.

    Args:
        t: Time in years until payment.  May be a scalar, numpy array, or
            pandas Index.
        r: Annual interest rate (decimal, e.g. 0.05 for 5%).

    Returns:
        The discount factor(s).  The return type matches the type of ``t``:
        scalar for scalar input, ``np.ndarray`` for array input, and
        ``pd.Series`` for a pandas Index.
    """
    return (1 + r) ** (-t)


def present_value(
    liabilities: pd.Series,
    r: float,
) -> float:
    """Compute the present value of a stream of future liabilities.

    Each liability is discounted from its payment date back to the present
    using annual compounding at rate ``r``.

    Args:
        liabilities: A pandas Series whose index holds payment times in years
            and whose values hold the corresponding cash-flow amounts.
        r: Annual interest rate (decimal, e.g. 0.05 for 5%).

    Returns:
        The total present value (a scalar float) of all liabilities.
    """
    dates = liabilities.index
    discounts = discount(dates, r)
    return (discounts * liabilities).sum()


def funding_ratio(
    assets: float,
    liabilities: pd.Series,
    r: float,
) -> float:
    """Compute the funding ratio of assets to the present value of liabilities.

    The funding ratio measures the proportion of future liabilities that are
    covered by current assets.  A value greater than 1 indicates a surplus
    (assets exceed the present value of liabilities); a value less than 1
    indicates a shortfall.

    Args:
        assets: Current value of assets available to meet the liabilities.
        liabilities: A pandas Series whose index holds payment times in years
            and whose values hold the corresponding cash-flow amounts.
        r: Annual interest rate (decimal, e.g. 0.05 for 5%) used to discount
            the liabilities.

    Returns:
        The funding ratio, ``assets / present_value(liabilities, r)``.
    """
    return assets / present_value(liabilities, r)


def cir(
    n_years: int = 10,
    n_scenarios: int = 100,
    a: float = 0.05,
    b: float = 0.03,
    sigma: float = 0.05,
    steps_per_year: int = 12,
    r_0: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate interest rate paths using the Cox-Ingersoll-Ross (CIR) model.

    Uses Euler–Maruyama discretization of the SDE:

    .. math::
        dr_t = a (b - r_t) dt + \\sigma \\sqrt{r_t} dW_t

    Args:
        n_years: Number of years to simulate.
        n_scenarios: Number of independent trajectories.
        a: Mean-reversion speed.
        b: Long-run mean rate.
        sigma: Volatility.
        steps_per_year: Number of discrete time steps per year.
        r_0: Initial short rate.  Defaults to ``b`` when ``None``.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame of shape ``(steps, n_scenarios)`` with simulated annual
        short rates.  The first row corresponds to :math:`t=0`.
    """
    if r_0 is None:
        r_0 = b
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / steps_per_year
    steps = int(n_years * steps_per_year) + 1

    rates = np.zeros((steps, n_scenarios))
    rates[0, :] = r_0

    for t in range(1, steps):
        r_prev = rates[t - 1, :]
        # Ensure non-negativity before sqrt
        r_prev_pos = np.maximum(r_prev, 0.0)
        dr = a * (b - r_prev) * dt + sigma * np.sqrt(r_prev_pos) * np.sqrt(dt) * np.random.normal(size=n_scenarios)
        rates[t, :] = r_prev + dr

    return pd.DataFrame(rates)



