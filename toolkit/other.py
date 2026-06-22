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



def inst_to_ann(
    r: float | np.ndarray,
) -> float | np.ndarray:
    """Convert an instantaneous (continuously compounded) rate to an annual rate.

    Computes :math:`e^{r} - 1`, the annual effective rate equivalent to a
    continuously compounded rate ``r``.

    Args:
        r: Instantaneous interest rate (decimal, e.g. 0.05 for 5%).
            May be a scalar float or a numpy array.

    Returns:
        The equivalent annual rate(s).  Return type matches the type of ``r``.
    """
    return np.expm1(r)


def ann_to_inst(
    r: float | np.ndarray,
) -> float | np.ndarray:
    """Convert an annual interest rate to an instantaneous (continuously compounded) rate.

    Computes :math:`\\ln(1 + r)`, the continuously compounded rate equivalent
    to the annual effective rate ``r``.

    Args:
        r: Annual interest rate (decimal, e.g. 0.05 for 5%).
            May be a scalar float or a numpy array.

    Returns:
        The equivalent instantaneous rate(s).  Return type matches the type
        of ``r``.
    """
    return np.log1p(r)


def cir(
    n_years: int = 10,
    n_scenarios: int = 1,
    a: float = 0.05,
    b: float = 0.03,
    sigma: float = 0.05,
    steps_per_year: int = 12,
    r_0: float | None = None,
) -> pd.DataFrame:
    """Generate random interest rate paths using the Cox-Ingersoll-Ross (CIR) model.

    Simulates the short-rate process

    .. math::

        dr_t = a (b - r_t) dt + \\sigma \\sqrt{r_t} dW_t

    using a straightforward Euler–Maruyama discretisation.  The parameters
    ``b`` and ``r_0`` are supplied as *annual* rates; they are internally
    converted to instantaneous form for the simulation, and the output is
    converted back to annual rates.

    Args:
        n_years: Number of years to simulate.
        n_scenarios: Number of independent rate paths (scenarios).
        a: Mean-reversion speed (positive).
        b: Long-run mean of the *annual* interest rate (decimal).
        sigma: Annualized volatility of the short rate.
        steps_per_year: Number of discretisation steps per year.
        r_0: Initial annual interest rate (decimal).  Defaults to ``b``
            when ``None``.

    Returns:
        A pandas DataFrame of shape ``(steps, n_scenarios)`` where each
        column is a simulated path of annual interest rates and each row
        is a time step.  All paths start from ``r_0`` (or ``b``).
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    b_inst = ann_to_inst(b)
    dt = 1.0 / steps_per_year
    num_steps = int(n_years * steps_per_year) + 1

    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b_inst - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        # use abs() to guard against tiny negative values from discretisation
        rates[step] = abs(r_t + d_r_t)

    return pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
