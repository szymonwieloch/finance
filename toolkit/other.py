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