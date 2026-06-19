# 📈 Finance

Financial experiments in Python — interactive notebooks demonstrating key concepts in quantitative finance, portfolio theory, and risk management, built with [marimo](https://marimo.io).

## 📂 Project Structure

```
finance/
├── demos/                  # Interactive marimo notebooks
│   ├── prices_returns.py       Price series and return calculations
│   ├── normality.py            Distribution analysis of returns
│   ├── annualization.py        Annualizing returns and volatility
│   ├── rolling_window.py       Rolling statistics and windows
│   ├── sharp_ratio.py          Sharpe ratio analysis
│   ├── drawdown.py             Drawdown visualization
│   ├── value_at_risk.py        Value at Risk (VaR) estimation
│   ├── efficient_frontier1.py  Markowitz efficient frontier (2-asset)
│   ├── efficient_frontier2.py  Markowitz efficient frontier (n-asset)
│   ├── tracking_err.py         Tracking error and information ratio
│   ├── crash_correlations.py   Correlation breakdown during crashes
│   ├── candle_sticks.py        Candlestick charting
│   └── valuation_multiples.py  Valuation multiples analysis
├── toolkit/                # Core library
│   ├── data.py                 Data fetching and preprocessing
│   ├── general.py              General-purpose utilities
│   ├── portfolio.py            Portfolio construction & optimization
│   ├── risk.py                 Risk measurement and analytics
│   └── ui.py                   Visualization helpers
├── research/               # 🔬 Personal market research (future)
├── tests/                  # Test suite
└── requirements.txt        # Dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone <repo-url>
cd finance
pip install -r requirements.txt
```

### Running the Notebooks

All demos are [marimo](https://marimo.io) notebooks. To run any demo:

```bash
marimo edit demos/<notebook>.py
```

Or start the marimo server to browse all notebooks:

```bash
marimo edit demos/
```

## 📊 Demos Overview

| Notebook | Topic |
|----------|-------|
| `prices_returns` | Converting price series to returns (simple & log) |
| `normality` | Testing whether returns follow a normal distribution |
| `annualization` | Scaling daily statistics to annual equivalents |
| `rolling_window` | Computing rolling means, volatility, and correlations |
| `sharp_ratio` | Risk-adjusted return measurement |
| `drawdown` | Peak-to-trough decline analysis |
| `value_at_risk` | Parametric, historical, and Monte Carlo VaR |
| `efficient_frontier1` | Two-asset portfolio optimization |
| `efficient_frontier2` | N-asset Markowitz mean-variance optimization |
| `tracking_err` | Benchmark-relative risk metrics |
| `crash_correlations` | How correlations spike during market stress |
| `candle_sticks` | OHLC candlestick charting with volume |
| `valuation_multiples` | P/E, P/B, EV/EBITDA and other multiples |

## 🧪 Research

The `research/` directory is reserved for my own market research — explorations, backtests, and empirical studies beyond textbook theory. This is where future work and findings will live.

## 🛠️ Toolkit

The `toolkit/` package provides reusable building blocks:

- **`data`** — Fetch market data via yfinance, clean and transform series
- **`general`** — Math and statistical helpers
- **`portfolio`** — Portfolio weights, optimization, efficient frontier
- **`risk`** — VaR, CVaR, drawdowns, risk decompositions
- **`ui`** — Plotly and matplotlib-based visualization functions

## 📝 License

MIT © 2026 Szymon Wieloch — see [LICENSE](LICENSE).
