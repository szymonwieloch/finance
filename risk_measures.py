import marimo

__generated_with = "0.19.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns
    from toolkit import risk
    import yfinance
    from pytickersymbols import PyTickerSymbols
    import datetime
    import itertools
    return PyTickerSymbols, datetime, itertools, mo, yfinance


@app.cell
def _(PyTickerSymbols, itertools):
    stock_data = PyTickerSymbols()
    sp500_tickers = stock_data.get_yahoo_ticker_symbols_by_index('S&P 500')
    tickers = list(itertools.chain.from_iterable(sp500_tickers))
    return (tickers,)


@app.cell
def _(datetime, mo):
    def get_last_year(d):
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            # This triggers if d is Feb 29 and the previous year isn't a leap year
            return d.replace(year=d.year - 1, day=28)

    today = datetime.date.today()
    year_ago = get_last_year(today)
    date_range = mo.ui.date_range(stop=today.isoformat(), value=(year_ago.isoformat(), today.isoformat()))
    return (date_range,)


@app.cell
def _(mo, tickers):
    stocks = mo.ui.dropdown(tickers, value=tickers[0])
    return (stocks,)


@app.cell(hide_code=True)
def _(date_range, mo, stocks):
    mo.md(rf"""
    {stocks}
    {date_range}
    """)
    return


@app.cell
def _(date_range):
    date_range.value
    return


@app.cell
def _(date_range, yfinance):
    df = yfinance.download("AAPL", start=date_range.value[0], end=date_range.value[1])
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
