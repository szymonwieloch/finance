import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import yfinance as yf
    import marimo as mo

    return mo, pd, yf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Valuation Multiples

    We can compare stocks P/E and P/B ratios with comparable companies in the same industry and use it as buy or sell signals.
    """)
    return


@app.cell
def _():
    def price_to_book(ticker) -> float:
        # shares = ticker.info.get('sharesOutstanding')
        # price = ticker.info.get('currentPrice')
        market_cap = ticker.info.get('marketCap')
        total_assets = ticker.balance_sheet.loc['Total Assets'].iloc[0]
        total_liabilities = ticker.balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
        return market_cap/(total_assets - total_liabilities)

    def price_to_earnings(ticker) -> float:
        return ticker.info.get('forwardPE')
    
    

    return price_to_book, price_to_earnings


@app.cell
def _(yf):
    TICKER = 'BP'
    CMOMPARABLES = tickers = ["XOM", "CVX", "SHEL", "TTE", "E", "EQNR", "COP", "PBR", "SU", "IMO", "REPYY", "EOG", "OXY", "DVN", "CNQ"]
    ALL = [TICKER] + CMOMPARABLES

    tickers = yf.Tickers(' '.join(ALL))
    return ALL, TICKER, tickers


@app.cell
def _(ALL, pd, price_to_book, price_to_earnings, tickers):
    tbl = pd.DataFrame({name: [price_to_book(tickers.tickers[name]), price_to_earnings(tickers.tickers[name])] for name in ALL}, index=['P/B', 'P/E']).T
    return (tbl,)


@app.cell
def _(tbl):
    tbl
    return


@app.cell
def _(TICKER, tbl):
    comparables = tbl.drop(TICKER)
    pb_avg = comparables['P/B'].mean()
    pe_avg = comparables['P/E'].mean()
    ticker_pb = tbl.loc[TICKER, 'P/B']
    ticker_pe = tbl.loc[TICKER, 'P/E']
    pb_signal =  'buy' if pb_avg > ticker_pb else 'sell'
    pe_signal =  'buy' if pe_avg > ticker_pe else 'sell'
    return pb_avg, pb_signal, pe_avg, pe_signal, ticker_pb, ticker_pe


@app.cell(hide_code=True)
def _(TICKER, mo, pb_avg, pb_signal, pe_avg, pe_signal, ticker_pb, ticker_pe):
    mo.md(rf"""
    ## Analysis

    ### P/B

    {TICKER} P/B value is {ticker_pb} while comparable companies have {pb_avg}. This is a {pb_signal} signal.

    ### P/E

    {TICKER} P/E value is {ticker_pe} while comparable companies have {pe_avg}. This is a {pe_signal} signal.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
