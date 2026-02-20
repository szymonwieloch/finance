import yfinance as yf
from dateutil.relativedelta import relativedelta
import datetime
import pandas as pd

def get_stock_data(ticker:str|list[str], years=10) -> pd.DataFrame:
  today = datetime.date.today()
  start = today - relativedelta(years=years)
  data = yf.download(ticker, start=start.isoformat(), end=today.isoformat())
  data.index = pd.to_datetime(data.index).to_period('D')
  return data
  
  