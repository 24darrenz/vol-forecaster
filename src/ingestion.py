import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_price_data(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a given ticker.
    
    Args:
        ticker: e.g. 'SPY', 'TSLA'
        start: 'YYYY-MM-DD'
        end: 'YYYY-MM-DD', defaults to today
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index.name = 'date'
    # flatten MultiIndex columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    
    return df


if __name__ == "__main__":
    df = fetch_price_data("SPY", start="2018-01-01")
    print(df.head())
    print(f"Shape: {df.shape}")