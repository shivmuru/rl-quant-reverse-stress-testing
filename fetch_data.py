"""Download the daily market data used by calibrate.py into data/.

S&P 500, QQQ and VIX come from Yahoo Finance (via yfinance, unofficial);
the 10-year Treasury yield is FRED series DGS10. No API keys needed.

    pip install yfinance pandas-datareader
    python fetch_data.py
"""
from pathlib import Path

import pandas_datareader.data as web
import yfinance as yf

START = "2000-01-01"
out = Path("data")
out.mkdir(exist_ok=True)

for name, ticker in {"spx": "^GSPC", "qqq": "QQQ", "vix": "^VIX"}.items():
    df = yf.download(ticker, start=START, auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)  # flatten yfinance's multi-index
    df.to_csv(out / f"{name}.csv")
    print(name, len(df), df.index.min().date(), df.index.max().date())

dgs10 = web.DataReader("DGS10", "fred", START).dropna()
dgs10.to_csv(out / "dgs10.csv")
print("dgs10", len(dgs10), dgs10.index.min().date(), dgs10.index.max().date())
