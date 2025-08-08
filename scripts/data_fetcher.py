import yfinance as yf
import pandas as pd
import logging
import argparse
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(tickers: list, start_date: str, end_date: str) -> None:
    """Fetch ticker data from YFinance and save to CSV."""
    RAW_DATA_DIR = "../data/raw"
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    try:
        logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            progress=True,
            auto_adjust=False
        )
        
        if data.empty:
            raise ValueError("No data returned from YFinance!")
        
        # Save each ticker to separate CSV
        for ticker in tickers:
            ticker_data = data[ticker]
            filepath = f"{RAW_DATA_DIR}/{ticker.lower()}_raw.csv"
            ticker_data.to_csv(filepath)
            logger.info(f"Saved {ticker} data to {filepath}")
        
        # Save metadata
        metadata = {
            "fetched_at": pd.Timestamp.now().isoformat(),
            "tickers": tickers,
            "date_range": [start_date, end_date]
        }
        with open(f"{RAW_DATA_DIR}/metadata.json", "w") as f:
            json.dump(metadata, f)
            
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["TSLA", "BND", "SPY"])
    parser.add_argument("--start", default="2015-07-01")
    parser.add_argument("--end", default="2025-07-31")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fetch_data(tickers=args.tickers, start_date=args.start, end_date=args.end)