# src/utils/data_processing_utils.py
from pathlib import Path
import pandas as pd
import numpy as np

def load_ticker_data(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load and validate raw ticker data."""
    df = pd.read_csv(
        data_dir / f"{ticker.lower()}_raw.csv",
        parse_dates=["Date"],
        index_col="Date"
    )
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert all(col in df.columns for col in expected_cols), f"Missing columns in {ticker}"
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to business days, forward-fill prices, zero-fill volume."""
    df = df.asfreq("B")
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    df[price_cols] = df[price_cols].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    return df

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns and volatility metrics."""
    df = df.copy()
    df["Daily Return"] = df["Adj Close"].pct_change()
    df["Log Return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))  
    df["21D Volatility"] = df["Daily Return"].rolling(21).std()
    return df