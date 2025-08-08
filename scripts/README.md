# Scripts

This directory contains utility scripts for the TimeSeriesForecasting project.

## data_fetcher.py

A utility script for fetching financial data from Yahoo Finance using the `yfinance` library.

### Purpose

The `data_fetcher.py` script downloads historical stock data for specified tickers and saves them as CSV files in the `data/raw/` directory. It also creates a metadata file to track when the data was fetched and what parameters were used.

### Features

- Downloads historical stock data from Yahoo Finance
- Supports multiple tickers in a single run
- Configurable date ranges
- Automatic CSV file generation with standardized naming
- Metadata tracking for data provenance
- Comprehensive logging
- Error handling and validation

### Usage

#### Command Line Interface

```bash
# Basic usage with default parameters (TSLA, BND, SPY from 2015-07-01 to 2025-07-31)
python data_fetcher.py

# Specify custom tickers
python data_fetcher.py --tickers AAPL MSFT GOOGL

# Specify custom date range
python data_fetcher.py --start 2020-01-01 --end 2024-12-31

# Full custom example
python data_fetcher.py --tickers TSLA BND SPY --start 2015-07-01 --end 2025-07-31
```

#### Parameters

- `--tickers`: List of stock tickers to fetch (default: `["TSLA", "BND", "SPY"]`)
- `--start`: Start date in YYYY-MM-DD format (default: `"2015-07-01"`)
- `--end`: End date in YYYY-MM-DD format (default: `"2025-07-31"`)

### Output

The script creates the following files in the `data/raw/` directory:

- `{ticker}_raw.csv`: Historical data for each ticker (e.g., `tsla_raw.csv`, `bnd_raw.csv`, `spy_raw.csv`)
- `metadata.json`: Metadata file containing:
  - `fetched_at`: Timestamp when data was fetched
  - `tickers`: List of tickers that were processed
  - `date_range`: Start and end dates used for fetching

### Data Format

Each CSV file contains the following columns:
- `Open`: Opening price
- `High`: Highest price during the day
- `Low`: Lowest price during the day
- `Close`: Closing price
- `Adj Close`: Adjusted closing price
- `Volume`: Trading volume

### Dependencies

- `yfinance`: Yahoo Finance data downloader
- `pandas`: Data manipulation and CSV handling
- `argparse`: Command line argument parsing
- `logging`: Logging functionality

### Error Handling

The script includes comprehensive error handling:
- Validates that data was successfully retrieved from Yahoo Finance
- Creates necessary directories if they don't exist
- Logs all operations and errors
- Raises exceptions for critical failures

### Examples

#### Example 1: Fetch default data
```bash
python data_fetcher.py
```
This will fetch TSLA, BND, and SPY data from 2015-07-01 to 2025-07-31.

#### Example 2: Fetch specific tech stocks
```bash
python data_fetcher.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2024-12-31
```
This will fetch Apple, Microsoft, and Google data for the year 2020.

#### Example 3: Fetch single ticker
```bash
python data_fetcher.py --tickers TSLA --start 2023-01-01 --end 2024-01-01
```
This will fetch only Tesla data for the year 2023.

### Notes

- The script automatically creates the `data/raw/` directory if it doesn't exist
- All ticker names are converted to lowercase in the output filenames
- The script uses Yahoo Finance's `auto_adjust=False` setting to preserve original data
- Data is grouped by ticker for efficient processing
- Progress is displayed during the download process
