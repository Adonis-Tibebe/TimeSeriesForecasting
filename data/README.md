# Data

This directory contains all financial data used in the TimeSeriesForecasting project.

## Directory Structure

```
data/
├── raw/                    # Raw data from Yahoo Finance
│   ├── tsla_raw.csv       # Tesla (TSLA) historical data
│   ├── bnd_raw.csv        # Bond ETF (BND) historical data
│   ├── spy_raw.csv        # S&P 500 ETF (SPY) historical data
│   ├── metadata.json      # Data fetching metadata
│   └── *.dvc              # DVC version control files
├── processed/             # Cleaned and processed data
│   ├── tsla_processed.csv # Processed Tesla data with derived metrics
│   ├── bnd_processed.csv  # Processed BND data with derived metrics
│   ├── spy_processed.csv  # Processed SPY data with derived metrics
│   ├── risk_metrics.csv   # Risk analysis metrics
│   └── *.dvc              # DVC version control files
├── forecasts/             # Model-generated forecasts
│   └── tsla_6m_forecast.csv # 6-month TSLA price forecasts
├── backtesting/           # Portfolio optimization outputs
│   ├── optimal_weights.pkl # Optimal portfolio weights
│   └── asset_prices.csv   # Historical asset prices for backtesting
└── README.md              # This file
```

## Data Sources

- **Raw Data**: Yahoo Finance via `yfinance` library
- **Date Range**: 2015-07-01 to 2025-07-31
- **Assets**: TSLA, BND, SPY
- **Frequency**: Daily data

## Data Processing

Raw data is processed through the `data_processing.ipynb` notebook to:
- Handle missing values (forward-fill prices, zero-fill volume)
- Calculate derived metrics (daily returns, volatility, log returns)
- Validate data quality and completeness

## Forecasting

The `Forecast.ipynb` notebook generates:
- **6-month price forecasts** using trained LSTM model
- **Risk bands** and confidence intervals
- **Trend analysis** for portfolio optimization

## Backtesting

The `portfolio_optimization.ipynb` notebook creates:
- **Optimal portfolio weights** based on Modern Portfolio Theory
- **Asset price data** for historical backtesting analysis
- **Portfolio performance metrics** for risk-return optimization

## Version Control

Data files are tracked using DVC (Data Version Control) for efficient storage and versioning of large files.

## Usage

- **Raw data**: Use for initial data exploration and validation
- **Processed data**: Use for analysis, modeling, and forecasting
- **Risk metrics**: Use for portfolio optimization and risk assessment
- **Forecasts**: Use for strategic planning and risk management
- **Backtesting**: Use for portfolio performance validation and optimization
