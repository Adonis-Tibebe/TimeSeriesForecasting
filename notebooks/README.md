# Notebooks

This directory contains Jupyter notebooks for data processing, exploratory data analysis, and modeling in the TimeSeriesForecasting project.

## data_processing.ipynb

### Purpose
A comprehensive data processing pipeline that cleans and prepares TSLA, BND, and SPY financial data for modeling and analysis.

### Key Objectives
1. **Load raw data** from `data/raw/` directory
2. **Validate data types** and handle missing values
3. **Calculate derived metrics** (returns, volatility)
4. **Save processed data** to `data/processed/` directory

### Data Processing Steps

#### 1. Data Loading and Validation
- Loads raw CSV files for TSLA, BND, and SPY from `data/raw/`
- Validates data types (datetime index, numeric columns)
- Confirms data completeness (2535 entries per ticker from 2015-07-01 to 2025-07-30)

#### 2. Missing Value Handling
- **Price data**: Forward-fill gaps (markets closed on weekends/holidays)
- **Volume data**: Set missing values to 0 on non-trading days
- Ensures no null values remain in processed data

#### 3. Feature Engineering
Creates new derived metrics:
- **Daily Return**: `(Adj Close_t / Adj Close_{t-1}) - 1`
- **21D Volatility**: 21-day rolling standard deviation of returns
- **Log Return**: Natural logarithm of price ratios

#### 4. Data Export
- Saves processed data as CSV files: `{ticker}_processed.csv`
- Files saved to `data/processed/` directory
- Maintains data provenance and processing history

### Key Features
- **Modular design**: Uses utility functions from `src.utils.data_processing_utils`
- **Comprehensive validation**: Ensures data quality and completeness
- **Reproducible pipeline**: Clear step-by-step processing workflow
- **Visual validation**: Includes data distribution plots for verification

### Output Files
- `tsla_processed.csv`: Processed Tesla data with derived metrics
- `bnd_processed.csv`: Processed BND (bond ETF) data with derived metrics
- `spy_processed.csv`: Processed SPY (S&P 500 ETF) data with derived metrics

---

## EDA.ipynb

### Purpose
Comprehensive exploratory data analysis of the processed financial data to identify trends, volatility patterns, and stationarity characteristics.

### Key Objectives
1. **Price trend analysis** across different asset classes
2. **Return distribution analysis** and outlier detection
3. **Volatility pattern identification** and risk assessment
4. **Stationarity testing** for modeling suitability
5. **Correlation analysis** for portfolio construction

### Analysis Sections

#### 1. Price Trends Over Time
**Key Findings:**
- **TSLA**: Exponential growth (10x from 2020-2022) with high volatility
- **SPY**: Steady growth (~12% CAGR) with market-typical drawdowns
- **BND**: Flat trend with minimal price appreciation (yield-driven returns)

**Financial Implications:**
- Diversification benefits from BND during market crashes
- TSLA outperformed but required tolerance for 50%+ drawdowns
- SPY provides efficient baseline for benchmarking

#### 2. Daily Returns and Volatility Analysis
**Distribution Characteristics:**
- **TSLA**: Fat-tailed distribution with frequent extreme moves
- **SPY**: More normal distribution with occasional outliers
- **BND**: Tight distribution with minimal volatility

**Outlier Analysis (>3σ daily returns):**
- **TSLA**: 46 outlier days, largest swing ±22.7%
- **BND**: 30 outlier days, tight range ±5.4%
- **SPY**: 36 outlier days, more negative outliers

#### 3. Stationarity Analysis
**Augmented Dickey-Fuller Test Results:**
- **Prices**: Non-stationary (p-values > 0.05) - differencing required
- **Returns & Log Returns**: Stationary (p-values ≈ 0.000) - suitable for modeling

**Key Takeaway**: Model returns (not prices) for time series forecasting

#### 4. Risk-Return Profiles

| Metric           | TSLA     | SPY      | BND      |
|------------------|----------|----------|----------|
| **Max Drawdown** | −109.38% | −38.16%  | −20.23%  |
| **Sharpe Ratio** | 0.73     | 0.67     | −0.02    |
| **VaR (95%)**    | −5.33%   | −1.69%   | −0.48%   |
| **Volatility**   | 58.10%   | 17.91%   | 5.39%    |

#### 5. Correlation Analysis
**Daily Returns Correlation Matrix:**
- **TSLA vs SPY**: 0.49 (moderate correlation)
- **TSLA vs BND**: 0.06 (very low correlation)
- **SPY vs BND**: 0.11 (weak correlation)

**Portfolio Insight**: BND provides effective diversification when combined with TSLA or SPY

### Key Insights

#### Risk Assessment
1. **TSLA**: High-growth but extreme risk (VaR >5%, Sharpe <1) - requires active management
2. **SPY**: Efficient baseline (Sharpe ~0.7) with predictable risk profile
3. **BND**: Effective hedge despite 2022 anomaly

#### Portfolio Construction
- **60% SPY / 30% TSLA / 10% BND** blend would have captured growth while mitigating 2022 losses
- BND allocation of 20% reduces portfolio VaR by ~18-20% vs. 100% SPY or TSLA

### Output Files
- `risk_metrics.csv`: Comprehensive risk metrics for modeling phase

### Next Steps Identified
1. **Modeling**: ARIMA/SARIMA for SPY/BND, LSTM + GARCH for TSLA
2. **Optimization**: Use risk metrics to constrain Efficient Frontier
3. **Backtesting**: Simulate portfolio strategies with stress testing

---

## modeling.ipynb

### Purpose
Comprehensive time series forecasting analysis comparing ARIMA and LSTM models for Tesla stock returns prediction, with model evaluation, selection, and deployment.

### Key Objectives
1. **Implement ARIMA modeling** with automatic parameter selection
2. **Build LSTM neural network** with volatility features
3. **Compare model performance** using multiple metrics
4. **Select optimal model** based on business requirements
5. **Save trained models** for production inference

### Modeling Approach

#### 1. ARIMA (AutoRegressive Integrated Moving Average)
**Implementation:**
- Uses `pmdarima.auto_arima` for automatic parameter selection
- Best model: ARIMA(2,0,2) with intercept
- Trains on 2015-2023 data, tests on 2024 data

**Key Features:**
- Linear statistical model with interpretable coefficients
- Fast inference and training
- Handles basic trends and seasonality

**Limitations Identified:**
- Cannot capture non-linear patterns
- Assumes constant variance (no volatility clustering)
- Struggles with extreme market movements

#### 2. LSTM (Long Short-Term Memory)
**Architecture:**
```python
Sequential([
    LSTM(200, return_sequences=True, recurrent_dropout=0.1),
    Dropout(0.2),
    LSTM(100),
    Dense(1)
])
```

**Key Features:**
- 20-day lookback window for sequence creation
- Incorporates volatility features (21D rolling volatility)
- Uses MinMaxScaler for feature normalization
- Early stopping to prevent overfitting

**Training Configuration:**
- **Optimizer**: Adam with learning rate 0.0005
- **Loss**: Mean Squared Error (MSE)
- **Batch Size**: 128
- **Epochs**: 60 with early stopping (patience=5)

### Model Performance Comparison

| Metric | ARIMA | LSTM |
|--------|-------|------|
| **MAE** | 0.029097 | 0.028221 |
| **RMSE** | 0.041115 | 0.039595 |
| **MAPE** | 107.39% | 514.12% |

**Key Findings:**
- **LSTM outperforms ARIMA** on MAE and RMSE metrics
- **ARIMA shows better MAPE** due to handling of zero returns
- **LSTM captures volatility patterns** better than linear ARIMA

### Model Selection Rationale

#### Why LSTM Was Chosen
1. **Volatility Sensitivity**: Better captures TSLA's non-linear risk dynamics
2. **Feature Extensibility**: Easy to add macro indicators and cross-asset correlations
3. **Business Alignment**: Fits GMF's focus on cutting-edge technology
4. **Automated Retraining**: Supports operational workflows

#### Tradeoffs Accepted
- ~5% higher computational cost
- Requires periodic volatility rescaling
- Less responsive to extreme outliers but better volatility reaction

### Data Preparation
- **Features**: Log returns + 21D volatility
- **Sequence Creation**: 20-day lookback window
- **Train/Test Split**: 80/20 chronological split
- **Scaling**: MinMaxScaler applied to each feature separately

### Model Persistence
**Saved Models:**
- `arima_model.pkl`: Trained ARIMA model
- `lstm_model.h5`: Trained LSTM model (Keras format)
- `lstm_scalers.pkl`: Preprocessing scalers and parameters

**Storage Location:** `../models/` directory

### Key Insights

#### Model Strengths
1. **ARIMA**: Excellent baseline for stable market periods
2. **LSTM**: Superior for volatile, non-linear market conditions

#### Business Applications
1. **Portfolio Management**: Use LSTM for dynamic allocation strategies
2. **Risk Management**: Leverage volatility features for position sizing
3. **Trading Signals**: Combine both models for robust decision making

### Next Steps
1. **Feature Engineering**: Add macro indicators (VIX, interest rates)
2. **Ensemble Methods**: Combine ARIMA and LSTM predictions
3. **Real-time Deployment**: Implement model serving infrastructure
4. **Performance Monitoring**: Track model drift and retraining needs

---

## Forecast.ipynb

### Purpose
Generates 6-month price forecasts using the trained LSTM model and provides trend analysis for portfolio optimization.

### Key Features
- **6-Month Forecasting**: Uses pre-trained LSTM model to predict TSLA prices
- **Monte Carlo Simulation**: Generates confidence intervals and risk bands
- **Trend Analysis**: Calculates 6M/3M slopes and volatility ratios
- **Risk Assessment**: Identifies high-volatility periods and risk zones

### Output
- **Forecast Data**: 180-day price predictions saved to `data/forecasts/tsla_6m_forecast.csv`
- **Visualizations**: Price forecasts with confidence bands and risk zones
- **Analysis**: Trend strength and volatility comparison metrics

### Dependencies
- Trained LSTM model from `models/lstm_model.h5`
- Preprocessing scalers from `models/lstm_scalers.pkl`
- Functions from `src.models.forecasts` module

---

## Usage Instructions

### Prerequisites
1. Ensure raw data is available in `data/raw/` directory
2. Install required dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`
3. Run notebooks in order: `data_processing.ipynb` → `EDA.ipynb` → `modeling.ipynb`

### Running the Notebooks
```bash
# Navigate to notebooks directory
cd notebooks

# Start Jupyter
jupyter lab

# Run notebooks in sequence
# 1. data_processing.ipynb
# 2. EDA.ipynb
# 3. modeling.ipynb
```

### Dependencies
- **Core**: `pandas`, `numpy`, `matplotlib`
- **Visualization**: `seaborn`
- **Statistics**: `statsmodels`
- **Machine Learning**: `tensorflow`, `keras`, `pmdarima`, `sklearn`
- **Utilities**: `pathlib`, `joblib`

### Data Flow
```
data/raw/ → data_processing.ipynb → data/processed/ → EDA.ipynb → insights/ → modeling.ipynb → models/
```

### Notes
- All three notebooks are designed to be run sequentially
- Data processing notebook must be run first to generate processed data
- EDA notebook provides comprehensive analysis for modeling decisions
- Modeling notebook requires significant computational resources for LSTM training
- All visualizations, metrics, and trained models are saved for future reference
