# TimeSeriesForecasting

A comprehensive Python project for financial time series forecasting and analysis, focusing on TSLA, BND, and SPY data with advanced data processing, exploratory analysis, and modeling capabilities.

## Business Objective

**Guide Me in Finance (GMF) Investments** is a forward-thinking financial advisory firm that specializes in personalized portfolio management. GMF leverages cutting-edge technology and data-driven insights to provide clients with tailored investment strategies. By integrating advanced time series forecasting models, GMF aims to predict market trends, optimize asset allocation, and enhance portfolio performance. The company's goal is to help clients achieve their financial objectives by minimizing risks and capitalizing on market opportunities.

### Situational Overview

As a Financial Analyst at GMF Investments, our objective is to apply time series forecasting to historical financial data to enhance portfolio management strategies. our role involves analyzing data, building predictive models, and recommending portfolio adjustments based on forecasted trends.

**You will:**
- Utilize YFinance data to extract historical financial information such as stock prices, market indices, and other relevant financial metrics
- Preprocess and analyze this data to identify trends and patterns
- Develop and evaluate forecasting models to predict future market movements
- Use the insights gained to recommend changes to client portfolios that aim to optimize returns while managing risks

### Industry Context

The **Efficient Market Hypothesis** suggests that predicting exact stock prices using only historical price data is exceptionally difficult. Therefore, in an industry setting, these models are more often used to:
- **Forecast volatility** for risk management
- **Identify momentum factors** for trend analysis
- **Serve as inputs** into larger decision-making frameworks
- **Support portfolio optimization** rather than direct, standalone price prediction

At GMF Investments, financial analysts play a crucial role in interpreting complex financial data and providing actionable insights. By utilizing real-time financial data from sources like YFinance, GMF ensures its strategies are based on the latest market conditions, thereby maintaining a competitive edge.

## Project Overview

This project provides a complete pipeline for financial time series forecasting, from data acquisition to model development and deployment. It includes:

- **Data Acquisition**: Automated fetching of financial data from Yahoo Finance
- **Data Processing**: Comprehensive cleaning, validation, and feature engineering
- **Exploratory Analysis**: Deep dive into trends, volatility patterns, and risk metrics
- **Modeling Framework**: Advanced time series forecasting with ARIMA and LSTM models
- **Model Evaluation**: Comprehensive performance metrics and model selection
- **Model Persistence**: Trained models saved for production inference
- **Testing**: Unit and integration tests for data processing and modeling utilities

## Key Features

- **Multi-Asset Analysis**: Support for TSLA (Tesla), BND (Bond ETF), and SPY (S&P 500 ETF)
- **Automated Data Pipeline**: End-to-end data processing from raw to analysis-ready
- **Comprehensive EDA**: Advanced statistical analysis and risk assessment
- **Advanced Modeling**: ARIMA and LSTM models for time series forecasting
- **Model Comparison**: Performance evaluation and selection based on business requirements
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Production Ready**: Testing, logging, configuration management, and model persistence
- **Industry-Focused**: Designed for real-world financial analysis and portfolio management

## Project Structure

```
TimeSeriesForecasting/
├── config/                 # Configuration settings
│   ├── __init__.py
│   └── settings.py
├── data/                   # Data storage
│   ├── raw/               # Raw data from Yahoo Finance
│   ├── processed/         # Cleaned and processed data
│   └── README.md
├── docs/                  # Project documentation
│   └── README.md
├── examples/              # Usage examples
│   ├── __init__.py
│   └── README.md
├── models/                # Trained and saved models
│   ├── arima_model.pkl   # Trained ARIMA model
│   ├── lstm_model.h5     # Trained LSTM model
│   └── lstm_scalers.pkl  # LSTM preprocessing scalers
├── notebooks/             # Jupyter notebooks
│   ├── data_processing.ipynb  # Data cleaning and feature engineering
│   ├── EDA.ipynb             # Exploratory data analysis
│   ├── modeling.ipynb        # ARIMA and LSTM modeling
│   └── README.md
├── scripts/               # Utility scripts
│   ├── data_fetcher.py   # Yahoo Finance data downloader
│   └── README.md
├── src/                   # Source code
│   ├── core/             # Core functionality
│   ├── models/           # Forecasting models and utilities
│   │   ├── Models_utils.py  # Model utility functions
│   │   └── __init__.py
│   ├── services/         # Business logic services
│   ├── utils/            # Utility functions
│   │   ├── data_processing_utils.py
│   │   └── __init__.py
│   └── __init__.py
├── tests/                # Test suite
│   ├── integration/      # Integration tests
│   ├── unit/            # Unit tests
│   │   ├── test_data_processing_utils.py
│   │   └── test_models_utils.py
│   └── __init__.py
├── .gitignore           # Git ignore rules
├── Makefile             # Build and development commands
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- GPU support recommended for LSTM training (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TimeSeriesForecasting
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Pipeline

1. **Fetch raw data**
   ```bash
   cd scripts
   python data_fetcher.py --tickers TSLA BND SPY --start 2015-07-01 --end 2025-07-31
   ```

2. **Process and analyze data**
   ```bash
   cd notebooks
   jupyter lab
   ```
   - Run `data_processing.ipynb` to clean and prepare data
   - Run `EDA.ipynb` for comprehensive analysis
   - Run `modeling.ipynb` for ARIMA and LSTM modeling

## Data Pipeline

### 1. Data Acquisition (`scripts/data_fetcher.py`)

- **Source**: Yahoo Finance via `yfinance` library
- **Assets**: TSLA, BND, SPY (configurable)
- **Date Range**: 2015-07-01 to 2025-07-31 (configurable)
- **Output**: Raw CSV files in `data/raw/`

**Usage:**
```bash
# Default (TSLA, BND, SPY from 2015-2025)
python data_fetcher.py

# Custom tickers and date range
python data_fetcher.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2024-12-31
```

### 2. Data Processing (`notebooks/data_processing.ipynb`)

- **Input**: Raw CSV files from `data/raw/`
- **Processing**:
  - Data type validation and cleaning
  - Missing value handling (forward-fill for prices, zero-fill for volume)
  - Feature engineering (daily returns, volatility, log returns)
- **Output**: Processed CSV files in `data/processed/`

### 3. Exploratory Analysis (`notebooks/EDA.ipynb`)

- **Price Trends**: Historical price analysis across assets
- **Risk Metrics**: VaR, Sharpe ratio, max drawdown, volatility
- **Stationarity Testing**: Augmented Dickey-Fuller tests
- **Correlation Analysis**: Portfolio diversification insights
- **Outlier Detection**: Statistical analysis of extreme returns

### 4. Modeling (`notebooks/modeling.ipynb`)

- **ARIMA Modeling**: Automatic parameter selection with `pmdarima.auto_arima`
- **LSTM Neural Network**: Deep learning model with volatility features
- **Model Comparison**: Performance evaluation using MAE, RMSE, and MAPE
- **Model Selection**: Business-driven model choice based on requirements
- **Model Persistence**: Trained models saved for production use

## 🔍 Key Insights

### Risk-Return Profiles

| Metric           | TSLA     | SPY      | BND      |
|------------------|----------|----------|----------|
| **Max Drawdown** | −109.38% | −38.16%  | −20.23%  |
| **Sharpe Ratio** | 0.73     | 0.67     | −0.02    |
| **VaR (95%)**    | −5.33%   | −1.69%   | −0.48%   |
| **Volatility**   | 58.10%   | 17.91%   | 5.39%    |

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

### Portfolio Construction

- **60% SPY / 30% TSLA / 10% BND** blend captures growth while mitigating losses
- **BND allocation** of 20% reduces portfolio VaR by ~18-20%
- **Diversification benefits** from low correlation between assets
- **LSTM model** selected for future tasks due to superior volatility capture

## Development

### Project Structure

- **`src/utils/`**: Core data processing utilities
- **`src/models/`**: Time series forecasting models and utilities
  - `Models_utils.py`: Sequence creation and metrics calculation functions
- **`src/services/`**: Business logic services (future)
- **`src/core/`**: Core functionality (future)

### Testing

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run specific test file
python -m pytest tests/unit/test_models_utils.py
```

### Code Quality

- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration test coverage
- **Logging**: Structured logging throughout

## Documentation

- **`notebooks/README.md`**: Detailed notebook documentation including modeling
- **`scripts/README.md`**: Script usage and examples
- **`docs/`**: Project documentation (future)
- **`examples/`**: Usage examples (future)

## Configuration


### Settings

Configuration can be customized in `config/settings.py`:

```python
# Data processing settings
DEFAULT_TICKERS = ["TSLA", "BND", "SPY"]
DEFAULT_START_DATE = "2015-07-01"
DEFAULT_END_DATE = "2025-07-31"

# Analysis settings
VOLATILITY_WINDOW = 21
CORRELATION_THRESHOLD = 0.3

# Modeling settings
LSTM_WINDOW_SIZE = 20
ARIMA_SEASONAL = False
```

## Model Deployment

### Saved Models

The project includes pre-trained models in the `models/` directory:

- **`arima_model.pkl`**: Trained ARIMA(2,0,2) model for TSLA returns
- **`lstm_model.h5`**: Trained LSTM model with volatility features
- **`lstm_scalers.pkl`**: Preprocessing scalers and parameters

### Model Usage

```python
import joblib
from tensorflow import keras

# Load ARIMA model
arima_model = joblib.load('models/arima_model.pkl')

# Load LSTM model
lstm_model = keras.models.load_model('models/lstm_model.h5')

# Load scalers
scalers = joblib.load('models/lstm_scalers.pkl')
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write tests for new functionality
- Update documentation for API changes
- Ensure model performance meets business requirements

## License

None

## Acknowledgments

- **Yahoo Finance**: Data source via `yfinance` library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Statsmodels**: Statistical analysis
- **TensorFlow/Keras**: Deep learning framework
- **PMDARIMA**: Automatic ARIMA parameter selection
- **Scikit-learn**: Machine learning utilities

---

**Note**: This project is for educational and research purposes. Financial decisions should not be based solely on this analysis. The models and insights provided are intended to support informed decision-making as part of a comprehensive investment strategy, not as standalone investment advice.
