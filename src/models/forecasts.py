import numpy as np
import pandas as pd
def prepare_last_window(data_path: str, scalers) -> np.ndarray:
    """Load and prepare the input window for forecasting"""
    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    returns = data['Log Return'].dropna()
    features = pd.DataFrame({
    'Log_Return': returns,
    'Volatility_21D': data['21D Volatility']
    }).dropna()
    last_window = features[['Log_Return', 'Volatility_21D']].iloc[-20:]
    
    scaled = np.column_stack([
        scalers['scaler_returns'].transform(last_window[['Log_Return']]),
        scalers['scaler_vol'].transform(last_window[['Volatility_21D']])
    ])
    return scaled.reshape(1, 20, 2)  # LSTM input shape


def generate_forecasts(
    model, 
    data,
    scalers,
    initial_window: np.ndarray, 
    steps: int = 180,
    n_simulations: int = 100
) -> pd.DataFrame:
    """Generate price forecasts with confidence intervals"""
    forecasts = []
    current_window = initial_window.copy()
    returns = data['Log Return'].dropna()
    volatility = data["21D Volatility"].dropna()
    
    mean_return = returns.mean()
    std_return = returns.std()
    volatility_noise = scalers['scaler_vol'].transform((np.random.uniform(low=0.010576, high=0.04771, size= 20)).reshape(-1,1)).flatten()
    #print("Starting Forecasting...")
    for _ in range(steps):
        # Predict next step
        pred = model.predict(current_window, verbose=0)
        forecast = pred[0,0]
        #adjusted_forecast = forecast + np.random.choice(pred_noise)
        forecasts.append(forecast)
        
        # Update window (remove oldest, add prediction)
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1, 0] = forecast
        current_window[0, -1, 1] = np.random.choice(volatility_noise)
        # Volatility assumed to take a random value within the 21 day volatility range
    #print("Converting forecasted log returns to price")    
    # Convert to price series
    last_price = data['Adj Close'].iloc[-1]
    log_returns = scalers['scaler_returns'].inverse_transform(
        np.array(forecasts).reshape(-1, 1)).flatten()
    
    log_returns += np.random.normal(loc=mean_return, scale=std_return*0.8, size=log_returns.shape)
    price_forecast = last_price * np.exp(log_returns.cumsum())
    
    return pd.DataFrame({
        'date': pd.date_range(start=data.index[-1], periods=steps+1, freq='B')[1:],
        'forecast_price': price_forecast
    })

def simulate_forecasts(model, data, scalers, initial_window, n_simulations=50, days=20):
    """Generate probabilistic forecasts with noise injection"""
    simulations = []
    for _ in range(n_simulations):
        # Add noise to initial window (simulate uncertainty)
        noise = np.random.normal(0, 0.01, initial_window.shape)
        noisy_window = np.clip(initial_window + noise, -1, 1)  # Keep within [-1,1] scaled range
        
        # Generate forecast
        sim = generate_forecasts(model,data,scalers, noisy_window, steps=days)
        simulations.append(sim['forecast_price'])
    
    return pd.DataFrame(simulations).T

def analyze_trends(forecasts: pd.DataFrame, historical_data: pd.DataFrame):
    """Quantify trends and volatility"""
    # Price trends
    forecasts['30D_Slope'] = forecasts['forecast_price'].rolling(30).apply(
        lambda x: np.polyfit(range(30), x, 1)[0])
    
    # Volatility analysis
    forecasts['30D_Vol'] = forecasts['forecast_price'].pct_change().rolling(21).std()
    hist_vol = historical_data['21D Volatility'].iloc[-1]
    
    # Risk zones (highlight 2σ moves)
    forecasts['Upper_Band'] = forecasts['forecast_price'] * (1 + 2*forecasts['30D_Vol'])
    forecasts['Lower_Band'] = forecasts['forecast_price'] * (1 - 2*forecasts['30D_Vol'])
    
    return {
        'slopes': {
            '6M': np.polyfit(range(len(forecasts)), forecasts['forecast_price'], 1)[0],
            '3M': np.polyfit(range(90), forecasts['forecast_price'][:90], 1)[0]
        },
        'vol_ratio': forecasts['30D_Vol'].mean() / hist_vol,
    }