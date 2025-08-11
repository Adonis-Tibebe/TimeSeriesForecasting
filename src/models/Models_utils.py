import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create sequences (lookback = 20 days)
def create_sequences(data, window=20):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def calculate_metrics(actual, predicted):
    for i in range(0,2):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mask = actual != 0
        mape = np.mean(np.abs((actual - (predicted)[mask])/(actual)[mask])) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}