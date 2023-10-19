import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import tensorflow as tf
from datetime import datetime, timedelta 
from flask import Flask, jsonify 

app = Flask(__name__)

# Establecer semillas aleatorias para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

@app.route('/')
def home():
    return "Hello"

def get_binance_data(api_key, symbol="ETHUSDT", interval="1h", limit=2000):
    endpoint = f"https://api.binance.com/api/v3/klines"
    url = f"{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"
    
    headers = {
        'X-MBX-APIKEY': api_key,
        'Cache-Control': 'no-cache'  
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size])
    return np.array(X), np.array(Y)

def predict_next_hour_price(model, recent_data, scaler, window_size):
    if len(recent_data) != window_size:
        raise ValueError(f"Expected input of length {window_size}. Got {len(recent_data)}.")
    recent_data_normalized = scaler.transform(np.array(recent_data).reshape(-1, 1))
    input_data = np.reshape(recent_data_normalized, (1, window_size, 1))
    predicted_normalized = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_normalized)
    return predicted_price[0][0]

def train_and_evaluate_model(api_key):
    df = get_binance_data(api_key)
    price = df['close'].astype(float).values
    scaler = MinMaxScaler()
    price_scaled = scaler.fit_transform(price.reshape(-1, 1))
    window_size = 50
    X, Y = create_dataset(price_scaled, window_size)
    tscv = TimeSeriesSplit(n_splits=5)
    model = Sequential([
        GRU(50, input_shape=(window_size, 1), return_sequences=True),
        GRU(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        history = model.fit(X_train, Y_train, epochs=42, validation_data=(X_test, Y_test), batch_size=64, verbose=0)
    loss = model.evaluate(X_test, Y_test)
    last_complete_hour_prices = price[-window_size-1:-1]
    last_complete_hour_timestamps = df['timestamp'][-window_size-1:-1].values
    last_timestamp = pd.to_datetime(last_complete_hour_timestamps[-1])
    predicted_price_next_hour = predict_next_hour_price(model, last_complete_hour_prices, scaler, window_size)
    return last_timestamp, last_complete_hour_prices[-1], predicted_price_next_hour

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import tensorflow as tf
from datetime import datetime, timedelta 
from flask import Flask, jsonify 

app = Flask(__name__)

# Establecer semillas aleatorias para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

@app.route('/')
def home():
    return "Hello"

def get_binance_data(api_key, symbol="ETHUSDT", interval="1h", limit=2000):
    endpoint = f"https://api.binance.com/api/v3/klines"
    url = f"{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"
    
    headers = {
        'X-MBX-APIKEY': api_key,
        'Cache-Control': 'no-cache'  
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size])
    return np.array(X), np.array(Y)

def predict_next_hour_price(model, recent_data, scaler, window_size):
    if len(recent_data) != window_size:
        raise ValueError(f"Expected input of length {window_size}. Got {len(recent_data)}.")
    recent_data_normalized = scaler.transform(np.array(recent_data).reshape(-1, 1))
    input_data = np.reshape(recent_data_normalized, (1, window_size, 1))
    predicted_normalized = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_normalized)
    return predicted_price[0][0]

def train_and_evaluate_model(api_key):
    df = get_binance_data(api_key)
    price = df['close'].astype(float).values
    scaler = MinMaxScaler()
    price_scaled = scaler.fit_transform(price.reshape(-1, 1))
    window_size = 50
    X, Y = create_dataset(price_scaled, window_size)
    tscv = TimeSeriesSplit(n_splits=5)
    model = Sequential([
        GRU(50, input_shape=(window_size, 1), return_sequences=True),
        GRU(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        history = model.fit(X_train, Y_train, epochs=42, validation_data=(X_test, Y_test), batch_size=64, verbose=0)
    loss = model.evaluate(X_test, Y_test)
    last_complete_hour_prices = price[-window_size-1:-1]
    last_complete_hour_timestamps = df['timestamp'][-window_size-1:-1].values
    last_timestamp = pd.to_datetime(last_complete_hour_timestamps[-1])
    predicted_price_next_hour = predict_next_hour_price(model, last_complete_hour_prices, scaler, window_size)
    return last_timestamp, last_complete_hour_prices[-1], predicted_price_next_hour

@app.route('/predict', methods=['GET'])
def predict_price():
    api_key = ""  # Reemplaza esto con tu clave API
    last_timestamp, last_hour_price, predicted_next_hour_price = train_and_evaluate_model(api_key)
    
    return jsonify({
        "last_hour_timestamp": str(last_timestamp - timedelta(hours=5)),
        "last_hour_price": float(last_hour_price),
        "predicted_next_hour_price": float(predicted_next_hour_price)
    })




if __name__ == '__main__':
    app.run()



if __name__ == '__main__':
    app.run()
