import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pushbullet import Pushbullet 

PUSHBULLET_API_KEY = "   " #replace the empty space with your api key

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data[['Close']]

def prepare_data(data, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def send_notification(title, message):
    try:
        pb = Pushbullet(PUSHBULLET_API_KEY)
        pb.push_note(title, message)
        print(f"Notification sent: {title}")
    except Exception as e:
        print(f"Failed to send notification: {e}")

def main():
    try:
        stocks = ['AAPL', 'TSLA', 'GOOGL', 'AMZN']  
        start_date = '2020-01-01'
        end_date = '2024-01-01'
        best_stock = None
        best_growth = -np.inf

        predictions_dict = {}

        for ticker in stocks:
            print(f"\nProcessing {ticker}...")

            data = get_stock_data(ticker, start_date, end_date)
            X, y, scaler = prepare_data(data.values)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
            actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

            predicted_growth = (predictions[-1] - actual_prices[-1]) / actual_prices[-1] * 100
            predictions_dict[ticker] = predicted_growth

            if predicted_growth > best_growth:
                best_growth = predicted_growth
                best_stock = ticker

            plt.figure(figsize=(10,5))
            plt.plot(actual_prices, label='Actual Price', color='blue')
            plt.plot(predictions, label='Predicted Price', color='red', linestyle='dashed')
            plt.title(f'Stock Price Prediction for {ticker}')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()

        if best_stock:
            message = f"The stock with the highest predicted growth is {best_stock} with a {best_growth[0]:.2f}% increase."
            send_notification("Stock Prediction Alert", message)
            print(message)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
