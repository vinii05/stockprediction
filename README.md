# Stock Price Prediction using LSTM with Push Notifications

This project uses deep learning (LSTM) to predict stock prices for major companies like Apple, Tesla, Google, and Amazon. It leverages historical stock data from Yahoo Finance and sends mobile notifications for the stock predicted to have the highest growth.

##  Problem Statement

Stock market investments can be risky due to unpredictable trends. Investors need reliable tools that use past data to make informed predictions. This project aims to help by predicting stock prices using machine learning.

---

##  Proposed Solution

The solution is proposed using LSTM (Long Short-Term Memory) neural networks to predict stock closing prices based on historical data. The model is trained on past stock prices, and the stock with the highest predicted growth is sent as a mobile notification using Pushbullet.

---

##  Technology Used

- **Python**  
- **TensorFlow / Keras**  
- **Yahoo Finance API (yfinance)**  
- **Matplotlib** (for visualization)  
- **scikit-learn** (for data preprocessing)  
- **Pushbullet API** (for mobile notifications)

---

##  Machine Learning Algorithm

### Algorithm Used:
- **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) that works well with time-series data like stock prices.

### Input Features:
- Only the **Closing Price** of each stock is used.

### Training Process:
- 80% of the historical closing price data is used to train the model.
- Data is normalized using MinMaxScaler.
- The model is trained for 30 epochs.

### Prediction Process:
- The trained model predicts prices for the remaining 20% of the data.
- The model compares predicted vs actual prices and calculates expected growth.
- A notification is sent for the stock with the highest predicted growth.

---

## Results & Visualizations

Each stock's predicted vs actual prices are plotted.  
The model performs well in identifying upward trends in stock prices.

## Notifications

The Pushbullet API is used to send real-time mobile alerts for the most promising stock prediction.

---

## Dataset Source

Historical stock data is downloaded using [Yahoo Finance](https://finance.yahoo.com/).  
You can view or download the data here:

- [Apple (AAPL)](https://finance.yahoo.com/quote/AAPL/history)
- [Tesla (TSLA)](https://finance.yahoo.com/quote/TSLA/history)
- [Google (GOOGL)](https://finance.yahoo.com/quote/GOOGL/history)
- [Amazon (AMZN)](https://finance.yahoo.com/quote/AMZN/history)

---

##  Future Improvements

- Add more technical indicators (e.g., RSI, MACD).
- Incorporate real-time stock data.
- Improve model accuracy using hybrid models (e.g., CNN + LSTM).
- Create a live dashboard for continuous monitoring.

---

##  References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Yahoo Finance Python API (yfinance)](https://pypi.org/project/yfinance/)
- [Pushbullet API](https://docs.pushbullet.com/)

---

##  Note

This project is for educational purposes only and should not be used for real-time investment decisions.

