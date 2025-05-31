# lstm_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Close']]

# Preprocess data for LSTM
def preprocess_data(data, window_size=60):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    if len(scaled_data) < window_size + 1:
        raise ValueError(f"Not enough data points to create sequences with window size {window_size}")

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler



# Build and train LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot predictions
def plot_predictions(true, predicted):
    plt.figure(figsize=(10,5))
    plt.plot(true, label='Actual Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data("data/AAPL.csv")
    data = df.values
    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]
    test_data = data[train_size - 60:]

    X_train, y_train, scaler = preprocess_data(train_data)
    X_test, y_test, _ = preprocess_data(test_data)

    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plot_predictions(actual, predictions)
