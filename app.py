# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lstm_model import load_data, preprocess_data, build_lstm_model
from rl_trading_env import StockTradingEnv
from stable_baselines3 import PPO
import numpy as np
import os

st.set_page_config(page_title="Stock Predictor & Trader", layout="wide")

st.title("📈 Stock Price Prediction & RL Trading Bot")

# Sidebar
st.sidebar.header("Settings")
stock_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if stock_file:
    df = pd.read_csv(stock_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    st.subheader("Raw Stock Data")
    st.dataframe(df.tail())

    # Show columns in the app for debugging
    st.write("Columns in uploaded CSV:", df.columns.tolist())

    # Find a column that ends with 'close' (case-insensitive)
    close_cols = [col for col in df.columns if col.lower().endswith('close')]

    if close_cols:
        st.line_chart(df[close_cols[0]])
    else:
        st.warning("No 'Close' column found. Available columns: " + ", ".join(df.columns))

    # --- LSTM PREDICTION ---
    st.subheader("🔮 LSTM Price Prediction")

    # Save uploaded file temporarily
    temp_path = "data/temp.csv"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    df.to_csv(temp_path)

    df_lstm = load_data(temp_path)
    data = df_lstm.values
    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]
    test_data = data[train_size - 60:]
    print("train_data shape:", train_data.shape)
    print("train_data sample:\n", train_data[:5])

    X_train, y_train, scaler = preprocess_data(train_data)
    X_test, y_test, _ = preprocess_data(test_data)

    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot predictions
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.set_title("LSTM Stock Price Prediction")
    ax.legend()
    st.pyplot(fig)

    # --- RL TRADING SIMULATION ---
    st.subheader("🤖 Reinforcement Learning Trading Bot")

    env = StockTradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)

    obs = env.reset()
    total_rewards = 0
    profits = []

    for _ in range(len(df) - 1):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_rewards += reward
        profits.append(env.total_profit)
        if done:
            break

    st.write(f"📊 Total Profit from RL Agent: **₹{profits[-1]:.2f}**")

    fig2, ax2 = plt.subplots()
    ax2.plot(profits)
    ax2.set_title("Profit Over Time")
    st.pyplot(fig2)
