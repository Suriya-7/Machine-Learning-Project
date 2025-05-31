# stock_data.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data for a given stock symbol
def fetch_stock_data(symbol="AAPL", start_date="2015-01-01", end_date="2024-12-31"):
    print(f"Fetching data for {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

# Save to CSV
def save_data(df, symbol):
    os.makedirs("data", exist_ok=True)
    path = f"data/{symbol}.csv"
    df.to_csv(path)
    print(f"Data saved to {path}")

# Plot the stock price
def plot_stock_data(df, symbol):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='Close', label="Close Price")
    plt.title(f"{symbol} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    symbol = "AAPL"
    df = fetch_stock_data(symbol)
    save_data(df, symbol)
    plot_stock_data(df, symbol)
