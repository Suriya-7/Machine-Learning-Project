# rl_train_agent.py

import pandas as pd
from stable_baselines3 import PPO
from rl_trading_env import StockTradingEnv
from stable_baselines3.common.env_checker import check_env

# Load data
df = pd.read_csv("data/AAPL.csv")
df = df[['Close']]

# Create environment
env = StockTradingEnv(df)
check_env(env, warn=True)  # optional: check for issues

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_stock_trader")
