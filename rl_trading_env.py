# rl_trading_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index()
        self.n_steps = len(df)
        self.initial_balance = 10000

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observations: [price, balance, shares held]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        price = self.df.loc[self.current_step, 'Close']
        return np.array([price, self.balance, self.shares_held], dtype=np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, 'Close']

        # Execute action
        if action == 1:  # Buy
            if self.balance >= price:
                self.shares_held += 1
                self.balance -= price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += price

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        total_assets = self.balance + self.shares_held * price
        reward = total_assets - self.initial_balance  # reward is profit
        self.total_profit = total_assets - self.initial_balance

        return self._next_observation(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Total Profit: {self.total_profit:.2f}")
