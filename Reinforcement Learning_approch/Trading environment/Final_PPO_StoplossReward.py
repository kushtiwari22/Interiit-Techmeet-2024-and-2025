import numpy as np
import pandas as pd
from PPO import Agent
import talib as ta
import torch as T
from untrade.client import Client
import uuid
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
from parameters import STRONG_BUY, STRONG_SELL, INITIAL_PORTFOLIO, WINDOW_SIZE, \
        SPLIT_RATIO, HOLD, SELL, BUY, TRAIN_BATCH_SIZE

class KalmanFilter:
    def __init__(self, Q=1e-7, R=1e-2):
        self.Q = Q
        self.R = R
        self.P = 1.0  
        self.X = 0.0  

    def predict(self):
        self.P = self.P + self.Q  
        return self.X

    def update(self, measurement):
        K = self.P / (self.P + self.R)  
        self.X = self.X + K * (measurement - self.X)  
        self.P = (1 - K) * self.P  

def roll_norm(series: pd.Series):
    if series.empty:
        return series

    min_val = series.iloc[0]
    max_val = series.iloc[0]
    
    normalized_values = []

    for i, val in enumerate(series):
        min_val = min(min_val, val)
        max_val = max(max_val, val)
        
        denominator = max_val - min_val
        if denominator == 0:
            normalized_val = 0.0
        else:
            normalized_val = (val - min_val) / denominator
            
        normalized_values.append(normalized_val)
    return pd.Series(normalized_values, index=series.index)

def calculate_portfolio_live(df: pd.DataFrame, slipage=0.002, initial_portfolio=1000.00):
    date_time = pd.to_datetime(df['datetime'])
    signal = df[df['signals'] != 0]
    signal = signal.reset_index(drop= True)
    returns = []
    trade_time = []

    for i in range(0, len(signal) - 1, 2):
        if signal.loc[i, 'signals'] == 1 and signal.loc[i, 'trade_type'] == STRONG_BUY:  
            entry = signal.loc[i, 'close']
            exit = signal.loc[i + 1, 'close']
            returns.append((exit - entry) / entry)
            trade_time.append(date_time.iloc[i+1] - date_time.iloc[i])
        elif signal.loc[i, 'trade_type'] == STRONG_BUY:  
            entry = signal.loc[i, 'close']
            exit = signal.loc[i + 1, 'close']
            returns.append((entry - exit) / entry)
            trade_time.append(date_time.iloc[i+1] - date_time.iloc[i])
    
    if len(signal) % 2 == 1:
        last_trade = signal.iloc[-1, :]
        entry_price = last_trade['close']
        current_price = df['close'].iloc[-1] 
        if last_trade['signals'] == 1: 
            returns.append((current_price - entry_price) / entry_price)
        else:  
            returns.append((entry_price - current_price) / entry_price)

    returns = np.array(returns)       
    returns_2 = 1 + returns - slipage  
    portfolio = [initial_portfolio]    
    for ret in returns_2:
        portfolio.append(portfolio[-1] * ret)
    max_drawdown = 0
    peak_portfolio = initial_portfolio
    for value in portfolio:
        if value > peak_portfolio:
            peak_portfolio = value
        else:
            drawdown = (peak_portfolio - value) / peak_portfolio
            max_drawdown = max(drawdown, max_drawdown)

    final_portfolio = portfolio[-1]
    return final_portfolio, max_drawdown

def drawdown(portfolio):
    max_drawdown = 0
    peak_portfolio = portfolio[0]
    for value in portfolio:
        if value > peak_portfolio:
            peak_portfolio = value
        else:
            drawdown = (peak_portfolio - value) / peak_portfolio
            max_drawdown = max(drawdown, max_drawdown)
    return max_drawdown

def logReturnReward(
    price_prev: float, 
    price_now: float,
    position_held: int,
    position_new: int,
    entry_price: float,
    current_atr: float,
    transaction_cost_pct: float = 0.0003,
) -> float:

    eps = 1e-9
    lambda_entry = 0.002
    hold_scale = 200.0
    profit_threshold = 0.002
    loss_multiplier = 5.0
    clip_max_penalty = 5.0

    is_entry = (position_held == 0) and (position_new != 0)
    is_hold  = (position_held != 0) and (position_new == position_held)
    is_close = (position_held != 0) and (position_new == 0)
    reward = 0.0

    if is_entry:
        reward -= lambda_entry

    if is_hold:
        if abs(price_prev) < eps:
            holding_pnl = 0.0
        else:
            holding_pnl = ((price_now - price_prev) / (price_prev + eps)) * position_held

        reward += hold_scale * holding_pnl

    if is_close and entry_price is not None and entry_price > 0.0:
        exit_price = price_now

        if position_held == 1:
            trade_pnl_pct = (exit_price - entry_price) / (entry_price + eps)
        else:
            trade_pnl_pct = (entry_price - exit_price) / (entry_price + eps)

        net_pnl_pct = trade_pnl_pct - 2.0 * float(transaction_cost_pct)
        scaled = 100.0 * net_pnl_pct

        if scaled <= 0.0:
            closing_bonus = -loss_multiplier * min(abs(scaled), clip_max_penalty)
        else:
            if net_pnl_pct <= profit_threshold:
                closing_bonus = 0.0
            else:
                closing_bonus = scaled

        reward += closing_bonus

    return float(reward)

class TradingEnv():
    def __init__(self, df_path, batch_size, label = "EBX", initial_portfolio = INITIAL_PORTFOLIO, window_size = WINDOW_SIZE, split_ratio = SPLIT_RATIO):
        self.df = pd.read_csv(df_path)

        self.initial_portfolio = initial_portfolio
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.portfolio = initial_portfolio
        self.max_drawdown = 0
        self.df.fillna(0, inplace= True)

        self.df['atr_20'] = ta.ATR(self.df['Price_high'].values, self.df['Price_low'].values, self.df['Price_close'].values, timeperiod=20)
        self.df['atr_20'].fillna(method='bfill', inplace=True)

        if label == "EBX":
            self.columns = [
                "Price_low.1",
                "Price_open.1",
                "V8_T4_T5",
                "V2_T11",
                "PB6_T7",
                "V8_T7_T12",
                "V2_T9-T7",
                "VB5_T6",
                "VB5_T9-T7",
                "BB10_T9",
                "Price_close.1",
                "V8_T7_T11",
                "V8_T6_T9",
                "Price_high.1",
                "BB3_T9-T7",
                "PB8_T9-T7",
                "V8_T4_T6",
                "V8_T8_T9",
                "V8_T8_T12",
                "PB1_T7",
                "PB5_T9",
                "V8_T11_T12",
                "BB3_T9-T8",
                "BB2_T9",
                "BB1_T9-T8",
                "BB4_T9-T7",
                "BB6_T9-T7",
                "V8_T9-T8",
                "PB17_T9-T7",
                "VB5_T10-T9",
                "V8_T9_T10",
                "V2_T5",
                "BB2_T8",
                "V8_T7_T8",
                "V2_T4",
                "V8_T10_T11",
                "BB4_T9-T8",
                "BB14_T8_diff",
                "V8_T9-T7",
                "V8_T10_T12",
                "BB1_T9-T7",
                "PB16_T9-T7",
                "BB13_T8_diff",
                "BB6_T9-T8",
                "VB5_T5",
                "BB15_T9-T7",
                "V8_T4_T8",
                "BB13_T9-T7",
                "V8_T4_T9",
                "V2_T6"]
        
        self.df['signals'] = 0
        self.df['position'] = 0
        self.df['datetime'] = self.df['Time']
        self.df['final_signals'] = 0
        self.df['final_position'] = 0

        self.positions = [0, 1, -1]
        self.agent = Agent(input_dims= len(self.columns), n_actions= 3)

        # Maintained parameters
        self.entry_price_optim = 0.0
        self.entry_price_wf = 0.0
        self.entry_price_trade = 0.0

    def checkdtypes(self):
        for column in self.columns:
            print(column, ":", self.df[column].dtype)

    def optimisation(self, optim_size : int, start : int):
        reward = 0.0
        avg_reward = 0.0
        flag = 0
        prev_flag = 0
        if start >= 1 and start < len(self.df): flag = self.df.loc[start-1, 'position']
        if start >= 2 and start < len(self.df): prev_flag = self.df.loc[start-2, 'position']
        end = min(start + optim_size, len(self.df))

        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd= False, flag= flag)
            current_price = self.df.loc[i, "Price_open"]

            prev_flag = flag

            # Take trades
            if flag != 0:
                # Agent cant act
                if flag == BUY:
                    if current_price >= (1.005 * self.entry_price_optim) or current_price <= (0.998 * self.entry_price_optim): # close position for profit or loss
                        flag = HOLD
                        self.df.loc[i, "signals"] = SELL
                elif flag == SELL:
                    if current_price <= (0.995 * self.entry_price_optim) or current_price >= (1.002 * self.entry_price_optim):
                        flag = HOLD
                        self.df.loc[i, "signals"] = BUY

            else:
                self.entry_price_optim = self.df.loc[i, "Price_open"]

                choice = self.positions[action]
                if flag == HOLD:
                    flag = choice
                    self.df.loc[i, "signals"] = flag
                elif flag == BUY and choice == SELL:
                    flag = HOLD
                    self.df.loc[i, "signals"] = SELL
                elif flag == BUY and choice == BUY:
                    flag = BUY
                    self.df.loc[i, "signals"] = HOLD
                elif flag == SELL and choice == BUY:
                    flag = HOLD
                    self.df.loc[i, "signals"] = BUY
                elif flag == SELL and choice == SELL:
                    flag = SELL
                    self.df.loc[i, "signals"] = HOLD

            self.df.loc[i, 'position'] = flag

            # Reward
            
            step_reward = 0
            entry_price_for_reward = 0.0
            current_atr = self.df.loc[i, 'atr_20']
            if i > start:
                price_now = self.df.loc[i, 'Price_open']
                price_prev = self.df.loc[i-1, 'Price_open']
            
                if prev_flag == 0 and flag != 0:
                    entry_price_for_reward = self.df.loc[i, 'Price_open']
                elif flag == 0:
                    entry_price_for_reward = 0.0

                step_reward = logReturnReward(
                    price_prev=price_prev,
                    price_now= price_now, position_held= prev_flag, 
                    position_new= flag, entry_price= entry_price_for_reward,
                    current_atr=current_atr
                )
            
            # Passed is the results of the taken action.
            done = (i >= (start + optim_size - 1))
            self.agent.remember(state= observation.cpu(), action= action, probs= probs, vals= value, reward= step_reward, done= done)
            reward += step_reward

            if i % 5 == 0:
                self.agent.learn()

        self.agent.learn()
        avg_reward = reward / optim_size

    def walkforward(self, wf_size : int, start : int):
        reward = 0.0
        avg_reward = 0.0
        flag = 0
        prev_flag = 0
        end = min(start + wf_size, len(self.df))
        if start >= 1 and start < len(self.df): flag = self.df.loc[start-1, 'position']
        if start >= 2 and start < len(self.df): prev_flag = self.df.loc[start-2, 'position']

        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd= True, flag= flag)
            current_price = self.df.loc[i, "Price_open"]

            prev_flag = flag

            # Take trades
            if flag != 0:
                # Agent cant act
                if flag == BUY:
                    if current_price >= (1.005 * self.entry_price_wf) or current_price <= (0.998 * self.entry_price_wf): # close position for profit or loss
                        flag = HOLD
                        self.df.loc[i, "signals"] = SELL
                elif flag == SELL:
                    if current_price <= (0.995 * self.entry_price_wf) or current_price >= (1.002 * self.entry_price_wf):
                        flag = HOLD
                        self.df.loc[i, "signals"] = BUY
            else:
                self.entry_price_wf = self.df.loc[i, "Price_open"]

                choice = self.positions[action]
                if flag == HOLD:
                    flag = choice
                    self.df.loc[i, "signals"] = flag
                elif flag == BUY and choice == SELL:
                    flag = HOLD
                    self.df.loc[i, "signals"] = SELL
                elif flag == BUY and choice == BUY:
                    flag = BUY
                    self.df.loc[i, "signals"] = HOLD
                elif flag == SELL and choice == BUY:
                    flag = HOLD
                    self.df.loc[i, "signals"] = BUY
                elif flag == SELL and choice == SELL:
                    flag = SELL
                    self.df.loc[i, "signals"] = HOLD

            self.df.loc[i, 'position'] = flag

            # Reward
            step_reward = 0
            entry_price_for_reward = 0
            current_atr = self.df.loc[i, 'atr_20']

            if i > start:
                price_now = self.df.loc[i, 'Price_open']
                price_prev = self.df.loc[i-1, 'Price_open']
                
                if prev_flag == 0 and flag != 0:
                    entry_price_for_reward = self.df.loc[i, 'Price_open']
                elif flag == 0:
                    entry_price_for_reward = 0.0

                step_reward = logReturnReward(
                    price_prev=price_prev,
                    price_now= price_now, position_held= prev_flag, 
                    position_new= flag, entry_price= entry_price_for_reward,
                    current_atr=current_atr
                )

            done = (i >= (start + wf_size - 1))
            reward += step_reward
            self.agent.remember(state= observation.cpu(), action= action, probs= probs, vals= value, reward= step_reward, done= done)

        self.agent.learn()
        avg_reward = reward / wf_size

    def train(self, batch_size : int, window_size : int, split_ratio : float, start : int):
        wf_size = int((1 - split_ratio) * window_size)
        optim_size = window_size - wf_size
        indices = range(0, batch_size, wf_size)

        for i in indices:
            if i + window_size <= batch_size:
                self.optimisation(optim_size= optim_size, start= i + start)
                self.walkforward(wf_size= wf_size, start= i + optim_size + start)
            else:
                size_ = batch_size - i
                self.optimisation(optim_size= size_, start= i + start)

    def trade(self, batch_size : int, start : int):
        flag = 0
        prev_flag = 0
        if start >= 1 and start < len(self.df): flag = self.df.loc[start-1, 'final_position']
        if start >= 2 and start < len(self.df): prev_flag = self.df.loc[start-2, 'final_position']
        end = min(start + batch_size, len(self.df))
        reward = 0
        net_profits = 0.0

        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd= True, flag= flag)
            current_price = self.df.loc[i, "Price_open"]

            prev_flag = flag

            # Take trades
            if flag != 0:
                # Agent cant act
                if flag == BUY:
                    if current_price >= (1.005 * self.entry_price_trade) or current_price <= (0.998 * self.entry_price_trade): # close position for profit or loss
                        flag = HOLD
                        self.df.loc[i, "final_signals"] = SELL

                        pnl = current_price - self.entry_price_trade
                        net_profits += pnl
                elif flag == SELL:
                    if current_price <= (0.995 * self.entry_price_trade) or current_price >= (1.002 * self.entry_price_trade):
                        flag = HOLD
                        self.df.loc[i, "final_signals"] = BUY

                        pnl = - (current_price - self.entry_price_trade)
                        net_profits += pnl
            else:
                self.entry_price_trade = self.df.loc[i, "Price_open"]

                choice = self.positions[action]
                if flag == HOLD:
                    flag = choice
                    self.df.loc[i, "final_signals"] = flag
                elif flag == BUY and choice == SELL:
                    flag = HOLD
                    self.df.loc[i, "final_signals"] = SELL
                elif flag == BUY and choice == BUY:
                    flag = BUY
                    self.df.loc[i, "final_signals"] = HOLD
                elif flag == SELL and choice == BUY:
                    flag = HOLD
                    self.df.loc[i, "final_signals"] = BUY
                elif flag == SELL and choice == SELL:
                    flag = SELL
                    self.df.loc[i, "final_signals"] = HOLD

            self.df.loc[i, 'final_position'] = flag

            step_reward = 0
            entry_price_for_reward = 0
            current_atr = self.df.loc[i, 'atr_20']

            if i > start:
                price_now = self.df.loc[i, 'Price_open']
                price_prev = self.df.loc[i-1, 'Price_open']
                
                if prev_flag == 0 and flag != 0: 
                    entry_price_for_reward = self.df.loc[i, 'Price_open']
                elif flag == 0: 
                    entry_price_for_reward = 0.0

                step_reward = logReturnReward(
                    price_prev=price_prev,
                    price_now= price_now, position_held= prev_flag, 
                    position_new= flag, entry_price= entry_price_for_reward,
                    current_atr=current_atr
                )

            reward += step_reward
            if i % 20 == 0:
                self.agent.learn()

            done = (i >= (start + batch_size - 1))
            self.agent.remember(state= observation.cpu(), action= action, probs= probs, vals= value, reward= step_reward, done= done)

        print("------------------------")
        print(f"Trade reward for batch: {reward}")
        print(f"Profit earned this batch: {net_profits}")
        print("------------------------")

    def run(self):
        start_index = 20
        indices = range(start_index, len(self.df), self.batch_size)
        if not indices:
            return
            
        num_batches = len(indices)
        print("Trading on batch number 0:")
        self.trade(batch_size= self.batch_size, start= indices[0])

        for i in range(num_batches - 1):
            print(f"Training on batch number {i}:")
            self.train(batch_size= self.batch_size, window_size= self.window_size, split_ratio= self.split_ratio,
                    start= indices[i])
            print(f"Trading on batch number {i+1}:")
            self.trade(batch_size= self.batch_size, start= indices[i+1])
        
        self.train(batch_size= self.batch_size, window_size= self.window_size, split_ratio= self.split_ratio, start= indices[-1])

        print(self.df['final_signals'].value_counts())

    def save_logs(self, i : int):
        print(f"Logging Day {i} results....")
        self.df.to_csv(f"/home/ashisv/InterIIT14_PS1/EBX_logs/Final_logs_PPO_{i}.csv")

def run_on_days(folder_path : str= "/home/ashisv/InterIIT14_PS1/EBX_processed_final", batch_size : int= TRAIN_BATCH_SIZE):
    num_files = len(os.listdir(folder_path))
    for i in range(0, num_files):
        csv_name = f"processed_day{i}.csv"
        csv_path = os.path.join(folder_path, csv_name)

        if not os.path.exists(csv_path):
            continue

        env = TradingEnv(df_path= csv_path, 
                        batch_size= TRAIN_BATCH_SIZE)
        if i != 0:
            env.agent.load_models()
        env.run()
        env.save_logs(i)
        env.agent.save_models()

if __name__ == '__main__':
    run_on_days()