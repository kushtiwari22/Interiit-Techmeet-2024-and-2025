import numpy as np
import pandas as pd
from TD3 import Agent
import talib as ta
import torch as T
from untrade.client import Client
import uuid
import os
from parameters import ROLLING_WINDOW, STRONG_BUY, STRONG_SELL, INITIAL_PORTFOLIO, WINDOW_SIZE, \
        SPLIT_RATIO, THRESHOLD, OPTIM_RATIO, WINDOW_SIZE_DATA, TRAIN_OPTIM, HOLD, SELL, BUY

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


def perform_backtest_large_csv(csv_file_path:str):
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0
    if total_size <= chunk_size:
        total_chunks = 1
        result = client.backtest(
            file_path=csv_file_path,
            leverage=1,
            jupyter_id="team57_zelta_hpps",
            # result_type="Q",
        )
        for value in result:
            print(value)

        return result

    with open(csv_file_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            chunk_file_path = f"/tmp/{file_id}_chunk_{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            # Large CSV Backtest
            result = client.backtest(
                file_path=chunk_file_path,
                leverage=1,
                jupyter_id="team57_zelta_hpps",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                # result_type="Q",
            )
            for value in result:
                print(value)
            os.remove(chunk_file_path)
            chunk_number += 1
    return result

def normalize_data(df : pd.DataFrame):
    # normalise open and close prices
    data = df.copy()
    price = data['close']
    data['rolling_mean'] = data['close'].rolling(ROLLING_WINDOW).mean()
    data['rolling_std'] = data['close'].rolling(ROLLING_WINDOW).std()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['normalized_close'] = (data['close'] - data['rolling_mean'])/data['rolling_std']
    
    return data

def cusum(data : pd.DataFrame, threshold):
    # cusum indicator function
    data = normalize_data(data)
    data['Sh'] = 0.0
    data['Sl'] = 0.0
    for i in range(1, len(data)) :
        data.loc[i, 'Sh'] = max(0, data.loc[i-1, 'Sh'] + data.loc[i, 'normalized_close'] - threshold)
        data.loc[i, 'Sl'] = max(0, data.loc[i-1, 'Sl'] - data.loc[i, 'normalized_close'] - threshold)
    
    return data

def create_hurst_exponent(data):
    # hurst exponent indicator function
    N = len(data)
    mean_data = np.mean(data)
    cumulative_dev = np.cumsum(data - mean_data)
    R = np.max(cumulative_dev) - np.min(cumulative_dev)  
    S = np.std(data)                                     

    if S == 0:  
        return np.nan

    H = np.log(R/S) / np.log(N)
    return H

def create_hurst(data):
    # hurst indicator function
    data2 = data.copy()
    window_size = ROLLING_WINDOW
    length = (int)(len(data2) / window_size)
    window_data = []
    for i in range(0, len(data2), window_size):
        window = data2.loc[i:i+window_size, 'close'].tolist()
        window_data.append(window)
        i+=window_size
    hurst = [
        create_hurst_exponent(window_data[i])
        for i in range(len(window_data))
    ]
    j=0
    data2['hurst_value'] = 0
    data2['weekly_hurst'] = 0
    for i in range(0, len(data2), window_size):
        if(j < len(hurst)):
            if(hurst[j] > 0.5):
                data2.loc[i:i+window_size-1, 'weekly_hurst'] = 1
                data2['hurst_value'] = data2['hurst_value'].astype(float)
                data2.loc[i:i+window_size-1, 'hurst_value'] = hurst[j]
            else:
                data2.loc[i:i+window_size-1, 'weekly_hurst'] = -1
                data2['hurst_value'] = data2['hurst_value'].astype(float)
                data2.loc[i:i+window_size-1, 'hurst_value'] = hurst[j]
            j+=1

    return data2

def calculate_portfolio_live(df: pd.DataFrame, slipage=0.002, initial_portfolio=1000.00):
    # simulate live trading
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
    # define drawdown
    max_drawdown = 0
    peak_portfolio = portfolio[0]
    for value in portfolio:
        if value > peak_portfolio:
            peak_portfolio = value
        else:
            drawdown = (peak_portfolio - value) / peak_portfolio
            max_drawdown = max(drawdown, max_drawdown)
    return max_drawdown

def perform_backtest_large_csv(csv_file_path:str):
    # Backtesting with untrade engine
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0
    if total_size <= chunk_size:
        total_chunks = 1
        # Normal Backtest
        result = client.backtest(
            file_path=csv_file_path,
            leverage=1,
            jupyter_id="team57_zelta_hpps",
            # result_type="Q",
        )
        for value in result:
            print(value)

        return result

    with open(csv_file_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            chunk_file_path = f"/tmp/{file_id}_chunk_{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            # Large CSV Backtest
            result = client.backtest(
                file_path=chunk_file_path,
                leverage=1,
                jupyter_id="team57_zelta_hpps",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                # result_type="Q",
            )

            for value in result:
                print(value)

            os.remove(chunk_file_path)

            chunk_number += 1

    return result

def reward_gen(portfolio_change, drawdown, hurst, action, b = 0.8):
    # Basic reward function
    """ write some function here """
    if np.isnan(portfolio_change) or np.isnan(drawdown) or np.isnan(hurst) or np.isnan(action):
        return 0 # Or 0, or some other penalty
    return (portfolio_change - b*drawdown) + hurst * action

# Dummy gym environment
class DummyActionSpace:
    def __init__(self, n_actions, min_val, max_val):
        self.n = n_actions
        # The agent expects these to be NumPy arrays
        self.low = np.array([min_val] * n_actions, dtype=np.float32)
        self.high = np.array([max_val] * n_actions, dtype=np.float32)

class DummyEnv: pass

class TradingEnv():
    # The entire RL + WFO trading environment
    def __init__(self, df_path, batch_size, initial_portfolio = INITIAL_PORTFOLIO, window_size = WINDOW_SIZE, split_ratio = SPLIT_RATIO):
        # Constructor
        self.df = pd.read_csv(df_path)

        # Initialise values
        self.initial_portfolio = initial_portfolio
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.portfolio = initial_portfolio
        self.max_drawdown = 0
        self.n_actions_for_td3 = 1
        self.min_action_value = -1.0
        self.max_action_value = 1.0

        # Data Preprocessing and Implementation of technical indicators
        self.df = cusum(self.df, 0.5)
        self.df['cusum'] = self.df['Sh'] - self.df['Sl']
        self.df['RSI_Smoothed'] = ta.RSI(self.df['close'], timeperiod= 14).rolling(window= 5).mean()
        roc1 = ta.ROC(self.df['close'], timeperiod=10)
        roc2 = ta.ROC(self.df['close'], timeperiod=15)
        roc3 = ta.ROC(self.df['close'], timeperiod=20)
        roc4 = ta.ROC(self.df['close'], timeperiod=30)
        self.df['KST'] = (ta.SMA(roc1, timeperiod= 10) + 2 * ta.SMA(roc2, timeperiod= 15) + 
                          3 * ta.SMA(roc3, timeperiod= 20) + 4 * ta.SMA(roc4, timeperiod= 30))
        self.df['KST_Signal'] = ta.SMA(self.df['KST'], timeperiod= 9)
        ema13 = ta.EMA(self.df['close'], timeperiod= 13)
        self.df['Bull_Power'] = self.df['high'] - ema13
        self.df['Bear_Power'] = self.df['low'] - ema13
        self.df['Plus_DI'] = ta.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        self.df['Minus_DI'] = ta.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        self.df['Plus_DI_s'] = ta.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=63)
        self.df['Minus_DI_s'] = ta.MINUS_DI(self.df['high'],self.df['low'], self.df['close'], timeperiod=63)
        ema12 = ta.EMA(self.df['close'], timeperiod= 12)
        ema26 = ta.EMA(self.df['close'], timeperiod= 26)
        self.df['MACD'] = ema12 - ema26
        self.df['Aroon_Osc'] = ta.AROONOSC(self.df['high'], self.df['low'], timeperiod= 25)
        self.df['Rolling_Threshold'] = self.df['close'].pct_change(5, fill_method= None).rolling(10).quantile(0.9)
        HA_close = (self.df['open'] + self.df['close'] + self.df['high'] + self.df['low']) / 4
        HA_open = self.df['open']
        HA_open = (HA_open.shift(1) + HA_close.shift(1))/ 2
        self.df['HA_Close>Open'] = (HA_close > HA_open).astype('float64')
        filtered_price=[]
        kf=KalmanFilter() 
        for i in range(len(self.df)):
            kf.update(self.df['close'].iloc[i])
            filtered_price.append(kf.predict())
        filtered_price = pd.Series(filtered_price)
        self.df['Filtered<Close'] = (self.df['close'] > filtered_price).astype('float64')
        self.df = create_hurst(self.df)
        self.df['hurst_value'] = self.df['hurst_value'].fillna(0.5)
        self.df['signals'] = 0
        self.df['trade_type'] = 0

        # Null handling
        self.df.fillna(0, inplace= True)

        # Choose columns as input to model
        self.columns = ['Filtered<Close', 'HA_Close>Open', 'Rolling_Threshold', 'Aroon_Osc', 'MACD', 'Plus_DI', 'Plus_DI_s', 'Minus_DI', 'Minus_DI_s',
                        'Bull_Power', 'Bear_Power', 'KST_Signal', 'KST', 'RSI_Smoothed', 'hurst_value', 'cusum']
        for column in self.columns:
            self.df[column] = self.df[column] / self.df[column].abs().max()
        
        # Final signals
        self.df['final_signals'] = 0
        self.df['final_trade_type'] = 0

        # Initialise TD3 model
        dummy_env = DummyEnv()
        dummy_env.action_space = DummyActionSpace(n_actions= self.n_actions_for_td3, min_val= self.min_action_value, max_val= self.max_action_value)
        self.agent = Agent(alpha= 0.001, input_dims= len(self.columns), tau= 0.005, env= dummy_env)

    def checkdtypes(self):
        # Confirms datatypes of chosen columns
        for column in self.columns:
            print(column, ":", self.df[column].dtype)

    def optimisation(self, optim_size : int, start : int): # optim size = window size * split ratio
        # Initialise values, and handle date time preprocessing
        reward = 0
        avg_reward = 0
        end = min(start + optim_size, len(self.df))

        for i in range(start, end - 1): # Note the '-1'
            current_state = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action = self.agent.choose_action(current_state)
            new_state = T.tensor(self.df[self.columns].iloc[i+1].values, dtype=T.float32).to(self.agent.actor.device)
            done = (i == start + optim_size - 2)

            self.df.loc[i, 'signals'] = action # Position sized
            self.df.loc[i, 'current_portfolio'], _ = calculate_portfolio_live(self.df[0:i+1])
            drawdown_value = drawdown(np.array(self.df.loc[start:i+1, 'current_portfolio']))
            portfolio_change = (self.df.loc[i, 'current_portfolio'] - self.df.loc[start, 'current_portfolio']) / (self.df.loc[start, 'current_portfolio'] + 1e-9)
            step_reward = reward_gen(portfolio_change, drawdown_value, self.df.loc[i, 'hurst_value'], action)
            reward += step_reward

            self.agent.remember(state= current_state, action= action, reward= step_reward, state_= new_state, done= done)
            
            if i % 10 == 0:
                self.agent.learn() # Store the observation in cyclic memory

        self.agent.learn()
        avg_reward = reward / optim_size

        print(f"Total Reward earned from {start} to {start + optim_size-1} in Optimization Phase = {reward}")
        print(f"Average Reward earned = {avg_reward}")

    def walkforward(self, wf_size : int, start : int):
        # Initialise values, and handle date time preprocessing
        reward = 0
        avg_reward = 0
        end = min(start + wf_size, len(self.df))

        for i in range(start, end - 1):
            current_state = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action = self.agent.choose_action(current_state)
            new_state = T.tensor(self.df[self.columns].iloc[i+1].values, dtype=T.float32).to(self.agent.actor.device)
            done = (i == start + wf_size - 2)

            self.df.loc[i, 'signals'] = action # Position sized
            self.df.loc[i, 'current_portfolio'], _ = calculate_portfolio_live(self.df[0:i+1])
            drawdown_value = drawdown(np.array(self.df.loc[start:i+1, 'current_portfolio']))
            portfolio_change = (self.df.loc[i, 'current_portfolio'] - self.df.loc[start, 'current_portfolio']) / (self.df.loc[start, 'current_portfolio'] + 1e-9)
            step_reward = reward_gen(portfolio_change, drawdown_value, self.df.loc[i, 'hurst_value'], action)
            reward += step_reward

            self.agent.remember(state= current_state, action= action, reward= step_reward, state_= new_state, done= done)

        self.agent.learn()
        avg_reward = reward / wf_size

        print(f"Total Reward earned from {start} to {start + wf_size-1} in Walk Forward Phase = {reward}")
        print(f"Average Reward earned = {avg_reward}")

    def train(self, batch_size : int, window_size : int, split_ratio : float, start : int):
        # For one batch
        wf_size = int((1 - split_ratio) * window_size)
        optim_size = window_size - wf_size
        indices = range(0, batch_size, wf_size)

        for i in indices:
            if i + window_size <= batch_size:
                self.optimisation(optim_size= optim_size, start= i + start)
                self.walkforward(wf_size= wf_size, start= i + optim_size + start) # for the main dataframe
            else:
                size_ = batch_size - i
                # Last phase is only optimisation, to clean up
                self.optimisation(optim_size= size_, start= i + start)

    def trade(self, batch_size : int, start : int):
        # For one batch
        flag = 0
        end = min(start + batch_size, len(self.df))
        for i in range(start, end):
            current_state = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action = self.agent.choose_action(current_state)
            
            self.df.loc[i, 'final_signals'] = action # Position sized

            """
            Do if intended to train on recent values during trade:
            drawdown_value = drawdown(np.array(self.df.loc[start:i+1, 'current_portfolio']))
            portfolio_change = (self.df.loc[i, 'current_portfolio'] - self.df.loc[start, 'current_portfolio']) / (self.df.loc[start, 'current_portfolio'] + 1e-9)
            step_reward = reward_gen(portfolio_change, drawdown_value, self.df.loc[i, 'hurst_value'], flag)
            done = (i >= (start + batch_size - 1))
            self.agent.remember(state= observation.cpu(), action= action, probs= probs, vals= value, reward= step_reward, done= done)
            if i % 20:
                self.agent.learn()
            """

    def run(self):
        indices = range(0, len(self.df), self.batch_size)
        num_batches = len(indices)
        print("Trading on batch number 0:")
        self.trade(batch_size= self.batch_size, start= indices[0])

        for i in range(num_batches - 1):
            print(f"Training on batch number {i}:")
            self.train(batch_size= self.batch_size, window_size= self.window_size, split_ratio= self.split_ratio,
                    start= indices[i])
            print(f"Trading on batch number {i+1}:")
            self.trade(batch_size= self.batch_size, start= indices[i+1])

if __name__ == '__main__':
    env = TradingEnv(df_path= '/run/media/ashisv/New Volume/Study/Inter IIT 14 PS1/BTC_2019_2023_6h.csv', 
                     batch_size= 500)
    env.run()