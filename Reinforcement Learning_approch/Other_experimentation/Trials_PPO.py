import numpy as np
import pandas as pd
from PPO import Agent
import torch as T
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
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


def reward_gen(portfolio_change, drawdown, flag, b = 0.8):
    # Basic reward function
    """ write some function here """
    if np.isnan(portfolio_change) or np.isnan(drawdown) or np.isnan(flag):
        return 0 # Or 0, or some other penalty
    return (portfolio_change - b*drawdown) + flag

def rewardFn(price_change : float, flag : int, slippage : int):
    return (price_change - slippage) * flag

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

        filtered_price=[]
        kf=KalmanFilter() 
        for i in range(len(self.df)):
            kf.update(self.df['close'].iloc[i])
            filtered_price.append(kf.predict())
        filtered_price = pd.Series(filtered_price)
        self.df['Filtered<Close'] = (self.df['close'] > filtered_price).astype('float64')

        self.df['signals'] = 0
        self.df['trade_type'] = 0

        self.columns = ['Filtered<Close' , 'BB4_T1', 'BB9_T1','BB9_T3', 'BB9_T7', 'BB10_T1', 'BB10_T3', 'PB1_T1', 'PB1_T2', 'PB1_T3','PB1_T4', 'PB1_T7', 'PB2_T1', 'PB2_T2', 'PB2_T3', 'PB2_T4', 'PB2_T5','PB2_T6', 'PB3_T1', 'PB3_T7', 'PB9_T1', 'PB10_T12', 'PB13_T8','PB13_T10', 'PB13_T12', 'VB1_T1', 'VB1_T2', 'VB1_T3', 'VB1_T4','VB1_T5', 'VB1_T7', 'VB2_T1', 'VB6_T1']
        for column in self.columns:
            self.df[column] = roll_norm(self.df[column])
        self.df['datetime'] = self.df['DateTime']
        # Final signals
        self.df['final_signals'] = 0
        self.df['final_trade_type'] = 0

        # Initialise PPO model
        self.positions = [0, 1, -1]
        self.agent = Agent(input_dims= len(self.columns), n_actions= 3)

    def checkdtypes(self):
        # Confirms datatypes of chosen columns
        for column in self.columns:
            print(column, ":", self.df[column].dtype)

    def optimisation(self, optim_size: int, start: int):
        reward = 0
        avg_reward = 0
        slippage = 0
        flag = 0
        end = min(start + optim_size, len(self.df))
        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd=False, flag=flag)

            prev_flag = flag  # store previous position
            if flag == HOLD:
                flag = self.positions[action]
                slippage += 0.001
                self.df.loc[i, "signals"] = flag
                self.df.loc[i, 'trade_type'] = HOLD if flag == HOLD else STRONG_BUY if flag == BUY else STRONG_SELL
            elif flag == BUY and self.positions[action] == SELL:
                flag = HOLD
                self.df.loc[i, "signals"] = SELL
                slippage += 0.001
                self.df.loc[i, 'trade_type'] = STRONG_SELL
            elif flag == SELL and self.positions[action] == BUY:
                flag = HOLD
                self.df.loc[i, "signals"] = BUY
                slippage += 0.001
                self.df.loc[i, 'trade_type'] = STRONG_BUY

            # --- new reward logic (using precomputed df columns) ---
            step_reward = 0
            if i > 0:
                transition = f"{prev_flag}_{flag}"
                col_name = f"reward_{transition}"
                if col_name in self.df.columns:
                    step_reward = self.df.loc[i, col_name]
                else:
                    step_reward = 0.0  # fallback for unexpected transitions
            # core difference - learning
            if i % 100 == 0:
                self.agent.learn()

            done = (i >= (start + optim_size - 1))
            self.agent.remember(state=observation.cpu(), action=action, probs=probs, vals=value, reward=step_reward, done=done)
            reward += step_reward
        self.agent.learn()
        avg_reward = reward / optim_size


    def walkforward(self, wf_size: int, start: int):
        reward = 0
        avg_reward = 0
        slippage = 0
        flag = 0
        end = min(start + wf_size, len(self.df))
        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd=True, flag=flag)

            prev_flag = flag  # store previous position
            if flag == HOLD:
                flag = self.positions[action]
                slippage += 0.001
                self.df.loc[i, "signals"] = flag
                self.df.loc[i, 'trade_type'] = HOLD if flag == HOLD else STRONG_BUY if flag == BUY else STRONG_SELL
            elif flag == BUY and self.positions[action] == SELL:
                flag = HOLD
                slippage += 0.001
                self.df.loc[i, "signals"] = SELL
                self.df.loc[i, 'trade_type'] = STRONG_SELL
            elif flag == SELL and self.positions[action] == BUY:
                flag = HOLD
                slippage += 0.001
                self.df.loc[i, "signals"] = BUY
                self.df.loc[i, 'trade_type'] = STRONG_BUY
            # self.df.loc[i, 'current_portfolio'], _ = calculate_portfolio_live(self.df[start:i+1])
            # --- new reward logic (using precomputed df columns) ---
            step_reward = 0
            if i > 0:
                transition = f"{prev_flag}_{flag}"
                col_name = f"reward_{transition}"
                if col_name in self.df.columns:
                    step_reward = self.df.loc[i, col_name]
                else:
                    step_reward = 0.0

            done = (i >= (start + wf_size - 1))
            reward += step_reward
            self.agent.remember(state=observation.cpu(), action=action, probs=probs, vals=value, reward=step_reward, done=done)
        # self.agent.learn()
        avg_reward = reward / wf_size


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
                self.optimisation(optim_size= size_, start= i + start)

    def trade(self, batch_size: int, start: int):
        # For one batch
        flag = 0
        slippage = 0
        end = min(start + batch_size, len(self.df))
        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd=True, flag=flag)
            # ---------------- Trade Logic ----------------
            if flag == HOLD:
                flag = self.positions[action]
                slippage += 0.001
                self.df.loc[i, "final_signals"] = flag
                self.df.loc[i, 'final_trade_type'] = (
                    HOLD if flag == HOLD else STRONG_BUY if flag == BUY else STRONG_SELL
                )
            elif flag == BUY and self.positions[action] == SELL:
                flag = HOLD
                slippage += 0.001
                self.df.loc[i, "final_signals"] = SELL
                self.df.loc[i, 'final_trade_type'] = STRONG_SELL
            elif flag == SELL and self.positions[action] == BUY:
                flag = HOLD
                slippage += 0.001
                self.df.loc[i, "final_signals"] = BUY
                self.df.loc[i, 'final_trade_type'] = STRONG_BUY
            step_reward = 0
            if i > 0:
                # Use precomputed reward from df instead of rewardFn
                if flag == 0:  # no position
                    if self.positions[action] == BUY:
                        step_reward = self.df.loc[i, 'reward_0_1']
                    elif self.positions[action] == SELL:
                        step_reward = self.df.loc[i, 'reward_0_-1']
                elif flag == BUY:
                    if self.positions[action] == 0:
                        step_reward = self.df.loc[i, 'reward_1_0']
                    elif self.positions[action] == SELL:
                        step_reward = self.df.loc[i, 'reward_1_-1']
                elif flag == SELL:
                    if self.positions[action] == 0:
                        step_reward = self.df.loc[i, 'reward_-1_0']
                    elif self.positions[action] == BUY:
                        step_reward = self.df.loc[i, 'reward_-1_1']
            # # Periodic learning (unchanged)
            # if i % 50 == 0:
            #     self.agent.learn()

            done = (i >= (start + batch_size - 1))
            self.agent.remember(
                state=observation.cpu(),
                action=action,
                probs=probs,
                vals=value,
                reward=step_reward,
                done=done
            )

    def run(self, day_number=None, output_folder='daily_signals'):
        """
        Run the trading strategy with walk-forward optimization
        
        Args:
            day_number: Day identifier for output filename
            output_folder: Folder to save signal files
        
        Returns:
            final_portfolio: Final portfolio value for the day
        """
        indices = range(0, len(self.df), self.batch_size)
        num_batches = len(indices)
        
        print("Trading on batch number 0:")
        self.trade(batch_size=self.batch_size, start=indices[0])
        
        for i in range(num_batches - 1):
            print(f"Training on batch number {i}:")
            self.train(batch_size=self.batch_size, window_size=self.window_size, 
                    split_ratio=self.split_ratio, start=indices[i])
            print(f"Trading on batch number {i+1}:")
            self.trade(batch_size=self.batch_size, start=indices[i+1])
        # Determine output filename
        if day_number is not None:
            output_filename = os.path.join(output_folder, f"signals_day{day_number}.csv")
        else:
            output_filename = "Final_logs_PPO.csv"
        
        # Save signals to day-specific file
        self.df.to_csv(output_filename, index=False)
        print(f"Signals saved to: {output_filename}")
        
        # Calculate and return final portfolio
        final_portfolio, max_dd = calculate_portfolio_live(
            self.df, 
            slipage=0.002, 
            initial_portfolio=self.initial_portfolio
        )
        return final_portfolio

def run_on_days(folder_path: str = 'new_eby_data', batch_size: int = 1450, output_folder: str = 'daily_signals'):
    """
    Run trading strategy across multiple days with proper tracking and saving
    
    Args:
        folder_path: Folder containing day CSV files
        batch_size: Batch size for processing
        output_folder: Folder to save daily signal files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    num_files = len(os.listdir(folder_path))
    
    # Track daily performance
    daily_performance = []
    
    print("="*80)
    print("STARTING MULTI-DAY TRADING SIMULATION")
    print("="*80)
    
    for i in range(0, num_files):
        csv_name = f"day{i}.csv"
        csv_path = os.path.join(folder_path, csv_name)
        
        print(f"\n{'='*80}")
        print(f"PROCESSING DAY {i}: {csv_name}")
        print(f"{'='*80}")
        print(f"Initial Portfolio: ${INITIAL_PORTFOLIO:.2f}")
        
        # Create trading environment with fresh portfolio
        env = TradingEnv(df_path=csv_path, batch_size=batch_size)
        
        # Load model from previous day (except first day)
        if i != 0:
            env.agent.load_models()
            print(f"Loaded model from Day {i-1}")
        
        # Run trading for the day and get final portfolio
        final_portfolio = env.run(day_number=i, output_folder=output_folder)
        
        # Save updated model
        env.agent.save_models()
        print(f"Saved model for Day {i}")
        
        # Calculate daily metrics
        daily_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
        
        # Store performance metrics
        daily_performance.append({
            'Day': i,
            'Date_File': csv_name,
            'Initial_Portfolio': INITIAL_PORTFOLIO,
            'Final_Portfolio': final_portfolio,
            'Daily_Return_Pct': daily_return,
            'Profit_Loss': final_portfolio - INITIAL_PORTFOLIO
        })
        
        # Print daily summary
        print(f"\n{'-'*80}")
        print(f"DAY {i} SUMMARY:")
        print(f"{'-'*80}")
        print(f"Initial Portfolio: ${INITIAL_PORTFOLIO:.2f}")
        print(f"Final Portfolio:   ${final_portfolio:.2f}")
        print(f"Daily P/L:         ${final_portfolio - INITIAL_PORTFOLIO:.2f}")
        print(f"Daily Return:      {daily_return:.2f}%")
        print(f"Signals saved to:  {output_folder}/signals_day{i}.csv")
        print(f"{'-'*80}\n")
    
    # Save overall performance summary
    summary_df = pd.DataFrame(daily_performance)
    summary_path = os.path.join(output_folder, 'daily_performance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL TRADING SUMMARY")
    print("="*80)
    print(f"Total Days Traded: {num_files}")
    print(f"Total P/L: ${summary_df['Profit_Loss'].sum():.2f}")
    print(f"Average Daily Return: {summary_df['Daily_Return_Pct'].mean():.2f}%")
    print(f"Best Day Return: {summary_df['Daily_Return_Pct'].max():.2f}% (Day {summary_df.loc[summary_df['Daily_Return_Pct'].idxmax(), 'Day']:.0f})")
    print(f"Worst Day Return: {summary_df['Daily_Return_Pct'].min():.2f}% (Day {summary_df.loc[summary_df['Daily_Return_Pct'].idxmin(), 'Day']:.0f})")
    print(f"Win Rate: {(summary_df['Daily_Return_Pct'] > 0).sum() / len(summary_df) * 100:.2f}%")
    print(f"\nPerformance summary saved to: {summary_path}")
    print("="*80)
    return summary_df
if __name__ == '__main__':
    # Create output directory for signals
    output_dir = 'daily_signals'
    
    # Run multi-day trading with tracking
    summary = run_on_days(
        folder_path='new_eby_data',
        batch_size=1450,
        output_folder=output_dir
    )
    print("\n✓ All days processed successfully!")
    print(f"✓ Individual signal files saved in: {output_dir}/")
    print(f"✓ Performance summary saved in: {output_dir}/daily_performance_summary.csv")