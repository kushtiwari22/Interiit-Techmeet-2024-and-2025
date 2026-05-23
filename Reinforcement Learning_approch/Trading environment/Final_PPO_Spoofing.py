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

# def hybridReward(
#     price_prev: float, 
#     price_now: float,
#     position_held: int,
#     position_new: int,
#     entry_price: float,
#     current_atr: float,
#     prev_smooth_return: float, # NEW: Needed for EMA calculation
#     transaction_cost_pct: float = 0.0005,
#     ema_alpha: float = 0.1     # NEW: Lower alpha = smoother trend (less noise)
# ) -> tuple[float, int, bool, float]:

#     # --- Constants ---
#     eps = 1e-9
#     lambda_entry = 0.02
    
#     # We scale the holding reward because vol_adjusted_return might be small
#     # If ATR is ~2.0 and Return is ~0.5, result is 0.25. 
#     # We want holding rewards to be small but noticeable.
#     hold_scale = 100.0 
    
#     profit_threshold = 0.002
#     loss_multiplier = 5.0
#     clip_max_penalty = 5.0

#     # --- Stop Loss Logic (From Old Function) ---
#     multiplier = 2.0
#     stop_loss_long = entry_price - (current_atr * multiplier) if entry_price > 0 else 0
#     stop_loss_short = entry_price + (current_atr * multiplier) if entry_price > 0 else 0

#     is_entry = (position_held == 0) and (position_new != 0)
#     is_hold  = (position_held != 0) and (position_new == position_held)
#     is_close = (position_held != 0) and (position_new == 0)
    
#     reward = 0.0

#     # --- 1. Stop Loss Check ---
#     stop_loss_hit = False
#     if position_held == 1 and price_now <= stop_loss_long:
#         stop_loss_hit = True
#         position_new = 0
#     elif position_held == -1 and price_now >= stop_loss_short:
#         stop_loss_hit = True
#         position_new = 0

#     # --- 2. Calculate Smoothed Return (From New Function) ---
#     # Calculate raw percent return for this step
#     if abs(price_prev) < eps:
#         raw_return = 0.0
#     else:
#         raw_return = (price_now - price_prev) / (price_prev + eps)
    
#     # Apply EMA Smoothing
#     # If we just entered, reset smooth return to current return to avoid lag
#     if is_entry:
#         current_smooth_return = raw_return
#     else:
#         current_smooth_return = (ema_alpha * raw_return) + ((1 - ema_alpha) * prev_smooth_return)

#     # --- 3. Entry Penalty ---
#     if is_entry:
#         reward -= lambda_entry

#     # --- 4. Holding Reward (Integrated Logic) ---
#     # We use the Smoothed, Volatility-Adjusted return for the holding reward.
#     # This filters out 15s noise and rewards stable trends.
#     if is_hold and not stop_loss_hit:
        
#         # Normalize by Volatility (ATR relative to price)
#         # We convert ATR to % terms so the ratio is dimensionless (Sharpe-like)
#         atr_pct = current_atr / (price_prev + eps)
        
#         if atr_pct < eps:
#             vol_adjusted_return = 0.0
#         else:
#             vol_adjusted_return = current_smooth_return / atr_pct

#         # Apply direction
#         reward += hold_scale * vol_adjusted_return * position_held

#     # --- 5. Closing Reward (From Old Function - More Robust) ---
#     if (is_close or stop_loss_hit) and entry_price is not None and entry_price > 0.0:
#         exit_price = price_now

#         if position_held == 1:
#             trade_pnl_pct = (exit_price - entry_price) / (entry_price + eps)
#         else:
#             trade_pnl_pct = (entry_price - exit_price) / (entry_price + eps)

#         net_pnl_pct = trade_pnl_pct - 2.0 * float(transaction_cost_pct)
#         scaled = 100.0 * net_pnl_pct

#         if stop_loss_hit:
#             # Heavy penalty for hitting stop loss
#             scaled *= 2.0
#             closing_bonus = -loss_multiplier * min(abs(scaled), clip_max_penalty)
#         elif scaled <= 0.0:
#             # Standard Loss
#             closing_bonus = -loss_multiplier * min(abs(scaled), clip_max_penalty)
#         else:
#             # Profit
#             if net_pnl_pct <= profit_threshold:
#                 closing_bonus = 0.0
#             else:
#                 closing_bonus = scaled

#         reward += closing_bonus

#     return float(reward), position_new, stop_loss_hit, current_smooth_return

def logReturnReward(
    price_prev: float, 
    price_now: float,
    position_held: int,
    position_new: int,
    entry_price: float,
    current_atr: float,
    transaction_cost_pct: float = 0.0005,
) -> float:

    eps = 1e-9
    lambda_entry = 0.02
    hold_scale = 200.0
    profit_threshold = 0.02
    loss_multiplier = 5.0
    clip_max_penalty = 5.0

    multiplier = 2.0
    stop_loss_long = entry_price - (current_atr * multiplier) if entry_price > 0 else 0
    stop_loss_short = entry_price + (current_atr * multiplier) if entry_price > 0 else 0

    is_entry = (position_held == 0) and (position_new != 0)
    is_hold  = (position_held != 0) and (position_new == position_held)
    is_close = (position_held != 0) and (position_new == 0)
    reward = 0.0

    stop_loss_hit = False
    if position_held == 1 and price_now <= stop_loss_long:
        stop_loss_hit = True
        position_new = 0
    elif position_held == -1 and price_now >= stop_loss_short:
        stop_loss_hit = True
        position_new = 0

    if is_entry:
        reward -= lambda_entry

    if is_hold and not stop_loss_hit:
        if abs(price_prev) < eps:
            holding_pnl = 0.0
        else:
            holding_pnl = ((price_now - price_prev) / (price_prev + eps)) * position_held

        reward += hold_scale * holding_pnl

    if (is_close or stop_loss_hit) and entry_price is not None and entry_price > 0.0:
        exit_price = price_now

        if position_held == 1:
            trade_pnl_pct = (exit_price - entry_price) / (entry_price + eps)
        else:
            trade_pnl_pct = (entry_price - exit_price) / (entry_price + eps)

        net_pnl_pct = trade_pnl_pct - 2.0 * float(transaction_cost_pct)
        scaled = 100.0 * net_pnl_pct

        if stop_loss_hit:
            scaled *= 2.0

        if scaled <= 0.0:
            closing_bonus = -loss_multiplier * min(abs(scaled), clip_max_penalty)
        else:
            if net_pnl_pct <= profit_threshold:
                closing_bonus = 0.0
            else:
                closing_bonus = scaled

        reward += closing_bonus

    return float(reward), position_new, stop_loss_hit

class StockDPSolver:
    def __init__(self):
        self.prices = []
        self.n = 0
        # DP table: dp[i][position]. Positions map: 0=FLAT, 1=LONG, 2=SHORT
        self.dp = np.zeros((1, 3))
        self.cost_bips = 0.001

    # MODIFIED: _solve now takes the start_signal and orchestrates the whole process
    def solve(self, prices: list, start_signal: int) -> list:
        """
        Solves for the optimal action sequence given a list of prices and a starting position.
        """
        self._fill_dp_table(prices)
        if self.n == 0:
            return []
        return self.get_action_sequence(start_signal)

    def _fill_dp_table(self, prices: list):
        """Fills the DP table using a bottom-up approach."""
        self.prices = prices
        self.n = len(prices)
        if self.n == 0:
            return

        self.dp = np.zeros((self.n + 1, 3))

        # Loop backwards from the last day to the first.
        # The special case for the last day is now REMOVED.
        for i in range(self.n - 1, -1, -1):
            transaction_cost = self.cost_bips * self.prices[i]

            # --- UNIFIED LOGIC FOR ALL DAYS ---
            # This logic now correctly handles the last day because self.dp[i + 1]
            # will point to the row of zeros (dp[n]) when i = n-1.

            # If LONG, you can either hold (future profit = dp[i+1][1]) or sell.
            profit_if_long = max(
                self.dp[i + 1][1],
                self.prices[i] + self.dp[i + 1][0] - transaction_cost
            )
            # If SHORT, you can either hold (future profit = dp[i+1][2]) or cover.
            profit_if_short = max(
                self.dp[i + 1][2],
                -self.prices[i] + self.dp[i + 1][0] - transaction_cost
            )
            # If FLAT, you can wait, buy, or sell short.
            profit_if_flat = max(
                self.dp[i + 1][0],
                -self.prices[i] + self.dp[i + 1][1] - transaction_cost,
                self.prices[i] + self.dp[i + 1][2] - transaction_cost
            )
            
            self.dp[i][0] = profit_if_flat
            self.dp[i][1] = profit_if_long
            self.dp[i][2] = profit_if_short

    def get_action_sequence(self, start_signal: int) -> list:
        """Backtracks through the solved DP table to find the optimal action sequence."""
        # This function is now "private" as it's called by the main solve() method
        action_sequence = []
        current_position = start_signal
        
        position_map = {0: 0, 1: 1, -1: 2} # Map RL signals to DP table indices
        
        for i in range(self.n):
            action_taken = 0  # Default action is HOLD
            current_pos_idx = position_map[current_position]
            transaction_cost = self.cost_bips * self.prices[i]
            max_future_profit = self.dp[i][current_pos_idx]

            # Future profits from the next state
            future_flat_profit = self.dp[i + 1][0]
            future_long_profit = self.dp[i + 1][1]
            future_short_profit = self.dp[i + 1][2]

            if current_position == 0: # FLAT
                profit_if_buy = -self.prices[i] + future_long_profit - transaction_cost
                profit_if_short_sell = self.prices[i] + future_short_profit - transaction_cost
                if np.isclose(max_future_profit, profit_if_buy):
                    action_taken = 1
                    current_position = 1
                elif np.isclose(max_future_profit, profit_if_short_sell):
                    action_taken = -1
                    current_position = -1

            elif current_position == 1: # LONG
                profit_if_sell = self.prices[i] + future_flat_profit - transaction_cost
                if np.isclose(max_future_profit, profit_if_sell):
                    action_taken = -1
                    current_position = 0
            
            elif current_position == -1: # SHORT
                profit_if_cover = -self.prices[i] + future_flat_profit - transaction_cost
                if np.isclose(max_future_profit, profit_if_cover):
                    action_taken = 1
                    current_position = 0
            
            action_sequence.append(action_taken)
            
        return action_sequence

class TradingEnv():
    def __init__(self, df_path, batch_size, initial_portfolio = INITIAL_PORTFOLIO, window_size = WINDOW_SIZE, split_ratio = SPLIT_RATIO):
        self.df = pd.read_csv(df_path)

        self.prev_smooth_return = 0.0
        self.initial_portfolio = initial_portfolio
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.portfolio = initial_portfolio
        self.max_drawdown = 0
        self.df.fillna(0, inplace= True)

        self.df['atr_20'] = ta.ATR(self.df['Price_high'].values, self.df['Price_low'].values, self.df['Price_close'].values, timeperiod=20)
        self.df['atr_20'].fillna(method='bfill', inplace=True)

        self.columns = [
            'Price_open','Price_high','Price_low','Price_close','PB1_T7','PB1_T8','PB1_T9','PB3_T7','PB3_T8','PB4_T8','PB4_T9','PB5_T8','PB5_T9','PB6_T7','PB7_T8','PB7_T9','PB8_T8','PB8_T9','PB12_T8','PB12_T9','PB13_T8','PB13_T9','PB14_T8','PB14_T9','PB15_T8','PB15_T9','PB16_T8','PB16_T9','PB17_T8','PB17_T9','PB18_T8','PB18_T9','VB5_T4','VB5_T5','VB5_T6','V2_T4','V2_T5','V2_T6','V2_T11','V8_T4_T5','V8_T4_T6','V8_T4_T8','V8_T4_T9','V8_T4_T10','V8_T4_T11','V8_T4_T12','V8_T5_T6','V8_T5_T8','V8_T5_T9','V8_T5_T10','V8_T5_T11','V8_T5_T12','V8_T6_T9','V8_T6_T10','V8_T6_T11','V8_T6_T12','V8_T7_T8','V8_T7_T11','V8_T7_T12','V8_T8_T9','V8_T8_T12','V8_T9_T10','V8_T9_T11','V8_T10_T11','V8_T10_T12','V8_T11_T12','BB2_T8','BB2_T9','BB3_T9','BB3_T10','BB3_T11','BB10_T9','PB8_T9-T7','PB8_T9-T8','PB15_T9-T7','PB16_T9-T7','PB17_T9-T7','VB5_T9-T7','VB5_T10-T9','V2_T9-T7','V8_T9-T7','V8_T10-T9','V8_T9-T8','BB1_T9-T7','BB1_T9-T8','BB3_T9-T7','BB3_T9-T8','BB4_T9-T7','BB4_T9-T8','BB5_T9-T7','BB6_T9-T7','BB6_T9-T8','BB13_T9-T7','BB14_T9-T7','BB15_T9-T7','Directional_power','Volatility_power','Price_open.1','Price_high.1','Price_low.1','Price_close.1','BB13_T8_diff','BB14_T8_diff','BB5_T8_diff','BB15_T8_diff','Trade_power','atr_20'
        ]
        
        self.df['signals'] = 0
        self.df['position'] = 0
        self.df['datetime'] = self.df['Time']
        self.df['final_signals'] = 0
        self.df['final_position'] = 0
        self.positions_map = {0:0, 1:1, -1:2}

        self.positions = [0, 1, -1]
        self.agent = Agent(input_dims= len(self.columns), n_actions= 3)
        self.dpsolver = StockDPSolver()

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
        returns_history = []

        # DP training
        price_slice_as_list = self.df['Price_open'][start:end].tolist()
        ideal_actions = self.dpsolver.solve(price_slice_as_list, flag)

        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd= False, flag= flag)

            prev_flag = flag
            expert_action = self.positions_map[ideal_actions[i-start]]
            prev_smooth_return = 0.0

            choice = ideal_actions[i-start] # Fool the agent
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

                step_reward, new_flag, stop_loss_hit = logReturnReward(
                    price_prev=price_prev,
                    price_now=price_now,
                    position_held=prev_flag,
                    position_new=flag,
                    entry_price=entry_price_for_reward,
                    current_atr=current_atr,
                    # prev_smooth_return= prev_smooth_return, # Pass current state
                    transaction_cost_pct=0.0005
                    # ema_alpha=0.2 # Tune this: 0.1 = very smooth, 0.5 = reactive
                )
                
                # Update the state variable for the next step
                # prev_smooth_return = new_smooth_val
                
                if stop_loss_hit:
                    flag = new_flag
                    self.df.loc[i, 'position'] = flag
                    if prev_flag == 1:
                        self.df.loc[i, "signals"] = SELL
                    else:
                        self.df.loc[i, "signals"] = BUY

            done = (i >= (start + optim_size - 1))
            
            self.agent.remember(state= observation.cpu(), action= expert_action, probs= probs, vals= value, reward= step_reward, done= done) # Use the ideal action
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
        returns_history = []
        end = min(start + wf_size, len(self.df))
        if start >= 1 and start < len(self.df): flag = self.df.loc[start-1, 'position']
        if start >= 2 and start < len(self.df): prev_flag = self.df.loc[start-2, 'position']
        prev_smooth_return = 0.0

        # DP training
        price_slice_as_list = self.df['Price_open'][start:end].tolist()
        ideal_actions = self.dpsolver.solve(price_slice_as_list, flag)

        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd= True, flag= flag)

            prev_flag = flag
            expert_action = self.positions_map[ideal_actions[i-start]]

            choice = ideal_actions[i-start]
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

                step_reward, new_flag, stop_loss_hit = logReturnReward(
                    price_prev=price_prev,
                    price_now=price_now,
                    position_held=prev_flag,
                    position_new=flag,
                    entry_price=entry_price_for_reward,
                    current_atr=current_atr,
                    # prev_smooth_return= prev_smooth_return, # Pass current state
                    transaction_cost_pct=0.0005
                    # ema_alpha=0.2 # Tune this: 0.1 = very smooth, 0.5 = reactive
                )
                
                if stop_loss_hit:
                    flag = new_flag
                    self.df.loc[i, 'position'] = flag
                    if prev_flag == 1:
                        self.df.loc[i, "signals"] = SELL
                    else:
                        self.df.loc[i, "signals"] = BUY

            done = (i >= (start + wf_size - 1))

            self.agent.remember(state= observation.cpu(), action= expert_action, probs= probs, vals= value, reward= step_reward, done= done)

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
        returns_history = []
        reward = 0
        prev_smooth_return = 0.0

        for i in range(start, end):
            observation = T.tensor(self.df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd= True, flag= flag)

            prev_flag = flag

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

                step_reward, new_flag, stop_loss_hit = logReturnReward(
                    price_prev=price_prev,
                    price_now=price_now,
                    position_held=prev_flag,
                    position_new=flag,
                    entry_price=entry_price_for_reward,
                    current_atr=current_atr,
                    # prev_smooth_return= prev_smooth_return, # Pass current state
                    transaction_cost_pct=0.0005
                    # ema_alpha=0.2 # Tune this: 0.1 = very smooth, 0.5 = reactive
                )
                
                if stop_loss_hit:
                    flag = new_flag
                    self.df.loc[i, 'final_position'] = flag
                    if prev_flag == 1:
                        self.df.loc[i, "final_signals"] = SELL
                    else:
                        self.df.loc[i, "final_signals"] = BUY

            reward += step_reward
            if i % 20 == 0:
                self.agent.learn()

            done = (i >= (start + batch_size - 1))
            self.agent.remember(state= observation.cpu(), action= action, probs= probs, vals= value, reward= step_reward, done= done)

        print("------------------------")
        print(f"Trade reward for batch: {reward}")
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
    for i in range(33, num_files):
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