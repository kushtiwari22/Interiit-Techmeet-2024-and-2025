import os
import warnings
import numpy as np
import pandas as pd
import torch as T
from data_preprocess import *
from PPO_7 import Agent
from parameters2 import (
    ROLLING_WINDOW, STRONG_BUY, STRONG_SELL, INITIAL_PORTFOLIO, WINDOW_SIZE,
    SPLIT_RATIO, THRESHOLD, OPTIM_RATIO, WINDOW_SIZE_DATA, TRAIN_OPTIM,
    HOLD, SELL, BUY, TRANSACTION_COST, HOLDING_BONUS, REWARD_CLIP_MIN,
    REWARD_CLIP_MAX, PNL_AMPLIFICATION, DIRECTIONAL_BONUS, DRAWDOWN_PENALTY,
    SHARPE_BONUS
)
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

class PortfolioTracker:
    def __init__(self, initial_portfolio=1000.0, window_size=252,
                 p_ppy=252, rf=0.0, transaction_cost=0.0004):
        self.portfolio_values = [initial_portfolio]
        self.window_size = window_size
        self.p_ppy = p_ppy
        self.rf = rf
        self.transaction_cost = transaction_cost
        self.epsilon = 1e-8
        self.initial_portfolio = initial_portfolio

    def update_portfolio(self, current_value):
        self.portfolio_values.append(current_value)
        if len(self.portfolio_values) > self.window_size * 2:
            self.portfolio_values.pop(0)

    def calculate_step_reward(
        self,
        position,
        prev_position,
        price_change,
        portfolio_value,
        current_price,
        entry_price,
        stop_loss_hit=False,
        take_profit_hit=False,
        typical_holding_minutes=10,
    ):
        reward = 0.0
        realized_pnl = 0.0
        
        if prev_position != 0 and position != prev_position and entry_price > 0:
            if prev_position == 1:
                realized_pnl = (current_price - entry_price) / entry_price
            elif prev_position == -1:
                realized_pnl = (entry_price - current_price) / entry_price
            reward += realized_pnl
        
        unrealized_pnl = 0.0
        if position == 1 and entry_price > 0:
            unrealized_pnl = (current_price - entry_price) / entry_price
        elif position == -1 and entry_price > 0:
            unrealized_pnl = (entry_price - current_price) / entry_price
        reward += 0.3 * unrealized_pnl
        
        # Apply penalties/rewards for stop loss and take profit
        if stop_loss_hit:
            reward -= 0.05  # Small penalty for hitting stop loss
        if take_profit_hit:
            reward += 0.03  # Small bonus for hitting take profit
        
        if typical_holding_minutes <= 0:
            typical_holding_minutes = 10
        per_min_cost = self.transaction_cost / typical_holding_minutes
        
        if position != 0 or prev_position != 0:
            reward -= per_min_cost
        
        if position == 0 and prev_position == 0:
            reward -= 0.00005
        
        reward = np.clip(reward, -0.5, 0.5)
        return reward

    @staticmethod
    def calculate_portfolio_live(df: pd.DataFrame,
                                 slipage=0.0004,
                                 initial_portfolio=1000.0):
        portfolio = initial_portfolio
        position = 0
        entry_price = 0
        portfolio_history = [initial_portfolio]
        trades = []
        
        # Absolute stop loss and take profit values
        ABSOLUTE_STOP_LOSS = 0.001  # 0.1% absolute stop loss
        ABSOLUTE_TAKE_PROFIT = 0.003  # 0.3% absolute take profit
        
        for i in range(len(df)):
            current_signal = df.loc[i, "final_signals"]
            current_price = df.loc[i, "Price_close"]
            
            # Check if we need to exit due to absolute stop loss or take profit
            early_exit = False
            exit_reason = ""
            if position != 0 and entry_price > 0:
                if position == 1:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= -ABSOLUTE_STOP_LOSS:
                        early_exit = True
                        exit_reason = "STOP_LOSS"
                    elif pnl_pct >= ABSOLUTE_TAKE_PROFIT:
                        early_exit = True
                        exit_reason = "TAKE_PROFIT"
                elif position == -1:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    if pnl_pct <= -ABSOLUTE_STOP_LOSS:
                        early_exit = True
                        exit_reason = "STOP_LOSS"
                    elif pnl_pct >= ABSOLUTE_TAKE_PROFIT:
                        early_exit = True
                        exit_reason = "TAKE_PROFIT"
            
            # Exit position if signal changes OR early exit condition met
            if position != 0 and (current_signal != position or early_exit):
                if position == 1:
                    pnl = (current_price - entry_price) / entry_price
                    portfolio = portfolio * (1 + pnl - slipage)
                    trades.append(("LONG", entry_price, current_price, pnl, exit_reason if early_exit else "SIGNAL"))
                elif position == -1:
                    pnl = (entry_price - current_price) / entry_price
                    portfolio = portfolio * (1 + pnl - slipage)
                    trades.append(("SHORT", entry_price, current_price, pnl, exit_reason if early_exit else "SIGNAL"))
                position = 0
                entry_price = 0
            
            # Enter new position if signal says so and we're not in a position
            if current_signal != 0 and position == 0:
                position = current_signal
                entry_price = current_price
            
            portfolio_history.append(portfolio)
        
        # Close any remaining position at the end
        if position != 0:
            current_price = df["Price_close"].iloc[-1]
            if position == 1:
                pnl = (current_price - entry_price) / entry_price
                portfolio = portfolio * (1 + pnl - slipage)
                trades.append(("LONG", entry_price, current_price, pnl, "END"))
            elif position == -1:
                pnl = (entry_price - current_price) / entry_price
                portfolio = portfolio * (1 + pnl - slipage)
                trades.append(("SHORT", entry_price, current_price, pnl, "END"))
        
        portfolio_history.append(portfolio)
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = portfolio_history[0]
        for value in portfolio_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        
        print(f"Portfolio Calculation: {len(trades)} trades, Final: ${portfolio:.2f}")
        if trades:
            winning_trades = [t for t in trades if t[3] > 0]
            print(f"Win Rate: {len(winning_trades) / len(trades) * 100:.1f}%")
            stop_loss_exits = len([t for t in trades if t[4] == 'STOP_LOSS'])
            take_profit_exits = len([t for t in trades if t[4] == 'TAKE_PROFIT'])
            print(f"Stop Loss Exits: {stop_loss_exits}")
            print(f"Take Profit Exits: {take_profit_exits}")
        
        return portfolio, max_drawdown, trades

    @staticmethod
    def drawdown(portfolio_history):
        max_drawdown = 0
        peak = portfolio_history[0]
        for value in portfolio_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        return max_drawdown


class TradingEnv:
    def __init__(self, df_path, batch_size,
                 initial_portfolio=INITIAL_PORTFOLIO,
                 window_size=WINDOW_SIZE,
                 split_ratio=SPLIT_RATIO):
        self.df = pd.read_csv(df_path)
        # self.df = main(self.df)
        # remove_cols = ["Time", "Price_open", "Price_low", "Price_high"]
        self._add_atr()
        self.columns = ['Price_close', 'PB1_T6', 'PB1_T9', 'PB1_T6_diff', 'PB1_T9_diff', 'PB1_T10_T7', 'PB2_T6', 'PB2_T9', 'PB2_T6_diff', 'PB3_T6', 'PB3_T9', 'PB3_T6_diff', 'PB3_T9_diff', 'PB3_T10_T7', 'PB4_T6', 'PB4_T9', 'PB4_T6_diff', 'PB4_T9_diff', 'PB4_T10_T7', 'PB5_T6', 'PB5_T9', 'PB5_T6_diff', 'PB5_T9_diff', 'PB5_T10_T7', 'PB6_T6', 'PB6_T6_diff', 'PB7_T6', 'PB7_T9', 'PB7_T6_diff', 'PB7_T9_diff', 'PB7_T10_T7', 'PB8_T6', 'PB8_T9', 'PB8_T6_diff', 'PB8_T9_diff', 'PB8_T10_T7', 'PB9_T6', 'PB9_T9', 'PB9_T6_diff', 'PB9_T9_diff', 'PB10_T6', 'PB10_T9', 'PB10_T6_diff', 'PB10_T9_diff', 'PB10_T10_T7', 'PB11_T6', 'PB11_T9', 'PB11_T6_diff', 'PB11_T9_diff', 'PB11_T10_T7', 'PB12_T6', 'PB12_T9', 'PB12_T6_diff', 'PB12_T9_diff', 'PB12_T10_T7', 'PB13_T6', 'PB13_T9', 'PB13_T6_diff', 'PB13_T9_diff', 'PB13_T10_T7', 'PB14_T6', 'PB14_T9', 'PB14_T6_diff', 'PB14_T9_diff', 'PB14_T10_T7', 'PB15_T6', 'PB15_T9', 'PB15_T6_diff', 'PB15_T9_diff', 'PB15_T10_T7', 'PB16_T6', 'PB16_T9', 'PB16_T6_diff', 'PB16_T9_diff', 'PB16_T10_T7', 'PB17_T6', 'PB17_T9', 'PB17_T6_diff', 'PB17_T9_diff', 'PB17_T10_T7', 'PB18_T6', 'PB18_T9', 'PB18_T6_diff', 'PB18_T9_diff', 'PB18_T10_T7', 'VB1_T6', 'VB1_T9', 'VB1_T6_diff', 'VB1_T9_diff', 'VB1_T10_T7', 'VB2_T6', 'VB2_T9', 'VB2_T6_diff', 'VB2_T9_diff', 'VB2_T10_T7', 'VB3_T6', 'VB3_T9', 'VB3_T6_diff', 'VB3_T9_diff', 'VB3_T10_T7', 'VB4_T6', 'VB4_T9', 'VB4_T6_diff', 'VB4_T9_diff', 'VB4_T10_T7', 'VB5_T6', 'VB5_T9', 'VB5_T6_diff', 'VB5_T9_diff', 'VB5_T10_T7', 'VB6_T6', 'VB6_T9', 'VB6_T6_diff', 'VB6_T9_diff', 'VB6_T10_T7', 'BB4_T6', 'BB4_T9', 'BB4_T6_diff', 'BB4_T9_diff', 'BB4_T10_T7', 'BB5_T6', 'BB5_T9', 'BB5_T6_diff', 'BB5_T9_diff', 'BB5_T10_T7', 'BB6_T6', 'BB6_T9', 'BB6_T6_diff', 'BB6_T9_diff', 'BB6_T10_T7', 'BB7_T6', 'BB7_T9', 'BB7_T6_diff', 'BB7_T9_diff', 'BB7_T10_T7', 'BB8_T6', 'BB8_T9', 'BB8_T6_diff', 'BB8_T9_diff', 'BB8_T10_T7', 'BB9_T6', 'BB9_T9', 'BB9_T6_diff', 'BB9_T9_diff', 'BB9_T10_T7', 'BB10_T6', 'BB10_T9', 'BB10_T6_diff', 'BB10_T9_diff', 'BB10_T10_T7', 'BB11_T6', 'BB11_T9', 'BB11_T6_diff', 'BB11_T9_diff', 'BB11_T10_T7', 'BB12_T6', 'BB12_T9', 'BB12_T6_diff', 'BB12_T9_diff', 'BB12_T10_T7', 'BB13_T6', 'BB13_T9', 'BB13_T6_diff', 'BB13_T9_diff', 'BB13_T10_T7', 'BB14_T6', 'BB14_T9', 'BB14_T6_diff', 'BB14_T9_diff', 'BB14_T10_T7', 'BB15_T6', 'BB15_T9', 'BB15_T6_diff', 'BB15_T9_diff', 'BB15_T10_T7']
        self.portfolio_tracker = PortfolioTracker(
            initial_portfolio=initial_portfolio,
            window_size=ROLLING_WINDOW,
            p_ppy=252,
            rf=0.0,
            transaction_cost=TRANSACTION_COST
        )
        self.initial_portfolio = initial_portfolio
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.current_portfolio = initial_portfolio
        self.max_drawdown = 0
        self.previous_price = None
        self.steps_since_last_learn = 0
        self.learn_frequency = 40  # Learn every 40 steps during trading
        
        # Absolute stop loss and take profit values for learning
        self.ABSOLUTE_STOP_LOSS = 0.001  # 0.1% absolute stop loss
        self.ABSOLUTE_TAKE_PROFIT = 0.003  # 0.3% absolute take profit

        self.df["signals"] = 0
        self.df["trade_type"] = 0
        self.df["final_signals"] = 0
        self.df["final_trade_type"] = 0
        self.df["log_signals"] = 0  # Separate column for logs with stop loss/take profit
        self.df["log_trade_type"] = 0
        self.positions = [0, 1, -1]
        feature_cols = [c for c in self.columns if c in self.df.columns and c != "Time"]
        self.columns = feature_cols  
        input_dims = len(self.columns) + 1  
        self.agent = Agent(input_dims=input_dims, n_actions=3)

    def _add_atr(self, atr_period=14):
        # Simple fix - handle both DataFrame and Series cases
        try:
            # Try to get as Series first
            price_close = self.df["Price_close"]
            if isinstance(price_close, pd.DataFrame):
                price_close = price_close.iloc[:, 0]
            
            self.df["prev_close"] = price_close.shift(1)
            
            # Get other price columns similarly
            for col in ["Price_high", "Price_low"]:
                if col in self.df.columns:
                    price_col = self.df[col]
                    if isinstance(price_col, pd.DataFrame):
                        self.df[col] = price_col.iloc[:, 0]
            
            self.df["tr1"] = self.df["Price_high"] - self.df["Price_low"]
            self.df["tr2"] = (self.df["Price_high"] - self.df["prev_close"]).abs()
            self.df["tr3"] = (self.df["Price_low"] - self.df["prev_close"]).abs()
            self.df["true_range"] = self.df[["tr1", "tr2", "tr3"]].max(axis=1)
            self.df["ATR"] = self.df["true_range"].rolling(window=atr_period, min_periods=1).mean()
            self.df.drop(columns=["prev_close", "tr1", "tr2", "tr3", "true_range"], inplace=True)
        except Exception as e:
            print(f"Error in _add_atr: {e}")
            print(f"Available columns: {self.df.columns.tolist()}")
            # Skip ATR calculation if it fails
            self.df["ATR"] = 0.0

    def _get_observation(self, idx, current_position):
        """
        Get observation with validation checks to ensure proper shape
        """
        try:
            # Get the column names excluding 'Time'
            feature_columns = [c for c in self.columns if c != "Time"]
            
            # Check if we have valid feature columns
            if not feature_columns:
                print(f"Warning: No feature columns found at idx {idx}")
                # Return default observation with correct dimensions
                default_features = np.zeros(len(self.columns), dtype=np.float32)
                default_features[-1] = float(current_position)
                return T.tensor(default_features, dtype=T.float32).to(self.agent.actor.device)
            
            # Get the row values
            row_values = self.df.loc[idx, feature_columns]
            
            # Convert to numpy array
            if hasattr(row_values, 'values'):
                features = row_values.values.astype(np.float32)
            else:
                # Handle scalar case
                features = np.array([row_values], dtype=np.float32)
            
            # Validate features
            if len(features) == 0 or np.any(np.isnan(features)):
                print(f"Warning: Invalid features at idx {idx}, using zeros")
                features = np.zeros(len(feature_columns), dtype=np.float32)
            
            # Ensure correct length
            if len(features) != len(feature_columns):
                print(f"Warning: Feature length mismatch at idx {idx}")
                print(f"Expected {len(feature_columns)}, got {len(features)}")
                features = np.zeros(len(feature_columns), dtype=np.float32)
            
            # Append current position
            observation = np.append(features, float(current_position))
            
            # Final validation
            expected_length = len(feature_columns) + 1
            if len(observation) != expected_length:
                print(f"Warning: Observation length mismatch at idx {idx}")
                print(f"Expected {expected_length}, got {len(observation)}")
                # Create proper observation
                observation = np.zeros(expected_length, dtype=np.float32)
                observation[-1] = float(current_position)
            
            return T.tensor(observation, dtype=T.float32).to(self.agent.actor.device)
            
        except Exception as e:
            print(f"Error in _get_observation at idx {idx}: {e}")
            print(f"Columns available: {self.columns}")
            print(f"DF shape: {self.df.shape}")
            
            # Return a safe default observation
            default_obs = np.zeros(len(self.columns) + 1, dtype=np.float32)
            default_obs[-1] = float(current_position)
            return T.tensor(default_obs, dtype=T.float32).to(self.agent.actor.device)

    def _calculate_price_change(self, current_idx):
        if current_idx == 0 or self.previous_price is None:
            return 0.0
        current_price = self.df.loc[current_idx, "Price_close"]
        return (current_price - self.previous_price) / self.previous_price

    def _check_stop_loss_take_profit(self, position, entry_price, current_price):
        """Check for absolute stop loss or take profit conditions"""
        if position == 0 or entry_price == 0:
            return False, False, 0.0
        
        if position == 1:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -self.ABSOLUTE_STOP_LOSS:
                return True, False, pnl_pct  # Stop loss hit
            elif pnl_pct >= self.ABSOLUTE_TAKE_PROFIT:
                return False, True, pnl_pct  # Take profit hit
        elif position == -1:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= -self.ABSOLUTE_STOP_LOSS:
                return True, False, pnl_pct  # Stop loss hit
            elif pnl_pct >= self.ABSOLUTE_TAKE_PROFIT:
                return False, True, pnl_pct  # Take profit hit
        
        return False, False, 0.0

    def optimisation(self, optim_size: int, start: int):
        """Training phase with absolute stop loss/take profit in reward calculation"""
        real_portfolio = self.current_portfolio
        position = 0
        entry_price = 0

        print(f" Optimisation: {optim_size} steps starting at {start}")
        print(f" Starting portfolio: ${real_portfolio:.2f}")

        end = min(start + optim_size, len(self.df))
        prev_experience = None

        for i in range(start, end):
            current_price = self.df.loc[i, "Price_close"]
            
            # Check for absolute stop loss/take profit (for reward adjustment only)
            stop_loss_hit, take_profit_hit, pnl_pct = self._check_stop_loss_take_profit(
                position, entry_price, current_price
            )

            observation = self._get_observation(i, position)

            action, probs, value = self.agent.choose_action(
                observation=observation, wlkfrwd=False, flag=position
            )

            price_change = self._calculate_price_change(i)
            intended_position = self.positions[action]

            # Exit position if signal changes
            if position != 0 and (intended_position != position or intended_position == 0):
                if position == 1:
                    pnl = (current_price - entry_price) / entry_price
                    real_portfolio = real_portfolio * (1 + pnl - TRANSACTION_COST)
                elif position == -1:
                    pnl = (entry_price - current_price) / entry_price
                    real_portfolio = real_portfolio * (1 + pnl - TRANSACTION_COST)
                position = 0
                entry_price = 0

            # Enter new position
            if intended_position != 0 and position == 0:
                position = intended_position
                entry_price = current_price

            self.portfolio_tracker.update_portfolio(real_portfolio)
            self.df.loc[i, "signals"] = position
            self.df.loc[i, "trade_type"] = (
                HOLD if position == HOLD else STRONG_BUY if position == BUY else STRONG_SELL
            )
            done = (i >= (start + optim_size - 1))
            if prev_experience is not None:
                prev_obs, prev_act, prev_prob, prev_val, prev_pos = prev_experience
                step_reward = self.portfolio_tracker.calculate_step_reward(
                    position, prev_pos, price_change, real_portfolio, 
                    current_price, entry_price, stop_loss_hit, take_profit_hit
                )
                self.agent.remember(
                    state=prev_obs.cpu().numpy() if isinstance(prev_obs, T.Tensor) else prev_obs,
                    action=prev_act,
                    probs=prev_prob,
                    vals=prev_val,
                    reward=step_reward,
                    done=done
                )

            prev_experience = (observation, action, probs, value, position)
            self.current_portfolio = real_portfolio
            self.previous_price = current_price

            if i % 30 == 0 and i > start:
                self.agent.learn()

        self.agent.learn()
        print(f" Optimisation complete. Final portfolio: ${real_portfolio:.2f}")
        self.current_portfolio = real_portfolio

    def walkforward(self, wf_size: int, start: int):
        """Walkforward phase with absolute stop loss/take profit in reward calculation"""
        real_portfolio = self.current_portfolio
        position = 0
        entry_price = 0

        print(f" Walkforward: {wf_size} steps starting at {start}")
        print(f" Starting portfolio: ${real_portfolio:.2f}")

        end = min(start + wf_size, len(self.df))
        prev_experience = None

        for i in range(start, end):
            current_price = self.df.loc[i, "Price_close"]
            
            # Check for absolute stop loss/take profit (for reward adjustment only)
            stop_loss_hit, take_profit_hit, pnl_pct = self._check_stop_loss_take_profit(
                position, entry_price, current_price
            )
            observation = self._get_observation(i, position)
            action, probs, value = self.agent.choose_action(
                observation=observation, wlkfrwd=True, flag=position
            )

            price_change = self._calculate_price_change(i)
            intended_position = self.positions[action]

            # Exit position if signal changes
            if position != 0 and (intended_position != position or intended_position == 0):
                if position == 1:
                    pnl = (current_price - entry_price) / entry_price
                    real_portfolio = real_portfolio * (1 + pnl - TRANSACTION_COST)
                elif position == -1:
                    pnl = (entry_price - current_price) / entry_price
                    real_portfolio = real_portfolio * (1 + pnl - TRANSACTION_COST)
                position = 0
                entry_price = 0

            # Enter new position
            if intended_position != 0 and position == 0:
                position = intended_position
                entry_price = current_price

            self.portfolio_tracker.update_portfolio(real_portfolio)
            self.df.loc[i, "signals"] = position
            self.df.loc[i, "trade_type"] = (
                HOLD if position == HOLD else STRONG_BUY if position == BUY else STRONG_SELL
            )

            done = (i >= (start + wf_size - 1))

            if prev_experience is not None:
                prev_obs, prev_act, prev_prob, prev_val, prev_pos = prev_experience
                step_reward = self.portfolio_tracker.calculate_step_reward(
                    position, prev_pos, price_change, real_portfolio, 
                    current_price, entry_price, stop_loss_hit, take_profit_hit
                )

                self.agent.remember(
                    state=prev_obs.cpu().numpy() if isinstance(prev_obs, T.Tensor) else prev_obs,
                    action=prev_act,
                    probs=prev_prob,
                    vals=prev_val,
                    reward=step_reward,
                    done=done
                )

            prev_experience = (observation, action, probs, value, position)
            self.current_portfolio = real_portfolio
            self.previous_price = current_price

        self.agent.learn()
        print(f" Walkforward complete. Final portfolio: ${real_portfolio:.2f}")
        self.current_portfolio = real_portfolio

    def train(self, batch_size: int, window_size: int,
              split_ratio: float, start: int):
        wf_size = int((1 - split_ratio) * window_size)
        optim_size = window_size - wf_size

        indices = range(0, batch_size, wf_size)

        print(f"Training batch: window_size={window_size}, "
              f"optim_size={optim_size}, wf_size={wf_size}")

        self.previous_price = None

        for i in indices:
            if i + window_size <= batch_size:
                self.optimisation(optim_size=optim_size, start=i + start)
                self.walkforward(wf_size=wf_size, start=i + optim_size + start)
            else:
                size_ = batch_size - i
                self.optimisation(optim_size=size_, start=i + start)

    def trade(self, batch_size: int, start: int):
        """Trading phase with learning and separate logs for absolute stop loss/take profit"""
        real_portfolio = self.current_portfolio
        position = 0  # PPO agent position
        entry_price = 0
        
        # Separate variables for logs with absolute stop loss/take profit
        log_position = 0
        log_entry_price = 0

        print(f"Trading: {batch_size} steps starting at {start}")
        print(f"Starting portfolio: ${real_portfolio:.2f}")

        end = min(start + batch_size, len(self.df))
        prev_experience = None

        for i in range(start, end):
            current_price = self.df.loc[i, "Price_close"]
            price_change = self._calculate_price_change(i)
            
            # Check for absolute stop loss/take profit for logs
            stop_loss_hit, take_profit_hit, pnl_pct = self._check_stop_loss_take_profit(
                log_position, log_entry_price, current_price
            )

            # PPO agent logic (with reward adjustment for stop loss/take profit)
            observation = self._get_observation(i, position)
            action, probs, value = self.agent.choose_action(
                observation=observation, wlkfrwd=True, flag=position
            )
            intended_position = self.positions[action]
            # Handle PPO agent position changes
            if position != 0 and (intended_position != position or intended_position == 0):
                if position == 1:
                    pnl = (current_price - entry_price) / entry_price
                    real_portfolio = real_portfolio * (1 + pnl - TRANSACTION_COST)
                elif position == -1:
                    pnl = (entry_price - current_price) / entry_price
                    real_portfolio = real_portfolio * (1 + pnl - TRANSACTION_COST)
                position = 0
                entry_price = 0
                # Reset log position when PPO agent exits
                log_position = 0
                log_entry_price = 0

            if intended_position != 0 and position == 0:
                position = intended_position
                entry_price = current_price
                
                # Start new log position
                log_position = intended_position
                log_entry_price = current_price

            # Handle absolute stop loss/take profit for logs
            if (stop_loss_hit or take_profit_hit) and log_position != 0:
                # Early exit in logs due to absolute stop loss/take profit
                if log_position == 1:
                    real_portfolio = real_portfolio * (1 + pnl_pct - TRANSACTION_COST)
                elif log_position == -1:
                    real_portfolio = real_portfolio * (1 + pnl_pct - TRANSACTION_COST)
                
                # Reset log position but keep PPO position (for learning)
                log_position = 0
                log_entry_price = 0

            self.portfolio_tracker.update_portfolio(real_portfolio)
            
            # Record PPO signals
            self.df.loc[i, "final_signals"] = position
            self.df.loc[i, "final_trade_type"] = (
                HOLD if position == HOLD else STRONG_BUY if position == BUY else STRONG_SELL
            )
            
            # Record log signals (with absolute stop loss/take profit exits)
            self.df.loc[i, "log_signals"] = log_position
            self.df.loc[i, "log_trade_type"] = (
                HOLD if log_position == HOLD else STRONG_BUY if log_position == BUY else STRONG_SELL
            )

            # Store experience and learn periodically
            done = (i >= (start + batch_size - 1))

            if prev_experience is not None:
                prev_obs, prev_act, prev_prob, prev_val, prev_pos = prev_experience
                step_reward = self.portfolio_tracker.calculate_step_reward(
                    position, prev_pos, price_change, real_portfolio, 
                    current_price, entry_price, stop_loss_hit, take_profit_hit
                )

                self.agent.remember(
                    state=prev_obs.cpu().numpy() if isinstance(prev_obs, T.Tensor) else prev_obs,
                    action=prev_act,
                    probs=prev_prob,
                    vals=prev_val,
                    reward=step_reward,
                    done=done
                )
            
            # Learn every 40 steps in trading phase
            self.steps_since_last_learn += 1
            if self.steps_since_last_learn >= self.learn_frequency:
                self.agent.learn()
                self.steps_since_last_learn = 0

            prev_experience = (observation, action, probs, value, position)
            self.current_portfolio = real_portfolio
            self.previous_price = current_price

        # Final learn at the end
        self.agent.learn()
        print(f" Trading complete. Final portfolio: ${real_portfolio:.2f}")
        self.current_portfolio = real_portfolio
        return real_portfolio

    def run(self, day_number=None, output_folder="daily_signals"):
        """Run training and trading for a day"""
        indices = range(0, len(self.df), self.batch_size)
        num_batches = len(indices)

        print(f"\nStarting run for day {day_number}: {num_batches} batches")
        print(f"Initial Portfolio: ${self.initial_portfolio:.2f}")

        self.previous_price = None
        self.steps_since_last_learn = 0

        print("Batch 0/{}: Trading".format(num_batches - 1))
        self.trade(batch_size=self.batch_size, start=indices[0])

        for i in range(num_batches - 1):
            print(f"\nBatch {i + 1}/{num_batches - 1}:")
            print(" Training phase...")
            self.train(batch_size=self.batch_size,
                      window_size=self.window_size,
                      split_ratio=self.split_ratio,
                      start=indices[i])

            print(" Trading phase...")
            self.trade(batch_size=self.batch_size, start=indices[i + 1])

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save detailed logs
        if day_number is not None:
            output_filename = os.path.join(output_folder, f"signals_day{day_number}.csv")
            compact_name = os.path.join(output_folder, f"day{day_number}_compact_log.csv")
            portfolio_log_name = os.path.join(output_folder, f"day{day_number}_portfolio_log.csv")
        else:
            output_filename = "Final_logs_PPO.csv"
            compact_name = "Final_logs_PPO_compact.csv"
            portfolio_log_name = "Final_portfolio_log.csv"

        # Save full dataframe
        self.df.to_csv(output_filename, index=False)
        print(f"Signals saved to: {output_filename}")

        # Create compact log with log signals (including absolute stop loss/take profit)
        compact_df = self.df[["Time", "Price_close", "log_signals"]].copy()
        compact_df.rename(columns={"log_signals": "signal"}, inplace=True)
        compact_df.to_csv(compact_name, index=False)
        print(f"Compact log with absolute stop loss/take profit saved to: {compact_name}")

        # Calculate portfolio performance based on logs
        final_portfolio, max_dd, trades = PortfolioTracker.calculate_portfolio_live(
            self.df,
            slipage=TRANSACTION_COST,
            initial_portfolio=self.initial_portfolio
        )

        daily_return = ((final_portfolio - self.initial_portfolio) /
                       self.initial_portfolio) * 100

        # Create portfolio log entry
        portfolio_log = pd.DataFrame([{
            'day': day_number,
            'initial_portfolio': self.initial_portfolio,
            'final_portfolio': final_portfolio,
            'daily_return_pct': daily_return,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'winning_trades': len([t for t in trades if t[3] > 0]) if trades else 0,
            'stop_loss_exits': len([t for t in trades if t[4] == 'STOP_LOSS']) if trades else 0,
            'take_profit_exits': len([t for t in trades if t[4] == 'TAKE_PROFIT']) if trades else 0
        }])
        # Save portfolio log
        portfolio_log.to_csv(portfolio_log_name, index=False)
        print(f"\nDay {day_number} Results:")
        print(f" Initial: ${self.initial_portfolio:.2f}")
        print(f" Final (with absolute stop loss/take profit): ${final_portfolio:.2f}")
        print(f" Return: {daily_return:.2f}%")
        print(f" Max DD: {max_dd:.2%}")
        print(f" Trades: {len(trades)}")
        if trades:
            winning_trades = [t for t in trades if t[3] > 0]
            print(f" Win Rate: {len(winning_trades) / len(trades) * 100:.1f}%")
            stop_loss_exits = len([t for t in trades if t[4] == 'STOP_LOSS'])
            take_profit_exits = len([t for t in trades if t[4] == 'TAKE_PROFIT'])
            print(f" Stop Loss Exits: {stop_loss_exits}")
            print(f" Take Profit Exits: {take_profit_exits}")
        return final_portfolio, portfolio_log


def run_on_days(folder_path: str = "EBY_processed_final",
                batch_size: int = 200,
                output_folder: str = "daily_signals"):
    """Run PPO trading on multiple days"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create daily portfolio summary file
    portfolio_summary_path = os.path.join(output_folder, "daily_portfolio_summary.csv")
    portfolio_summary = pd.DataFrame()

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    files=files[0:100]
    num_files = len(files)
    daily_performance = []
    
    print("=" * 80)
    print("STARTING MULTI-DAY TRADING SIMULATION")
    print("=" * 80)

    for i, csv_name in enumerate(files):
        csv_path = os.path.join(folder_path, csv_name)

        print(f"\n{'=' * 80}")
        print(f"PROCESSING DAY {i}: {csv_name}")
        print(f"{'=' * 80}")

        env = TradingEnv(df_path=csv_path, batch_size=batch_size)

        if i != 0:
            env.agent.load_models()
            print(f"Loaded model from Day {i - 1}")

        final_portfolio, day_portfolio_log = env.run(day_number=i, output_folder=output_folder)

        env.agent.save_models()
        print(f"Saved model for Day {i}")

        # Append to portfolio summary
        portfolio_summary = pd.concat([portfolio_summary, day_portfolio_log], ignore_index=True)
        portfolio_summary.to_csv(portfolio_summary_path, index=False)

        daily_return = ((final_portfolio - INITIAL_PORTFOLIO) /
                       INITIAL_PORTFOLIO) * 100

        daily_performance.append({
            "Day": i,
            "Date_File": csv_name,
            "Initial_Portfolio": INITIAL_PORTFOLIO,
            "Final_Portfolio": final_portfolio,
            "Daily_Return_Pct": daily_return,
            "Profit_Loss": final_portfolio - INITIAL_PORTFOLIO
        })

        print(f"{'-' * 80}")
        print(f"DAY {i} COMPLETE")
        print(f"{'-' * 80}")

    # Save overall performance summary
    summary_df = pd.DataFrame(daily_performance)
    summary_path = os.path.join(output_folder, "daily_performance_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("OVERALL TRADING SUMMARY")
    print("=" * 80)
    print(f"Total Days Traded: {num_files}")
    print(f"Total P/L: ${summary_df['Profit_Loss'].sum():.2f}")
    print(f"Average Daily Return: {summary_df['Daily_Return_Pct'].mean():.2f}%")
    print(f"Best Day Return: {summary_df['Daily_Return_Pct'].max():.2f}% "
          f"(Day {summary_df.loc[summary_df['Daily_Return_Pct'].idxmax(), 'Day']:.0f})")
    print(f"Worst Day Return: {summary_df['Daily_Return_Pct'].min():.2f}% "
          f"(Day {summary_df.loc[summary_df['Daily_Return_Pct'].idxmin(), 'Day']:.0f})")
    print(f"Win Rate: {(summary_df['Daily_Return_Pct'] > 0).sum() / len(summary_df) * 100:.2f}%")
    
    print(f"\nPortfolio summary saved to: {portfolio_summary_path}")
    print(f"Performance summary saved to: {summary_path}")
    print("=" * 80)

    return summary_df, portfolio_summary


if __name__ == "__main__":
    output_dir = "daily_signals"
    summary, portfolio_summary = run_on_days(
        folder_path=r"C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/EBY_processed3",
        batch_size=200,
        output_folder=output_dir
    )

    print("\nAll days processed successfully!")
    print(f"Individual signal files saved in: {output_dir}/")
    print(f"Performance summary saved in: {output_dir}/daily_performance_summary.csv")
    print(f"Portfolio summary saved in: {output_dir}/daily_portfolio_summary.csv")