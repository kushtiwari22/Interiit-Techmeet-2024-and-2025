import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

class MarketRegimeAnalyzerCausal:
    def __init__(self):
        self.data = None
        self.regime_colors = {
            'RANGING': 'lightblue',
            'TRENDING_UP': 'lightgreen',
            'TRENDING_DOWN': 'lightcoral',
            'UNCLEAR': 'lightgray'
        }
        
    def load_data(self, df):
        """
        Load and prepare data - CASUAL VERSION
        """
        self.data = df.copy()
        
        if 'Time' in self.data.columns:
            self.data['Time'] = pd.to_datetime(self.data['Time'])
            # self.data.set_index('Time', inplace=True)
        
        self.data.sort_index(inplace=True)
        return self
    
    def extract_feature_groups(self):
        """Extract and group features by category and timeframe"""
        self.pb_features = {}
        self.vb_features = {}
        self.bb_features = {}
        
        for col in self.data.columns:
            if 'Price_' in col:
                continue
            
            if '_T' in col:
                parts = col.split('_')
                if len(parts) >= 2:
                    feature_name = parts[0]
                    timeframe = parts[-1]
                    
                    if feature_name.startswith('PB'):
                        if timeframe not in self.pb_features:
                            self.pb_features[timeframe] = []
                        self.pb_features[timeframe].append(col)
                    
                    elif feature_name.startswith('VB'):
                        if timeframe not in self.vb_features:
                            self.vb_features[timeframe] = []
                        self.vb_features[timeframe].append(col)
                    
                    elif feature_name.startswith('BB'):
                        if timeframe not in self.bb_features:
                            self.bb_features[timeframe] = []
                        self.bb_features[timeframe].append(col)
        
        return self
    
    def calculate_category_scores(self):
        """Calculate PB, VB, BB scores for each timeframe - CASUAL"""
        all_timeframes = sorted(list(set(
            list(self.pb_features.keys()) + 
            list(self.vb_features.keys()) + 
            list(self.bb_features.keys())
        )))
        
        self.timeframes = all_timeframes
        
        for timeframe in all_timeframes:
            if timeframe in self.pb_features:
                pb_cols = self.pb_features[timeframe]
                self.data[f'PB_Score_{timeframe}'] = self.data[pb_cols].mean(axis=1)
            
            if timeframe in self.vb_features:
                vb_cols = self.vb_features[timeframe]
                self.data[f'VB_Score_{timeframe}'] = self.data[vb_cols].mean(axis=1)
            
            if timeframe in self.bb_features:
                bb_cols = self.bb_features[timeframe]
                self.data[f'BB_Score_{timeframe}'] = self.data[bb_cols].mean(axis=1)
        
        return self
    
    def group_timeframes(self):
        """Group timeframes into Trend, Swing, and Timing layers - CASUAL"""
        timeframe_minutes = {'T6': 2, 'T7': 5, 'T8': 10, 'T9': 15, 'T10': 25}
        
        available_tf = self.timeframes
        
        trend_tf = []
        swing_tf = []
        timing_tf = []
        
        for tf in available_tf:
            if tf in timeframe_minutes:
                mins = timeframe_minutes[tf]
                if mins >= 20:
                    trend_tf.append(tf)
                elif mins >= 8:
                    swing_tf.append(tf)
                else:
                    timing_tf.append(tf)
        
        if not trend_tf:
            trend_tf = [max(available_tf, key=lambda x: timeframe_minutes.get(x, 0))]
        if not swing_tf:
            remaining = [tf for tf in available_tf if tf not in trend_tf + timing_tf]
            if remaining:
                swing_tf = [remaining[0]]
        if not timing_tf:
            timing_tf = [min(available_tf, key=lambda x: timeframe_minutes.get(x, 100))]
        
        # Calculate group averages - ALL ON CURRENT DATA (causal since features already lagged)
        # Trend Group
        pb_trend_cols = [f'PB_Score_{tf}' for tf in trend_tf if f'PB_Score_{tf}' in self.data.columns]
        if pb_trend_cols:
            self.data['PB_Trend'] = self.data[pb_trend_cols].mean(axis=1)
        
        vb_trend_cols = [f'VB_Score_{tf}' for tf in trend_tf if f'VB_Score_{tf}' in self.data.columns]
        if vb_trend_cols:
            self.data['VB_Trend'] = self.data[vb_trend_cols].mean(axis=1)
        
        bb_trend_cols = [f'BB_Score_{tf}' for tf in trend_tf if f'BB_Score_{tf}' in self.data.columns]
        if bb_trend_cols:
            self.data['BB_Trend'] = self.data[bb_trend_cols].mean(axis=1)
        
        # Swing Group
        pb_swing_cols = [f'PB_Score_{tf}' for tf in swing_tf if f'PB_Score_{tf}' in self.data.columns]
        if pb_swing_cols:
            self.data['PB_Swing'] = self.data[pb_swing_cols].mean(axis=1)
        
        vb_swing_cols = [f'VB_Score_{tf}' for tf in swing_tf if f'VB_Score_{tf}' in self.data.columns]
        if vb_swing_cols:
            self.data['VB_Swing'] = self.data[vb_swing_cols].mean(axis=1)
        
        bb_swing_cols = [f'BB_Score_{tf}' for tf in swing_tf if f'BB_Score_{tf}' in self.data.columns]
        if bb_swing_cols:
            self.data['BB_Swing'] = self.data[bb_swing_cols].mean(axis=1)
        
        # Timing Group
        pb_timing_cols = [f'PB_Score_{tf}' for tf in timing_tf if f'PB_Score_{tf}' in self.data.columns]
        if pb_timing_cols:
            self.data['PB_Timing'] = self.data[pb_timing_cols].mean(axis=1)
        
        vb_timing_cols = [f'VB_Score_{tf}' for tf in timing_tf if f'VB_Score_{tf}' in self.data.columns]
        if vb_timing_cols:
            self.data['VB_Timing'] = self.data[vb_timing_cols].mean(axis=1)
        
        bb_timing_cols = [f'BB_Score_{tf}' for tf in timing_tf if f'BB_Score_{tf}' in self.data.columns]
        if bb_timing_cols:
            self.data['BB_Timing'] = self.data[bb_timing_cols].mean(axis=1)
        
        self.trend_tf = trend_tf
        self.swing_tf = swing_tf
        self.timing_tf = timing_tf
        
        return self
    
    def calculate_composites_causal(self):
        """
        Calculate composites with LAGS to ensure causality
        """
        # Calculate composites on CURRENT data (causal since features already calculated from past)
        if all(col in self.data.columns for col in ['PB_Trend', 'BB_Trend', 'VB_Trend']):
            self.data['TC'] = (
                0.5 * self.data['PB_Trend'] + 
                0.3 * self.data['BB_Trend'] + 
                0.2 * self.data['VB_Trend']
            )
        
        if all(col in self.data.columns for col in ['PB_Swing', 'BB_Swing', 'VB_Swing']):
            self.data['SC'] = (
                0.4 * self.data['PB_Swing'] + 
                0.4 * self.data['BB_Swing'] + 
                0.2 * self.data['VB_Swing']
            )
        
        if all(col in self.data.columns for col in ['PB_Timing', 'BB_Timing', 'VB_Timing']):
            self.data['TiC'] = (
                0.6 * self.data['PB_Timing'] + 
                0.3 * self.data['BB_Timing'] + 
                0.1 * self.data['VB_Timing']
            )
        
        # Create LAGGED versions for regime detection
        # At time t, we can only use composites calculated at t-1 or earlier
        self.data['TC_lag'] = self.data['TC'].shift(1)
        self.data['SC_lag'] = self.data['SC'].shift(1)
        self.data['TiC_lag'] = self.data['TiC'].shift(1)
        
        # Direction Alignment Score using LAGGED values
        self.data['DA_Score'] = 0
        self.data['DA_Score'] += np.where(self.data['TC_lag'] > 0, 1, 
                                         np.where(self.data['TC_lag'] < 0, -1, 0))
        self.data['DA_Score'] += np.where(self.data['SC_lag'] > 0, 1,
                                         np.where(self.data['SC_lag'] < 0, -1, 0))
        self.data['DA_Score'] += np.where(self.data['TiC_lag'] > 0, 1,
                                         np.where(self.data['TiC_lag'] < 0, -1, 0))
        
        # Calculate change in SC using only PAST information
        # SC_Change[t] = SC[t-1] - SC[t-2] (no lookahead)
        self.data['SC_Change'] = self.data['SC'].diff().shift(1)  # Critical: shift by 1
        
        # Absolute values of lagged composites
        self.data['TC_abs'] = self.data['TC_lag'].abs()
        self.data['SC_abs'] = self.data['SC_lag'].abs()
        
        # Lag other needed features too
        if 'VB_Swing' in self.data.columns:
            self.data['VB_Swing_lag'] = self.data['VB_Swing'].shift(1)
        if 'BB_Swing' in self.data.columns:
            self.data['BB_Swing_lag'] = self.data['BB_Swing'].shift(1)
        if 'BB_Trend' in self.data.columns:
            self.data['BB_Trend_lag'] = self.data['BB_Trend'].shift(1)
        if 'VB_Trend' in self.data.columns:
            self.data['VB_Trend_lag'] = self.data['VB_Trend'].shift(1)
        
        return self
    
    def detect_regime_causal(self):
        """
        Detect market regime using ONLY LAGGED information
        Only RANGING and TRENDING regimes remain
        """
        self.data['Regime'] = 'UNCLEAR'
        self.data['Regime_Direction'] = 'NEUTRAL'
        
        # Use ONLY LAGGED values
        tc_abs = self.data['TC_abs']
        sc_abs = self.data['SC_abs']
        da_score = self.data['DA_Score']  # Already uses lagged values
        vb_trend = self.data.get('VB_Trend_lag', 0)
        sc_change_abs = self.data['SC_Change'].abs()
        bb_swing_abs = self.data.get('BB_Swing_lag', 0).abs()
        bb_trend_abs = self.data.get('BB_Trend_lag', 0).abs()
        tc_lag = self.data['TC_lag']
        sc_lag = self.data['SC_lag']
        vb_swing_lag = self.data.get('VB_Swing_lag', 0)

        # print(vb_trend.describe())
        
        # RANGING REGIME - ADJUSTED THRESHOLDS (more sensitive/easier to trigger)
        ranging_condition = (
            (tc_abs < 0.4) & # Reverted to 0.4 (Var 3)
            (sc_abs < 0.4) & # Reverted to 0.4 (Var 3)
            (sc_change_abs < 0.05) # Stricter: Extremely low volatility
        )
        
        # TRENDING REGIME - ADJUSTED THRESHOLDS (stricter/harder to trigger)
        trending_condition = (
            (tc_abs > 0.7) &  # Reverted to 0.7 (Var 3)
            (sc_abs > 0.5) &  # Reverted to 0.5 (Var 3)
            (da_score.abs() == 3) & # Reverted to 3 (Strict)
            (sc_change_abs > 0.08) & # Reverted to 0.08 (Var 14 sweet spot)
            (vb_trend.abs() > 0.3) # Stricter: Stronger volume support
        )
        
        # Apply regime detection using ONLY lagged data
        self.data.loc[ranging_condition, 'Regime'] = 'RANGING'
        
        trending_mask = trending_condition & ~ranging_condition
        self.data.loc[trending_mask, 'Regime'] = 'TRENDING'
        
        # Determine trend direction
        trending_up = trending_mask & (tc_lag > 0)
        trending_down = trending_mask & (tc_lag < 0)
        self.data.loc[trending_up, 'Regime_Direction'] = 'UP'
        self.data.loc[trending_down, 'Regime_Direction'] = 'DOWN'
        
        # Combine regime and direction for plotting
        self.data['Regime_Plot'] = self.data['Regime']
        not_ranging = self.data['Regime'] != 'RANGING'
        self.data.loc[not_ranging, 'Regime_Plot'] = (
            self.data.loc[not_ranging, 'Regime'] + '_' + 
            self.data.loc[not_ranging, 'Regime_Direction']
        )
        return self
    
    def calculate_regime_confidence_causal(self):
        """Calculate confidence score using only lagged values"""
        confidence_scores = []
        
        for idx, row in self.data.iterrows():
            regime = row['Regime']
            confidence = 0.0
            
            if regime == 'RANGING':
                tc_val = abs(row.get('TC_lag', 0))
                sc_val = abs(row.get('SC_lag', 0))
                vb_val = row.get('VB_Trend_lag', 0)
                
                # More sensitive scoring for ranging
                tc_condition = 1.0 if tc_val < 0.15 else max(0, 1 - tc_val / 0.15)
                sc_condition = 1.0 if sc_val < 0.25 else max(0, 1 - sc_val / 0.25)
                vb_condition = 1.0 if vb_val < -0.1 else max(0, (0 - vb_val) / 0.1)
                confidence = (tc_condition + sc_condition + vb_condition) / 3
            
            elif regime == 'TRENDING':
                tc_strength = min(1.0, abs(row.get('TC_lag', 0)) / 0.7)  # Reverted
                sc_strength = min(1.0, abs(row.get('SC_lag', 0)) / 0.5)  # Reverted
                alignment = 1.0 if abs(row.get('DA_Score', 0)) >= 2 else 0.5  # More flexible
                confidence = (0.4 * tc_strength + 0.3 * sc_strength + 0.3 * alignment)
            
            else:  # UNCLEAR regime
                # Calculate a neutral confidence based on how unclear it is
                tc_strength = min(1.0, abs(row.get('TC_lag', 0)) / 0.3)
                sc_strength = min(1.0, abs(row.get('SC_lag', 0)) / 0.2)
                # Lower confidence when it's unclear
                confidence = (tc_strength + sc_strength) / 6  # Normalized to 0-0.33 range
            
            confidence_scores.append(min(1.0, max(0.0, confidence)))
        
        self.data['Regime_Confidence'] = confidence_scores
        return self
    
    def plot_regime_on_price(self, start_date=None, end_date=None, figsize=(16, 8)):
        """
        Plot single price chart with regime background coloring
        """
        plot_data = self.data.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            plot_data = plot_data[plot_data.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            plot_data = plot_data[plot_data.index <= end_date]
        
        if len(plot_data) == 0:
            print("No data in specified date range")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price line
        ax.plot(plot_data.index, plot_data['Price_close'], 
                label='Close Price', color='black', linewidth=2, zorder=5)
        
        # Add high/low range shading
        ax.fill_between(plot_data.index, 
                        plot_data['Price_low'], 
                        plot_data['Price_high'],
                        alpha=0.2, color='gray', label='High/Low Range', zorder=4)
        
        # Color background by regime
        unique_regimes = plot_data['Regime_Plot'].unique()
        
        for regime in unique_regimes:
            if regime in self.regime_colors:
                regime_mask = plot_data['Regime_Plot'] == regime
                regime_starts = plot_data.index[regime_mask]
                
                if len(regime_starts) > 0:
                    periods = []
                    current_start = regime_starts[0]
                    
                    for i in range(1, len(regime_starts)):
                        time_diff = regime_starts[i] - regime_starts[i-1]
                        if time_diff > pd.Timedelta(minutes=30):
                            periods.append((current_start, regime_starts[i-1]))
                            current_start = regime_starts[i]
                    
                    periods.append((current_start, regime_starts[-1]))
                    
                    for start, end in periods:
                        ax.axvspan(start, end, alpha=0.3, 
                                  color=self.regime_colors.get(regime, 'lightgray'), zorder=1)
        
        # Formatting
        ax.set_title('Price Chart with Causal Market Regime Detection (Ranging/Trending Only)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3, zorder=2)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # Create legend
        regime_patches = []
        for regime in sorted(unique_regimes):
            if regime in self.regime_colors:
                regime_patches.append(
                    Patch(color=self.regime_colors[regime], alpha=0.3, label=regime)
                )
        
        price_line = plt.Line2D([0], [0], color='black', linewidth=2, label='Close Price')
        range_patch = Patch(color='gray', alpha=0.2, label='High/Low Range')
        
        all_handles = [price_line, range_patch] + regime_patches
        ax.legend(handles=all_handles, loc='upper left', fontsize=10, 
                 framealpha=0.9, ncol=2)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_causal(self, df):
        self.load_data(df)
        self.extract_feature_groups()
        self.calculate_category_scores()
        self.group_timeframes()
        self.calculate_composites_causal()
        self.detect_regime_causal()
        self.calculate_regime_confidence_causal()
        
        return self.data
    
# Load your data
class RegimeStrategy(MarketRegimeAnalyzerCausal):
    def __init__(self):
        super().__init__()
        
    def generate_signals(self, df):
        # 1. Run Analysis Pipeline
        # self.load_data(df)
        # self.extract_feature_groups()
        # self.calculate_category_scores()
        # self.group_timeframes()
        # self.calculate_composites_causal()
        # self.detect_regime_causal()

        self.data = self.analyze_causal(df)
        
        # 2. State Machine for Position Sizing
        # 0 = Flat, 100 = Long, -100 = Short
        
        positions = np.zeros(len(self.data))
        current_pos = 0.0
        
        # Iterate to apply Hysteresis logic
        # We iterate because 'UNCLEAR' depends on 'current_pos'
        regime_plot = self.data['Regime_Plot'].values
        confidence = self.data['Regime_Confidence'].values
        
        for i in range(len(self.data)):
            regime = regime_plot[i]
            conf = confidence[i]
            target_pos = current_pos # Default: Hold current state (Hysteresis)
            
            # STATE MACHINE
            if 'TRENDING_UP' in regime:
                target_pos = 50.0
            elif 'TRENDING_DOWN' in regime:
                target_pos = -50.0
            elif 'RANGING' in regime:
                target_pos = 0.0
            else:
                # Regime is UNCLEAR or TRENDING_NEUTRAL
                # Hysteresis Logic:
                # If we are Long, stay Long. 
                # If we are Short, stay Short.
                # If we are Flat, stay Flat.
                target_pos = current_pos 
            
            positions[i] = target_pos
            current_pos = target_pos
            
        self.data['position'] = positions
        
        # 3. Generate Signals (The Delta)
        # Signal = Position_Now - Position_Prev
        # Valid signals: +100 (0->50), -100 (0->-50), +100 (-50->50), -100 (50->-50), etc.
        self.data['signals'] = self.data['position'].diff().fillna(0)
        
        # 4. Mandatory End of Day Square-off
        # Force the last position to be 0
        last_idx = self.data.index[-1]
        
        # If we are holding a position at the last bar, we must close it
        # The 'signals' for the last bar must reflect the change from pos -> 0
        if len(self.data) > 0:
            final_pos = self.data['position'].iloc[-1]
            
            if final_pos != 0:
                # 1. Force Position to 0
                pos_col_idx = self.data.columns.get_loc('position')
                self.data.iloc[-1, pos_col_idx] = 0.0
                
                # 2. Recalculate Last Signal (0 - prev_pos)
                prev_pos = self.data['position'].iloc[-2] if len(self.data) > 1 else 0.0
                sig_col_idx = self.data.columns.get_loc('signals')
                self.data.iloc[-1, sig_col_idx] = 0.0 - prev_pos

        return self.data[['Time', 'Price_open', 'Price_close', 'Regime_Direction', 'Regime', 'Regime_Plot', 'position', 'signals']]

# --- 3. Execution Wrapper ---
def run_strategy(file_path):
    df = pd.read_csv(file_path)
    print(df.columns)
    
    strat = RegimeStrategy()
    result = strat.generate_signals(df)
    result.to_csv(f'logs/{os.path.basename(file_path)}')
    
    # Save or Print Stats
    trades = result[result['signals'] != 0]
    print(f"File: {os.path.basename(file_path)}")
    print(f"Total Signals Generated: {len(trades)}")
    print(f"Final Position: {result['position'].iloc[-1]}") # Should be 0
    print("-" * 30)
    
    # Optional: Save back to file
    # result.to_csv(f"Results/{os.path.basename(file_path)}", index=False)
    return result

if __name__ == '__main__':
    # Example usage
    # Replace with your folder path loop
    # folder_path = 'data_processed'
    # for file in os.listdir(folder_path):
    #     run_strategy(os.path.join(folder_path, file))
    
    # Single file test
    
    try:
        log_folder = 'data_processed' 

        # Sorting files numerically
        files = [f for f in os.listdir(log_folder) if f.endswith('.csv')]
        try:
            files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        except:
            files.sort()

        for file_name in files:
            file_path = os.path.join(log_folder, file_name)
            run_strategy(file_path= file_path)
    except FileNotFoundError:
        print("File not found, please check path.")