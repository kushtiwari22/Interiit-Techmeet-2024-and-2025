import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

signals_column = 'final_signals'

def read_file(file_path: str):
    df=pd.read_csv(file_path) #reading logs file
    df['datetime']=pd.to_datetime(df['datetime']) #changing to datetime object
    signal=df[df[signals_column]!=0] #filtering signals
    return signal

def calculate_calmar_ratio(portfolio_history):
    """
    Calculates the total return over the max drawdown for a single period (e.g., one day).
    This is the correct, non-annualized version of a Calmar Ratio for intraday backtests.
    """
    portfolio_arr = np.array(portfolio_history)
    initial_capital = portfolio_arr[0]
    
    # 1. Calculate Total Return for the period (NOT annualized)
    total_return = (portfolio_arr[-1] - initial_capital) / initial_capital

    # 2. Calculate Max Drawdown
    rolling_max = np.maximum.accumulate(portfolio_arr)
    drawdowns = (rolling_max - portfolio_arr) / rolling_max
    max_drawdown = np.max(drawdowns)

    if max_drawdown == 0:
        return np.inf # Return infinity if there was no drawdown

    # 3. Calculate the simple ratio of return to drawdown for the period
    ratio = total_return / max_drawdown
    return ratio
# =====================================================================
        
def engine(file_path: str, slipage=0.00, initial_portfolio=100.00, signals_column=signals_column):
    """
    A state-based backtesting engine with ADDITIVE PnL.
    *** THIS VERSION RUNS WITH INVERTED SIGNALS FOR ANALYSIS. ***
    """
    df = pd.read_csv(file_path)
    if 'Time' not in df.columns:
        df['Time'] = df.index
    else:
        df['Time'] = pd.to_datetime(df['Time'])

    portfolio = initial_portfolio
    portfolio_history = [initial_portfolio]
    current_position = 0
    entry_price = 0
    entry_time = None
    trades = []

    COST_PER_SIDE = 0.0002

    for i in range(len(df)):
        
        # --- START OF MINIMAL CHANGE: INVERT THE SIGNAL ---
        signal = df[signals_column].iloc[i]
        # --- END OF MINIMAL CHANGE ---
        
        current_price = df['Price_open'].iloc[i]
        current_time = df['Time'].iloc[i]

        # Rule 1: EXIT a LONG position
        if current_position == 1 and signal == -1:
            exit_price = current_price
            
            profit_dollars = exit_price - entry_price
            trade_return_pct = profit_dollars / entry_price
            commission_dollars = (entry_price * COST_PER_SIDE) + (exit_price * COST_PER_SIDE)
            net_profit = trade_return_pct * initial_portfolio - commission_dollars
            portfolio += net_profit
            
            trades.append({
                'return': trade_return_pct,
                'pnl': net_profit,
                'holding_time': current_time - entry_time
            })
            
            current_position = 0
            entry_price = 0

        # Rule 2: EXIT a SHORT position
        elif current_position == -1 and signal == 1:
            exit_price = current_price
            
            profit_dollars = entry_price - exit_price
            trade_return_pct = profit_dollars / entry_price
            commission_dollars = (entry_price * COST_PER_SIDE) + (exit_price * COST_PER_SIDE)
            net_profit = trade_return_pct * initial_portfolio - commission_dollars
            portfolio += net_profit
            
            trades.append({
                'return': trade_return_pct,
                'pnl': net_profit,
                'holding_time': current_time - entry_time
            })
            
            current_position = 0
            entry_price = 0
            
        # Rule 3: ENTER a new position
        elif current_position == 0 and signal != 0:
            current_position = signal
            entry_price = current_price
            entry_time = current_time
        
        portfolio_history.append(portfolio)

    # (The rest of the function remains exactly the same)
    # # ...
    # if current_position != 0:
    #     # ... (closing logic) ...
    #     final_price = df['Price_open'].iloc[-1]
    #     final_time = df['Time'].iloc[-1]
    #     if current_position == 1:
    #         profit_dollars = final_price - entry_price
    #         trade_return_pct = profit_dollars / entry_price
    #         commission_dollars = (entry_price * COST_PER_SIDE) + (final_price * COST_PER_SIDE)
    #         net_profit = trade_return_pct * initial_portfolio - commission_dollars
    #         portfolio += net_profit
    #         trades.append({'return': trade_return_pct, 'pnl': net_profit, 'holding_time': final_time - entry_time})
    #     elif current_position == -1:
    #         profit_dollars = entry_price - final_price
    #         trade_return_pct = profit_dollars / entry_price
    #         commission_dollars = (entry_price * COST_PER_SIDE) + (final_price * COST_PER_SIDE)
    #         net_profit = trade_return_pct * initial_portfolio - commission_dollars
    #         portfolio += net_profit
    #         trades.append({'return': trade_return_pct, 'pnl': net_profit, 'holding_time': final_time - entry_time})
    #     portfolio_history[-1] = portfolio

    # ... (metrics and printing logic) ...
    benchmark = initial_portfolio * (1 + ((df['Price_open'].iloc[-1] - df['Price_open'].iloc[0]) / df['Price_open'].iloc[0])) - COST_PER_SIDE * (df['Price_open'].iloc[-1] + df['Price_open'].iloc[0])
    if not trades:
        print("No trades were completed.")
        print("___________________________________________")
        print(f"--- Backtest Report ---")
        print(f"Number of trades: 0")
        print(f"Final portfolio value: {initial_portfolio:.2f}")
        print(f"Max drawdown: 0.00%")
        print(f"Calmar Ratio: N/A")
        print(f"Benchmark portfolio value: {benchmark:.2f}")
        print("---------------------------------------------")
        return initial_portfolio, 0.0, 0.0, benchmark - initial_portfolio, benchmark

    returns = np.array([t['return'] for t in trades])
    pnl = np.array([t['pnl'] for t in trades])
    trade_time = np.array([t['holding_time'] for t in trades])
    pnl_loss = pnl[pnl < 0] if len(pnl[pnl < 0]) > 0 else np.array([0])
    pnl_profit = pnl[pnl >= 0] if len(pnl[pnl >= 0]) > 0 else np.array([0])
    portfolio_arr = np.array(portfolio_history)
    rolling_max = np.maximum.accumulate(portfolio_arr)
    drawdowns = (rolling_max - portfolio_arr) / rolling_max
    max_drawdown = np.max(drawdowns)
    calmar = calculate_calmar_ratio(portfolio_history)
    print("___________________________________________")
    print(f"--- Backtest Report ---")
    print(f"Number of trades: {len(returns)}")
    print(f"% wins are : {100 * len(pnl_profit[pnl_profit > 0]) / len(returns):.2f}" if len(returns) > 0 else "N/A")
    print(f"Average PnL (Points): {np.mean(pnl):.4f}")
    print(f"Average Profit (Points): {np.mean(pnl_profit):.4f}")
    print(f"Average Loss (Points): {np.mean(pnl_loss):.4f}")
    print(f"Average holding time: {np.mean(trade_time)}")
    print(f"Max drawdown: {max_drawdown * 100:.2f}%")
    print(f"Calmar Ratio: {calmar:.2f}")
    print(f"Final portfolio value: {portfolio_history[-1]:.2f}")
    print(f"Benchmark portfolio value: {benchmark:.2f}")
    print("---------------------------------------------")
    return portfolio_history[-1], (portfolio_history[-1] - initial_portfolio), (max_drawdown * 100), benchmark - initial_portfolio, benchmark

def plot_portfolio(portfolio: np.array, i : int):
    plt.figure()
    sns.lineplot(portfolio)
    folder_path = '/home/ashisv/InterIIT14_PS1/EBY_results'
    img_name = f"Day{i}"
    path = os.path.join(folder_path, img_name)
    plt.savefig(path)

if __name__ == '__main__':
    n = len(os.listdir('/home/ashisv/InterIIT14_PS1/EBY_logs'))
    total_profit = 0
    max_ddown = 0
    total_benchmarkpnl = 0
    portfolio_arr = []
    benchmark_arr = []
    win_days = 0
    loss_days = 0
    win_days_arr = []
    loss_days_arr = []
    win_benchmark = 0
    loss_benchmark = 0
    win_benchmark_arr = []
    loss_benchmark_arr = []
    total_days = 0
    for i in range(0, 102):
        folder_path = '/home/ashisv/InterIIT14_PS1/EBY_logs'
        file_name = f'Final_logs_PPO_{i}.csv'
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            continue
        print(f"Day {i}:")
        portfolio, pnl, ddown, bpnl, becnhmark = engine(file_path= file_path)
        portfolio_arr.append(portfolio)
        benchmark_arr.append(becnhmark)
        total_days += 1

        if pnl > 0:
            win_days += 1
            win_days_arr.append(pnl)
        elif pnl < 0:
            loss_days += 1
            loss_days_arr.append(pnl)

        if bpnl > 0:
            win_benchmark += 1
            win_benchmark_arr.append(bpnl)
        elif bpnl < 0:
            loss_benchmark += 1
            loss_benchmark_arr.append(bpnl)

        total_profit += pnl
        total_benchmarkpnl += bpnl
        max_ddown = max(max_ddown, ddown)

    print("------------")
    print(f"Total PnL = {total_profit:.2f}")
    print(f"Max Drawdown = {max_ddown:.2f}")
    print(f"Total Benchmark PnL = {total_benchmarkpnl:.2f}")
    print("------------")

    # Stats
    win_series = pd.Series(win_days_arr)
    win_benchmark_series = pd.Series(win_benchmark_arr)
    loss_series = pd.Series(loss_days_arr)
    loss_benchmark_series = pd.Series(loss_benchmark_arr)
    print(f"Win days %age: {((win_days/total_days) * 100):.2f}")
    print(f"Loss days %age: {((loss_days/total_days) * 100):.2f}")
    print("Win statistics (Portfolio):")
    print(win_series.describe())
    print("Loss statistics (Portfolio):")
    print(loss_series.describe())

    print("Win statistics (Benchmark):")
    print(win_benchmark_series.describe())
    print("Loss statistics (Benchmark):")
    print(loss_benchmark_series.describe())

    # Set a professional plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 7))

    # Plot the strategy's equity curve
    plt.plot(portfolio_arr, label='Strategy Equity Curve', color='blue', linewidth=2)

    # # Plot the benchmark's equity curve
    plt.plot(benchmark_arr, label='Benchmark (Buy & Hold)', color='gray', linestyle='--', linewidth=1.5)

    # Add titles and labels for clarity
    plt.title('Strategy vs. Benchmark Equity Curve', fontsize=16)
    plt.xlabel('Time (in 60s Bars over Full Period)', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Add a horizontal line for the starting capital
    plt.axhline(y= 100.0, color='red', linestyle='-', linewidth=1, label='Initial Capital')

    plt.show()