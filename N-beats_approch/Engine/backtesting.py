import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def read_file(file_path: str):
    df=pd.read_csv(file_path) #reading logs file
    df['datetime']=pd.to_datetime(df['datetime']) #changing to datetime object
    signal=df[df['signals']!=0] #filtering signals
    return signal

def engine(file_path:str, slipage=0.002, initial_portfolio=1000.00):
    signal=read_file(file_path)
    returns=[]     #list to track returns
    trade_time=[]  #list to track time of execution of each trade
    
    for i in range(0,len(signal),2):
        if signal['signals'].iloc[i]==1: #long position
            entry=signal['close'].iloc[i]
            exit=signal['close'].iloc[i+1]
            returns.append((exit-entry)/entry)
            trade_time.append(signal['datetime'].iloc[i+1]-signal['datetime'].iloc[i])
        else: #short position
            entry=signal['close'].iloc[i]
            exit=signal['close'].iloc[i+1]
            returns.append((entry-exit)/entry)
            trade_time.append(signal['datetime'].iloc[i+1]-signal['datetime'].iloc[i])
            
    trade_time=np.array(trade_time) #converting to numpy array
    returns=np.array(returns)       #converting to numpy array
    returns_2=1+returns-slipage     #portfolio mutliplication factors
    portfolio=[initial_portfolio]   #list to track portfolio values
    
    for i in returns_2:
        portfolio.append(portfolio[-1]*i)
        
    max_drawdown=0
    peak_portfolio=initial_portfolio
    
    for i in portfolio:
        if i > peak_portfolio:
            peak_portfolio=i
        else:
            drawdown=(peak_portfolio-i)/peak_portfolio
            max_drawdown=max(drawdown,max_drawdown)
            
    pnl=[] #list to track profit and loss
    
    for i in range(1,len(portfolio)):
        pnl.append(portfolio[i]-portfolio[i-1])
        
    pnl=np.array(pnl)
    pnl_loss=pnl[pnl<0]
    pnl_profit=pnl[pnl>=0]
    benchmark=initial_portfolio*(1-slipage+(signal['close'].iloc[-1]-signal['close'].iloc[0])/signal['close'].iloc[0])

    print(f"Average holding time: {np.mean(trade_time)}")
    print(f"Average Returns: {np.mean(pnl)}")
    print(f"Average Profit: {np.mean(pnl_profit)}")
    print(f"Average Loss: {np.mean(pnl_loss)}")
    print(f"Max holding time: {np.max(trade_time)}")
    print(f"Maximum Profit: {np.max(pnl)}")
    print(f"Maximum Loss: {np.min(pnl)}")
    print(f"Number of trades: {len(returns)}")
    print(f"Number of wins: {len(pnl_profit)}")
    print(f"Number of losses: {len(pnl_loss)}")
    print(f"% wins are : {100*len(pnl_profit)/len(returns)}")
    print(f'intial portfolio value: {initial_portfolio}')
    print(f'final portfolio value: {portfolio[-1]}')
    print(f'benchmark portfolio value: {benchmark}')
    print(f"Peak portfolio Value: {max(portfolio)}")
    print(f"lowest portfolio Value: {min(portfolio)}")
    print(f"max drawdown: {max_drawdown*100}%")
    print(f"From: {signal['datetime'].iloc[0]}")
    print(f"To: {signal['datetime'].iloc[-1]}")
    
    return portfolio

def plot_portfolio(portfolio: np.array):
    sns.lineplot(portfolio)