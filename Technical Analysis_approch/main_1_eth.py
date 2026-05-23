from scipy.signal import cheby1, filtfilt
import pandas as pd
import uuid
import numpy as np
import warnings
import os
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('work')
sys.path.insert(1,'packages')
import pandas_ta as ta
import talib as pta
from untrade.client import Client
warnings.filterwarnings('ignore')

#------ Backtesting for normal files-----#
def perform_backtest(csv_file_path:str):
    client = Client()
    result = client.backtest(
     jupyter_id="team57_zelta_hpps",  # the one you use to login to jupyter.untrade.io
     file_path=csv_file_path,
     leverage=1# Adjust leverage as needed
    )
    for value in result:
        print(value)
    return result


#-----Backtesting for large csv-------#
def perform_backtest_large_csv(csv_file_path:str):
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

#-------- Data processing and calculations------#
def process_data(df: pd.DataFrame):
    # Heiken Ashi Calculations (vectorized)
    df['Heiken_Close'] = (df['open'] + df['close'] + df['high'] + df['low']) / 4
    df['Heiken_Open'] = df['open']
    for i in range(1, len(df)+1):
        df['Heiken_Open'][i] = (df['Heiken_Open'][i-1] + df['Heiken_Close'][i-1]) / 2
    df['Heiken_High'] = df[['high', 'Heiken_Open', 'Heiken_Close']].max(axis=1)
    df['Heiken_Low'] = df[['low', 'Heiken_Open', 'Heiken_Close']].min(axis=1)
    
    # Exponential Moving Averages (EMAs)
    df["EMA20"] = ta.ema(df['close'], length=20)
    df["EMA12"] = ta.ema(df['close'], length=12)
    df["EMA26"] = ta.ema(df['close'], length=26)
    
    #MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df["MACD_Signal_Line"] = ta.ema(df['MACD'], length=9)
    
    #Chaikin Volatility
    High_Low_Range = df['high'] - df['low']
    df['Chaikin_volatility'] = ta.ema(High_Low_Range, timeperiod=50).pct_change(periods=50) * 100
    
    #directional movement index
    df['Plus_DI_s'] = pta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=63)#importing talib as pta for specifically PLUS_DI
    df['Minus_DI_s'] = pta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=63)
    
    #Chande Momentum Oscillator
    Change = df['close'].diff()
    Gain = Change.apply(lambda x: x if x > 0 else 0)
    Loss = Change.apply(lambda x: -x if x < 0 else 0)    
    Sum_Gain = Gain.rolling(window=14).sum()
    Sum_Loss = Loss.rolling(window=14).sum()
    df['CMO'] = 100 * ((Sum_Gain - Sum_Loss) / (Sum_Gain + Sum_Loss))
    
    #Donchian Channel
    df['Donchian_High'] = df['high'].rolling(window=20).max()
    df['Donchian_Low'] = df['low'].rolling(window=20).min()
    df['Donchian_Middle'] = (df['Donchian_High'] + df['Donchian_Low']) / 2
    
    #Aroon Oscillator
    df['Aroon_Oscillator'] = pta.AROONOSC(df['high'], df['low'], timeperiod=25)
    
    #Commodity Channel Index
    df['CCI'] = pta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
    
    # Calculate the rolling threshold for pct_change_2
    df['pct_change_2'] = df['close'].pct_change(2)
    df['rolling_threshold'] = df['pct_change_2'].rolling(10).quantile(0.9)
    
    #Relative Strength Index
    df['RSI'] = ta.rsi(df['close'], timeperiod=14)
    #Smoothened RSI
    df['RSI_smoothed'] = df['RSI'].rolling(window=14).mean()
    
    #chebyshev_filter
    order=2
    ripple=0.03
    cutoff=0.1
    b, a = cheby1(N=order, rp=ripple, Wn=cutoff, btype='low', analog=False)
    df['Chebyshev_Filtered']=filtfilt(b, a, df['close'])

    return df

#-------Trade Strategy--------#
def strat(df: pd.DataFrame):
    flag = 0  # 0 = no trade, 1 = long trade active, -1 = short trade active
    df['signals'] = 0
    df['trade_type'] = [''] * len(df)
    
    time_calculator = 0  # To track time in active position
    timeout_flag = False  # To ensure stricter rules apply after timeout
    strict_conditions_applied = False  # Flag to apply stricter rules
    
    df['time_calculator'] = 0
    
    for i in range(len(df)):

        if pd.isna(df.iloc[i]).sum()>0: #disregarding rows with null values
            continue
            
        # Skip new trade entry checks if a trade is already active
        if flag != 0:
            time_calculator += 1
            df.loc[i, 'time_calculator'] = time_calculator
    
            # Long Exit or Timeout Exit
            if  (flag == 1 
                 and ((df['EMA12'][i]      < df['EMA26'][i] 
                 and df['CMO'][i-1]        >= 20 
                 and df['Heiken_Open'][i]  > df['EMA20'][i] 
                 and df['Heiken_Close'][i] < df['EMA20'][i]) 
                 or time_calculator        >= 400)):
                
                df.loc[i, "signals"] = -1
                flag = 0
                timeout_flag = time_calculator >= 400
                df.trade_type[i] = "close"
    
            # Short Exit or Timeout Exit
            elif (flag == -1 
                   and ((df['CCI'][i]            > -20 
                   and df['Aroon_Oscillator'][i] > -50 
                   and df['CMO'][i-1]            <= 30 
                   and df['close'][i]            > df['Donchian_Middle'][i-1] 
                   and df['pct_change_2'][i]     <= df['rolling_threshold'][i]) 
                   or time_calculator            >= 400)):
                
                df.loc[i, "signals"] = 1
                flag = 0
                timeout_flag = time_calculator >= 400
                df.trade_type[i] = "close"
    
            # Reset stricter conditions after timeout exit
            if timeout_flag:
                strict_conditions_applied = True
                timeout_flag = False
    
            # Skip further processing for this iteration
            continue
    
        # Reset time_calculator if no trade is active
        time_calculator = 0
    
        # Long Entry
        if (flag == 0 
            and not strict_conditions_applied 
            and df['EMA12'][i]               > df['EMA26'][i] 
            and df['Heiken_Open'][i]         < df['EMA20'][i] 
            and df['Chaikin_volatility'][i]  < 55 
            and df['Heiken_Close'][i]        > df['EMA20'][i]):
            
            df.loc[i, "signals"] = 1
            flag = 1
            df.trade_type[i] = "long"
    
        # Short Entry
        elif (flag == 0 
              and not strict_conditions_applied 
              and df['Chebyshev_Filtered'][i]   < df['close'][i] 
              and df['Minus_DI_s'][i]           > 0 
              and df['EMA12'][i]                < df['EMA26'][i] 
              and df['Minus_DI_s'][i]           > df['Plus_DI_s'][i]):
            
            df.loc[i, "signals"] = -1
            flag = -1
            df.trade_type[i] = "short"
    
        # Stricter Long Entry
        elif (flag == 0 
              and strict_conditions_applied
              and df['EMA12'][i]              > df['EMA26'][i] 
              and df['Heiken_Open'][i]        < df['EMA20'][i] 
              and df['Chaikin_volatility'][i] < 50 
              and df['Heiken_Close'][i]       > df['EMA20'][i] #strict conditions
              and df['RSI_smoothed'][i]       > 60):           # Additional strict condition
            
            df.loc[i, "signals"] = 1
            flag = 1
            df.trade_type[i] = "long"
    
        # Stricter Short Entry
        elif (flag == 0 
              and strict_conditions_applied 
              and df['Chebyshev_Filtered'][i] < df['close'][i] 
              and df['EMA12'][i]              < df['EMA26'][i] 
              and df['Minus_DI_s'][i]          > df['Plus_DI_s'][i] 
              and df['RSI_smoothed'][i]          < 30):  # Additional strict condition
            
            df.loc[i, "signals"] = -1
            flag = -1
            df.trade_type[i] = "short"
    
        
        if(strict_conditions_applied):
            strict_conditions_applied=False #Resetting flag to check for normal conditions
    return df

#-------------- Main ------------------#
def main():
    data=pd.read_csv("eth_data/ETHUSDT_6h.csv")                   #reading in data
    data=process_data(data)                                       #processing the data
    data=strat(data)                                              #generating trade signals
    data.to_csv('final_logs_eth.csv',index=False)                 #saving logs 
    perform_backtest_large_csv('final_logs_eth.csv')              #backtesting results
    
if __name__=='__main__':
    main()
