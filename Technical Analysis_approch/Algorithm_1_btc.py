import pandas as pd
import numpy as np
from scipy.signal import cheby1, filtfilt
import os 
import warnings
import uuid
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('work')
sys.path.insert(1,'packages')
import pandas_ta as ta
from untrade.client import Client
warnings.filterwarnings('ignore')
import talib as pta
#-------Custom classses and functions-------#

class KalmanFilter:
    def __init__(self, Q=1e-7, R=1e-2):
        # Initialize variables for process variance (Q) and measurement variance (R)
        self.Q = Q
        self.R = R
        self.P = 1.0  # Initial estimate error covariance
        self.X = 0.0  # Initial state estimate

    def predict(self):
        # Prediction step (Kalman prediction)
        self.P = self.P + self.Q  # Increase uncertainty
        return self.X

    def update(self, measurement):
        # Update step (Kalman correction)
        K = self.P / (self.P + self.R)  # Kalman gain
        self.X = self.X + K * (measurement - self.X)  # Update the estimate
        self.P = (1 - K) * self.P  # Update the error covariance

def hawkes_process(data: pd.Series, kappa: float):
    assert kappa > 0.0 
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    
    return pd.Series(output, index=data.index) * kappa
    
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
def process_data(data: pd.DataFrame):
    
    #heiken Ashi 
    data['Heiken_Close'] = (data['open'] + data['close'] + data['high'] + data['low']) / 4
    data['Heiken_Open'] = data['open']
    for i in range(1, len(data)+1):
        data['Heiken_Open'][i] = (data['Heiken_Open'][i-1] + data['Heiken_Close'][i-1]) / 2
    data['Heiken_High'] = data[['high', 'Heiken_Open', 'Heiken_Close']].max(axis=1)
    data['Heiken_Low'] = data[['low', 'Heiken_Open', 'Heiken_Close']].min(axis=1)
    
    #Exponential Moving Averages
    data["EMA6"] = ta.ema(data['close'], length=6)
    data["EMA10"] = ta.ema(data['close'], length=10)
    data["EMA20"] = ta.ema(data['close'], length=20)
    data["EMA50"] = ta.ema(data['close'], length=50)
    
    #Moving Average Convergence Divergence
    EMA12=ta.ema(data['close'], length=12)
    EMA26=ta.ema(data['close'], length=26)
    
    data['MACD'] = EMA12 - EMA26
    data["MACD_Signal_Line"] = ta.ema(data['MACD'], length=9)
    
    #Relative Strength Index
    RSI = ta.rsi(data['close'], timeperiod=14)
    #Smoothened RSI
    data['RSI_smoothed'] = RSI.rolling(window=5).mean()
    
    #Chaikin Volatility
    High_Low_Range = data['high'] - data['low']
    data['Chaikin_volatility'] = ta.ema(High_Low_Range, timeperiod=10).pct_change(periods=10) * 100
    
    #Know Sure Thing(KST)
    roc1 = ta.roc(data['close'], timeperiod=10)
    roc2 = ta.roc(data['close'], timeperiod=15)
    roc3 = ta.roc(data['close'], timeperiod=20)
    roc4 = ta.roc(data['close'], timeperiod=30)
    
    data['KST'] = (
        ta.sma(roc1, timeperiod=10) +
        2 * ta.sma(roc2, timeperiod=10) +
        3 * ta.sma(roc3, timeperiod=10) +
        4 * ta.sma(roc4, timeperiod=15))
    data['KST_signal'] = ta.sma(data['KST'], timeperiod=9)  # KST Signal Line
    
    # Apply the Hawkes process
    data['Hawkes_close'] = hawkes_process(data['close'], kappa=0.06)
    
    #Elder Ray Index
    EMA13 = ta.ema(data['close'], length=13)
    data['Bull_Power'] = data['high'] - EMA13
    data['Bear_Power'] = data['low'] - EMA13
    
    # Calculate Directional Movement Index (DMI) components: +DI and -DI
    data['Plus_DI'] = pta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    data['Minus_DI'] = pta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    data['Plus_DI_s'] = pta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=63)
    data['Minus_DI_s'] = pta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=63)

    #Aroon Oscillator
    data['Aroon_Oscillator'] = pta.AROONOSC(data['high'], data['low'], timeperiod=25)
    
    # Calculate the rolling threshold for pct_change_5
    data['pct_change_5'] = data['close'].pct_change(5)
    data['rolling_threshold'] = data['pct_change_5'].rolling(10).quantile(0.9)
    
    #kalman filtering
    filtered_price=[]
    kf=KalmanFilter() #filter intialisation
    for i in range(len(data)):
        kf.update(data['close'].iloc[i])
        filtered_price.append(kf.predict())
    data['filtered_price']=filtered_price

    #chebyshev filter
    order=2
    ripple=0.03
    cutoff=0.1
    b, a = cheby1(N=order, rp=ripple, Wn=cutoff, btype='low', analog=False)
    data['Chebyshev_Filtered']=filtfilt(b, a, data['close'])
    return data

#-------Trade Strategy--------#
def strat(df: pd.DataFrame):
    flag = 0  # 0 = no trade, 1 = long trade active, -1 = short trade active
    df['signals']=0
    df['trade_type'] = ['']*len(df)
    
    # Iterate over rows to evaluate conditions
    for i in range(len(df)):
        
        if pd.isna(df.iloc[i]).sum()>0: #disregarding rows with null values
            continue
            
        # Long Entry
        if (flag == 0 
            and df['EMA20'].iloc[i]              > df['EMA50'].iloc[i]
            and df['Heiken_Open'].iloc[i]        < df['EMA20'].iloc[i]
            and df['Heiken_Close'].iloc[i]       > df['EMA20'].iloc[i]
            and df['MACD'].iloc[i]               < df['MACD_Signal_Line'].iloc[i]
            and df['RSI_smoothed'].iloc[i]       > 50
            and df['Hawkes_close'].iloc[i]       > df['close'].iloc[i]
            and df['Chaikin_volatility'].iloc[i] < 25
            and df['KST'].iloc[i]                < df['KST_signal'].iloc[i]
            and df['Bull_Power'].iloc[i]         > 0  # Bull Power positive
            and df['Bear_Power'].iloc[i]         < 0  # Bear Power negative
            and df['Plus_DI'].iloc[i]            > df['Minus_DI'].iloc[i]  # +DI higher than -DI for upward trend
            and df['Plus_DI'].iloc[i]            > 10):  # Significant positive DI
            
            df["signals"].iloc[i] = 1  # Signal to go long
            flag = 1  # Long trade active
            df['trade_type'].iloc[i]="long"
            
    
        # Long Exit
        elif (flag == 1 
              and df['EMA20'].iloc[i]        < df['EMA50'].iloc[i]
              and df['filtered_price'][i]    < df['close'].iloc[i]
              and df['Heiken_Open'].iloc[i]  > df['EMA20'].iloc[i]
              and df['Heiken_Close'].iloc[i] < df['EMA20'].iloc[i]):
            
              df["signals"].iloc[i] = -1  # Signal to exit long trade
              flag = 0  # No trade active
              df['trade_type'].iloc[i]="close"
              
    
        # Short Entry
        elif (flag == 0 
              and df['EMA6'].iloc[i]         < df['EMA10'].iloc[i]
              and df['Heiken_Open'].iloc[i]  > df['EMA6'].iloc[i]
              and df['Heiken_Close'].iloc[i] < df['EMA6'].iloc[i]
              and df['MACD'].iloc[i]         > df['MACD_Signal_Line'].iloc[i]
              and df['Minus_DI_s'].iloc[i]   > df['Plus_DI_s'].iloc[i]
              and df['Chebyshev_Filtered'][i]   < df['close'][i]
              and df['RSI_smoothed'].iloc[i] < 40):
                
              df["signals"].iloc[i] = -1  # Signal to go short
              flag = -1  # Short trade active
              df['trade_type'].iloc[i]="short"
              
    
        # Short Exit
        elif (flag == -1 
              and df['EMA6'].iloc[i]         > df['EMA10'].iloc[i] 
              and df['Heiken_Open'].iloc[i]  < df['EMA6'].iloc[i]
              and df['Heiken_Close'].iloc[i] > df['EMA6'].iloc[i]
              and df['Aroon_Oscillator'][i] > -10
              and df['pct_change_5'][i] <= df['rolling_threshold'][i]):
                
              df.loc[i, "signals"] = 1  # Signal to exit short trade
              flag = 0  # No trade active
              df.trade_type[i]="close"
    return df

#-------------- Main ------------------#
def main():
    data=pd.read_csv('btc_data/BTC_2019_2023_2h.csv',index_col=0) #reading in data
    data=process_data(data)                                       #processing the data
    data=strat(data)                                              #generating trade signals
    data.to_csv('final_logs_btc.csv',index=False)                 #saving logs 
    perform_backtest_large_csv('final_logs_btc.csv')              #backtesting results
    
if __name__=='__main__':
    main()