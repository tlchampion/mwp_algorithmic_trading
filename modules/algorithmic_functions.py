import modules.helpers as helpers
import pandas as pd
import numpy as np
from pathlib import Path
from yahoo_fin.stock_info import get_data
import datetime
import pandas_ta as ta
import yfinance as yf
from pathlib import Path
from pandas.tseries.offsets import DateOffset






# create function to pull combined price information for a single risk-level portfolio
def get_portfolio_summary(name, start_year, start_month, start_day):
    tickers = helpers.get_ticker_by_port_name(name)
    stocks = helpers.get_stocks(tickers, start_year, start_month, start_day)
    weights = helpers.get_weights_by_name(name)
    ticker_data = stocks
    df = pd.DataFrame(np.zeros(stocks[tickers[0]].shape), columns=stocks[tickers[0]].columns, index=stocks[tickers[0]].index)
    for ticker in tickers:
        dfs = ticker_data[ticker]
        weight = weights.loc[ticker,'weight']
        dfs_weighted = dfs * weight
        df = df + dfs_weighted
        
    return df


# add indicators to portfolio dataframe
def add_indicators(df):
    MyStrategy = ta.Strategy(
    name="custom",
    ta=[
        {"kind": "sma","length": 30},
        {"kind": "sma","length": 100},
        {"kind": "macd"},
        {"kind": "bbands", "length": 20,"std" : 2}
        
    ]
)
    df.ta.strategy(MyStrategy)
    return df



# add signals to portfolio dataframe
def add_signals(df):
    # add columns for daily returns and use those to populate a column
    # indicating buy/sell/hold based on daily performance
    df.ta.log_return(cumulative=True, append=True)
    df.ta.log_return(cumulative=False, append=True)
    df.ta.percent_return(append=True, cumulative=True)
    df.ta.percent_return(append=True, cumulative=False)
    df['performance_signal'] = 0
    df['SMA_signal'] = 0
    df['MACD_signal'] = 0
    df['BB_signal'] = 0
    sma_position = 0
    macd_position = 0
    bb_position = 0
    for index, row in df.iterrows():
        if row['PCTRET_1'] > 0:
            df.loc[index,'performance_signal'] = 1
        elif row['PCTRET_1'] < 0:
            df.loc[index,'performance_signal'] = -1
    
        # create signal column based upon SMA 
        if row['SMA_30'] > row['SMA_100'] and sma_position != 1:
            df.loc[index,'SMA_signal'] = 1
            sma_position = 1
        elif row['SMA_30'] < row['SMA_100'] and sma_position != -1:
            df.loc[index,'SMA_signal'] = -1
            sma_position = -1
            
        # create signal column based upon MACD
        if row['MACD_12_26_9'] > row['MACDs_12_26_9'] and macd_position != 1:
            df.loc[index,'MACD_signal'] = 1
            macd_position = 1
        if row['MACD_12_26_9'] < row['MACDs_12_26_9'] and macd_position != -1:
            df.loc[index,'MACD_signal'] = -1
            macd_position = -1
            
        # create signal column based upon Bollinger Bands
        if row['close'] <  row['BBL_20_2.0'] and bb_position != 1:
            df.loc[index,'BB_signal'] = 1
            bb_position = 1
        if row['close'] >  row['BBU_20_2.0'] and bb_position != -1:
            df.loc[index,'BB_signal'] = -1
            bb_position = -1
    
    return df   

# build data to feed to ML model for daily predictions
def build_ml_prediction_data(name, year, month, day):
    df = get_portfolio_summary(name, year - 1, month, day)
    df.ta.log_return(cumulative=True, append=True)
    df.ta.log_return(cumulative=False, append=True)
    df.ta.percent_return(append=True, cumulative=True)
    df.ta.percent_return(append=True, cumulative=False)
    df = add_indicators(df)
    df = df.dropna()
    df['performance_signal'] = 0
    for index, row in df.iterrows():
        if row['PCTRET_1'] > 0:
            df.loc[index,'performance_signal'] = 1
        elif row['PCTRET_1'] < 0:
            df.loc[index,'performance_signal'] = -1
    df = df.drop(['open', 'high', 'low', 'close', 'adjclose', 'volume','CUMLOGRET_1','LOGRET_1', 'CUMPCTRET_1', 'PCTRET_1'], axis=1)
    start = str(datetime.datetime(year, month, day).date())
    return df.loc[start:,]

# build dataframe showing Bollinger Bands, SMA and MACD signals and 
# dataframe used for training ML models
def build_portfolio_signal_ml_df(name, start_year, start_month, start_day):
    summary = get_portfolio_summary(name, start_year, start_month, start_day)
    indicators = add_indicators(summary)
    signals = add_signals(indicators)
    signals = signals.dropna()
    ml = signals.drop(['open', 'high', 'low', 'close', 'adjclose', 'volume', 'SMA_signal', 'MACD_signal',
                       'BB_signal', 'CUMLOGRET_1','LOGRET_1', 'CUMPCTRET_1', 'PCTRET_1'], axis=1)
    return signals, ml.dropna()

def create_train_test(ml_df):
    X = ml_df.drop('performance_signal', axis=1).copy()
    cats = X.columns
    X = X[cats].shift().dropna()

    y = ml_df['performance_signal'].copy()
    
    training_begin = X.index.min()
    training_end = X.index.min() + DateOffset(months=36)


    X_train = X.loc[training_begin:training_end]
    X_test = X.loc[training_end:]
    y_train = y.loc[training_begin:training_end]
    y_test = y.loc[training_end:]

    # create X train/test datasets using various combinations of indicators
    X_train_sma = X_train[['SMA_30', 'SMA_100']]
    X_test_sma = X_test[['SMA_30', 'SMA_100']]
    X_train_macd = X_train[['MACD_12_26_9', 'MACDh_12_26_9','MACDs_12_26_9']]
    X_test_macd = X_test[['MACD_12_26_9', 'MACDh_12_26_9','MACDs_12_26_9']]
    X_train_bb = X_train[['BBL_20_2.0','BBM_20_2.0','BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
    X_test_bb = X_test[['BBL_20_2.0','BBM_20_2.0','BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
    
    # save X train/test datasets
    X_train.to_csv(Path("./data/X_train_full.csv"))
    X_test.to_csv(Path("./data/X_test_full.csv"))
    X_train_sma.to_csv(Path("./data/X_train_sma.csv"))
    X_test_sma.to_csv(Path("./data/X_test_sma.csv"))
    X_train_macd.to_csv(Path("./data/X_train_macd.csv"))
    X_test_macd.to_csv(Path("./data/X_test_macd.csv"))
    X_train_bb.to_csv(Path("./data/X_train_bb.csv"))
    X_test_bb.to_csv(Path("./data/X_test_bb.csv"))
    
    # save y train/test datasets
    y_train.to_csv(Path("./data/y_train.csv"))
    y_test.to_csv(Path("./data/y_test.csv"))