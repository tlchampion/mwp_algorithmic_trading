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

default_test_start_year = 2022
default_test_start_month = 1
default_test_start_day = 10
default_test_start_date = datetime.datetime.strptime('2022-1-10', '%Y-%m-%d')

default_initial_investment = 150000
default_share_size = {'conservative': 650,
                      'balanced': 200,
                      'growth': 400,
                      'aggressive': 350,
                      'alternative': 400}


# create function to pull combined price information for a single risk-level portfolio
def get_portfolio_summary(name, start_year=2017, start_month=12, start_day=31):
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
        {"kind": "sma", "length": 200},
        {"kind": "ema", "length": 50},
        {"kind": "macd"},
        {"kind": "bbands", "length": 20,"std" : 2},
        {"kind": "rsi"},
        {"kind": "hlc3"},
        {"kind": "ohlc4"},
        {"kind": "linreg"},
        {"kind": "stoch"}
        
    ]
)
    df.ta.strategy(MyStrategy)
    return df
  


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
    df['RSI_signal'] = 0
    df['STOCH_signal'] = 0

    for index, row in df.iterrows():
        if row['PCTRET_1'] >= 0:
            df.loc[index,'performance_signal'] = 1
        # elif row['PCTRET_1'] < 0:
        #     df.loc[index,'performance_signal'] = -1
        
        sma_position = 0
        # create signal column based upon SMA 
        if row['SMA_30'] >= row['SMA_100'] and sma_position != 1:
            df.loc[index,'SMA_signal'] = 1
            sma_position = 1
        elif row['SMA_30'] < row['SMA_100'] and sma_position != 0:
            df.loc[index,'SMA_signal'] = 0
            sma_position = 0
            
        # create signal column based upon MACD
        if row['MACD_12_26_9'] >= row['MACDs_12_26_9']:
            df.loc[index,'MACD_signal'] = 1
       
        # if row['MACD_12_26_9'] < row['MACDs_12_26_9'] and macd_position != -1:
        #     df.loc[index,'MACD_signal'] = -1
        #     macd_position = -1
            
        # create signal column based upon Bollinger Bands
        bb_position = 0
        if row['close'] <=  row['BBL_20_2.0'] and bb_position != 1:
            df.loc[index,'BB_signal'] = 1
            bb_position = 1
        elif row['close'] >  row['BBU_20_2.0'] and bb_position != 0:
            df.loc[index,'BB_signal'] = 0
            bb_position = 0
        else: 
            df.loc[index,'BB_signal'] = bb_position
        
        # generate RSI signal column
        rsi_position = 0
        if row['RSI_14'] <= 30 and rsi_position != 1:
            df.loc[index,'RSI_signal'] = 1
            rsi_position = 1
        elif row['RSI_14'] >=  70 and rsi_position != 0:
            df.loc[index,'RSI_signal'] = 0
            rsi_position = 0
        else: 
            df.loc[index,'RSI_signal'] = rsi_position 
            
        # generate STOCH signal
        stoch_position = 0
        if row['STOCHk_14_3_3'] < 20 and stoch_position != 1:
            df.loc[index, 'STOCH_signal'] = 1
            stoch_position = 1
        elif row['STOCHk_14_3_3'] > 80 and stoch_position != 0:
            df.loc[index, 'STOCH_signal'] = 0
            stoch_position = 0
        else:
            df.loc[index, 'STOCH_signal'] = stoch_position
            
    
    return df 

# build data to feed to ML model for daily predictions
def build_ml_prediction_data(portfolio_class, 
                             year=default_test_start_year,
                             month=default_test_start_month,
                             day=default_test_start_day):
    df = get_portfolio_summary(portfolio_class, year - 1, month, day)
    df.ta.log_return(cumulative=True, append=True)
    df.ta.log_return(cumulative=False, append=True)
    df.ta.percent_return(append=True, cumulative=True)
    df.ta.percent_return(append=True, cumulative=False)
    df = add_indicators(df)
    df = df.dropna()
    df['performance_signal'] = 0
    for index, row in df.iterrows():
        if row['PCTRET_1'] >= 0:
            df.loc[index,'performance_signal'] = 1
        # elif row['PCTRET_1'] < 0:
        #     df.loc[index,'performance_signal'] = -1
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
                       'BB_signal', 'RSI_signal', 'STOCH_signal','CUMLOGRET_1','LOGRET_1', 'CUMPCTRET_1', 'PCTRET_1'], axis=1)
    return signals, ml.dropna()

# def create_train_test():
    
#     portfolios=['conservative', 'balanced','growth',
#                                   'aggressive', 'alternative']
#     # loop through portfolios and create datasets
    
#     for port in portfolios:
#         signals_df, ml_df = build_portfolio_signal_ml_df(f'{port}',2017,12,31)

#         X = ml_df.drop('performance_signal', axis=1).copy()
#         cats = X.columns
#         X = X[cats].shift().dropna()

#         y = ml_df['performance_signal'].copy()

#         training_begin = X.index.min()
#         training_end = X.index.min() + DateOffset(months=36)


#         X_train = X.loc[training_begin:training_end]
#         X_test = X.loc[training_end:]
#         y_train = y.loc[training_begin:training_end]
#         y_test = y.loc[training_end:]


#         # # save X_train/test datasets
#         X_train.to_csv(Path(f"./data/X_train_full_{port}.csv"))
#         X_test.to_csv(Path(f"./data/X_test_full_{port}.csv"))


#         # save y train/test datasets
#         y_train.to_csv(Path(f"./data/y_train_{port}.csv"))
#         y_test.to_csv(Path(f"./data/y_test_{port}.csv"))


# create dataframe to use for graphing portfoliio activity over time based upon a
# specified trading signal
def create_portfolio_performance_data(df, signal, initial_capital=default_initial_investment, share_size=500):
    
 
    df['Position'] = (share_size * df[signal]) + share_size


    # Determine the points in time where shares are bought or sold
    df['Entry/Exit Position'] = df['Position'].diff()
    df.loc[df.index[0],'Entry/Exit Position'] = df.loc[df.index[0],'Position']

    # Multiply the close price by the number of shares held, or the Position
    df['Portfolio Holdings'] = df['close'] * df['Position']

    # Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
    df['Portfolio Cash'] = initial_capital - (df['close'] * df['Entry/Exit Position']).cumsum()

    # Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
    df['Portfolio Total'] = df['Portfolio Cash'] + df['Portfolio Holdings']

    # Calculate the portfolio daily returns
    df['Portfolio Daily Returns'] = df['Portfolio Total'].pct_change()

    # Calculate the portfolio cumulative returns
    df['Portfolio Cumulative Returns'] = (1 + df['Portfolio Daily Returns']).cumprod() - 1
    
    # Calculate the daily returns for non-strategy trading
    df['Base Daily Returns'] = df['close'].pct_change()
    
    # Calculate the cumulative returns for non-strategy trading
    df['Base Cumulative Returns'] = (1 + df['Base Daily Returns']).cumprod() - 1

    # return dataframe
    return df

# convert ML model predictions to dataframe to be used for portfolio performance review
def prep_ml_prediction_signals(predictions, test_data, port_class):
    preds_df = pd.DataFrame(index=test_data.index)
    preds_df['model_signal'] = predictions
    signals, _ = af.build_portfolio_signal_ml_df(port_class, 2017,12,31)
    signals = signals.loc[preds_df.index[0]:]
    signals = pd.concat([signals, preds_df], axis=1)
    return signals