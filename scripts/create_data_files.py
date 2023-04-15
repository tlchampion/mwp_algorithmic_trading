

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas_ta as ta

import modules.helpers as helpers
import modules.HistoricalData as hst
import modules.MCTab as MCTab
import modules.intro as intro
import modules.profile as prf
import modules.algorithmic_functions as af
import modules.AlgoTab as at
from pandas.tseries.offsets import DateOffset
from joblib import dump, load
from modules.MCForecastTools import MCSimulation
import hvplot.pandas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from pandas.tseries.offsets import DateOffset
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score 
from sklearn.ensemble import RandomForestClassifier


default_initial_investment = 150000
default_share_size = {'conservative': 650,
                      'balanced': 200,
                      'growth': 400,
                      'aggressive': 350,
                      'alternative': 400}

dataset_type = {'conservative': 'full', 'balanced': 'reduced', 'growth': 'full', 'aggressive': 'full', 'alternative': 'full'}


# download historical stock information for each portfolio
# combine individual stock information into one comprehensive set
# based upon weight rations of stocks in portfolio
def download_historical_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']

    for c in classes:
        tickers = helpers.get_ticker_by_port_name(c)
        stocks = helpers.get_stocks(tickers, 2017, 12, 31)
        weights = helpers.get_weights_by_name(c)
        ticker_data = stocks.copy()
        df = pd.DataFrame(np.zeros(stocks[tickers[0]].shape), columns=stocks[tickers[0]].columns, index=stocks[tickers[0]].index)
        for ticker in tickers:
            dfs = ticker_data[ticker]
            weight = weights.loc[ticker,'weight']
            dfs_weighted = dfs * weight
            df = df + dfs_weighted
        filepath = Path(f"../data/historical/{c}.csv")   
        df.to_csv(filepath)


# add indicators to portfolio dataframe using pandas-ta library
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


# add buy/sell signals based upon markent indicators and daily stock performance
def add_signals(df):
    # add columns for daily returns and use those to populate a column
    # indicating buy/sell/hold based on daily performance
    df.ta.log_return(cumulative=True, append=True)
    df.ta.log_return(cumulative=False, append=True)
    df.ta.percent_return(append=True, cumulative=True)
    df.ta.percent_return(append=True, cumulative=False)

    # default is a 'sell' status for all signals
    df['performance_signal'] = 0
    df['SMA_signal'] = 0
    df['MACD_signal'] = 0
    df['BB_signal'] = 0
    df['RSI_signal'] = 0
    df['STOCH_signal'] = 0

    # if there is a positive percent return we flag a 'buy' status
    # otherwise keep it a 'sell'
    for index, row in df.iterrows():
        if row['PCTRET_1'] >= 0:
            df.loc[index,'performance_signal'] = 1

        
        sma_position = 0
        # create signal column based upon SMA
        # buy if sma30 >= sma100, sell if SMA30 < SMA 100

        # if row['SMA_30'] >= row['SMA_100'] and sma_position != 1:
        #     df.loc[index,'SMA_signal'] = 1
        #     sma_position = 1
        # elif row['SMA_30'] < row['SMA_100'] and sma_position != 0:
        #     df.loc[index,'SMA_signal'] = 0
        #     sma_position = 0

        if row['SMA_30'] >= row['SMA_100']:
            df.loc[index,'SMA_signal'] = 1
           
        elif row['SMA_30'] < row['SMA_100']:
            df.loc[index,'SMA_signal'] = 0
           
            
        # create signal column based upon MACD
        # buy if MACD >= MACDs, sell if MACD < MACDs
        if row['MACD_12_26_9'] >= row['MACDs_12_26_9']:
            df.loc[index,'MACD_signal'] = 1
       
    
            
        # create signal column based upon Bollinger Bands
        # if closing price is <= lower bollinger band trigger a buy condition if not already in a buy condition
        # if closing price is > lower bollinger band trigger a sell condition if not already in a sell condition
        # otherwise no change in buy/sell condition
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
        # if rsi <= 30 a buy condition is set unless already in a buy condition
        # if rsi > 30 a sell condition is set unless already in a sell condition
        # other wise no change to buy/sell condition
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
        # if STOCHk < 20 a buy condition is triggered unless already in a buy condition
        # if STOCHk > 80 a sell condition is triggered unless already in a sell condition
        # otherwise no change in buy/sell condition
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


# convert stock performance dataframe into dataframe containing market indicators
# and associated buy/sell signals
def build_portfolio_signal_ml_df(name):
    summary = pd.read_csv(Path(f"../data/historical/{name}.csv"), parse_dates=True, infer_datetime_format=True, index_col="Unnamed: 0")
    indicators = add_indicators(summary)
    signals = add_signals(indicators)
    signals = signals.dropna()
    min_date = signals.index.min()
    sp_close = get_sp_close(min_date)
    signals = pd.concat([signals, sp_close], axis=1)
    ml = signals.drop(['open', 'high', 'low', 'close', 'adjclose', 'volume', 'SMA_signal', 'MACD_signal',
                       'BB_signal', 'RSI_signal', 'STOCH_signal','CUMLOGRET_1','LOGRET_1', 'CUMPCTRET_1', 'PCTRET_1','sp_close'], axis=1)
    return signals, ml.dropna()



# add columns to historical portoflio data to track daily stock holding, investment value,
# cash holdings, daily returns and cumulative returns based upon investment strategy buy/sell signals
# also add daily returns and cumulative returns for portfolio without factoring in buy/sell signals
def calculate_daily_values(df, signal, initial_capital=default_initial_investment, share_size=500):
    
    # daily position in portfolio is determined by a base investment in the portfolio equal to
    # that portfolios share size (X).
    # if in a 'buy' status for that day the share size is doubled (position = 2X)
    df['Position'] = (share_size * df[signal]) + share_size


    # Determine the points in time where shares are bought or sold
    # This is based upon the difference in positions between days
    # first row always has an Entry/Exit equal to that day's position
    df['Entry/Exit Position'] = df['Position'].diff()
    df.loc[df.index[0],'Entry/Exit Position'] = df.loc[df.index[0],'Position']

    # Multiply the close price by the number of shares held, or the Position,
    # to determine the value of the portfolio holdings
    df['Portfolio Holdings'] = df['close'] * df['Position']

    # Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
    # this indicates the amount left in the 'cash reserve'
    df['Portfolio Cash'] = initial_capital - (df['close'] * df['Entry/Exit Position']).cumsum()

    # Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
    df['Portfolio Total'] = df['Portfolio Cash'] + df['Portfolio Holdings']

    # Calculate the portfolio daily returns
    df['Portfolio Daily Returns'] = df['Portfolio Total'].pct_change()

    # Calculate the portfolio cumulative returns
    df['Portfolio Cumulative Returns'] = (1 + df['Portfolio Daily Returns']).cumprod() - 1
    
    # Calculate the daily returns for non-strategy trading
    # this is based on the portfolios close price, not calcualted portfolo value
    df['Base Daily Returns'] = df['close'].pct_change()
    
    # Calculate the cumulative returns for non-strategy trading
    # this is based on the portfolios close price, not calcualted portfolo value
    df['Base Cumulative Returns'] = (1 + df['Base Daily Returns']).cumprod() - 1

    # Calculate the daily returns for S&P 500
    df['Market Daily Returns'] = df['sp_close'].pct_change()
    
    # Calculate the cumulative returns for S&P 500
    df['Market Cumulative Returns'] = (1 + df['Market Daily Returns']).cumprod() - 1

    # return dataframe
    return df


# loop through all portfolio/investment strategy options and create performance data
# for use on dashboard
def create_performance_data():
    # define the portfolio classes to be used
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']

    # define the portfolio classes that use TensorFlow for the machine learning strategy signal
    nn_classes = ['balanced']

    # define if the full or reduced feature dataset was used for the chosen TensorFlow model
    nn_data_type = "reduced"

    # define the strategies available for each portfolio
    strategies_list = {'conservative': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'balanced': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'growth': ['sma', 'rsi', 'macd','stoch','bb'],
              'aggressive': ['sma', 'rsi', 'macd','stoch','bb'],
              'alternative': ['sma', 'rsi', 'macd','stoch','bb']
             }


    for c in classes:

        
        df, ml = build_portfolio_signal_ml_df(c)
        # we will only show performance data for the time period not encompassed by the train/test datasets used for ML training
        # this way comparisons between signal and ml strategies are using the same timeperiod

        start_date = df.index.min() + DateOffset(months=36)
        share_size = default_share_size[c]
        df_ml = df.copy()
        df = df.loc[start_date:,]



        # create performance dataframes for each strategy defined for the portfolio class
        # save dataframes for on-demand use on dashboard
        strategies = strategies_list[c]
        for s in strategies:
            ind = s.upper() + '_signal'
            
            performance = calculate_daily_values(df, ind, share_size=share_size)
            performance = performance[['close', ind, 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                      'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns', 'Market Daily Returns', 'Market Cumulative Returns']].loc[start_date:,]
            performance = performance.loc[start_date:,]
            performance.reset_index(inplace=True)
            file_name = f"performance_data_{s}_{c}.csv"
            file_path = Path(f"../data/performance/{file_name}")
            performance.to_csv(file_path, index=False)
            
        # create performance dataframes for each portfolio for the select ML model


        # first import and make predictions with the appropriate model for each portfolio
        # prediction dataset was created and saved at the time train/test data was created
        if c not in nn_classes:
    
    
            # import model 
            filepath = Path(f"../modeling/saved_models/{c}.joblib")
            
            with open(filepath, 'rb') as file:
                model = load(file) 
            # load data to make predictions on
            dataset = dataset_type[c]
            file = Path(f"../data/ml_prediction_data/X_pred_{dataset}_{c}.csv")
            pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'Unnamed: 0')
            preds = model.predict(pred_data)
            
        else:
            filepath = Path(f"../modeling/saved_models/{c}.h5")
            dataset = dataset_type[c]
            model = tf.keras.models.load_model(filepath)
            file = Path(f"../data/ml_prediction_data/X_pred_{dataset}_{c}.csv")
            pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'Unnamed: 0')
            predictions = model.predict(pred_data)
            preds = tf.cast(predictions >= 0.5, tf.int32)
        
        
        
        # create dataframe holding model predictions
        preds_df = pd.DataFrame(index=pred_data.index)
        preds_df['model_signal'] = preds


        # add predictions back to a copy of the historical data for the portfolio
        df_ml = df_ml.loc[preds_df.index[0]:]
        df_ml = pd.concat([df_ml, preds_df], axis=1)

        # create performance dataframe and save for on-demand usage by dashboard
        performance = calculate_daily_values(df_ml, 'model_signal', share_size=share_size)

        performance = performance[['close', 'model_signal', 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                  'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns', 'Market Daily Returns', 'Market Cumulative Returns']].loc[start_date:,]
        performance = performance.loc[start_date:,]
        performance.reset_index(inplace=True)

        filename = f"performance_data_ml_{c}.csv"
        file_path = Path(f"../data/performance/{filename}")
        performance.to_csv(file_path, index=False)
    
# # create dataframe with daily and cumulative returns for S&P 500
# def create_market_data():
#     market = helpers.get_stocks(['^GSPC'])
#     market = market['^GSPC']
#     market = market.loc[af.default_test_start_date:,]
#     market['market_daily_returns'] = market['close'].pct_change()
#     market.dropna(inplace=True)
#     market['market_cum_returns'] = (1 + market['market_daily_returns']).cumprod() - 1
#     market.reset_index(inplace = True)
#     market.to_csv(Path("../data/at_market_data.csv"), index=False)
 
# get closing price for S&P 500   
def get_sp_close(start_date):
    market = helpers.get_stocks(['^GSPC'])
    market = market['^GSPC']
    market = market.loc[start_date:,]
    market_close = market[['close']]
    market_close.columns = ['sp_close']
    return market_close


# def create_ml_prediction_data():
#     classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
#     reduced = ['balanced']
#     for c in classes:
#         df = af.build_ml_prediction_data(c)
#         df.drop(['performance_signal'], axis=1, inplace=True)
#         if c in reduced:
#             df = df[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
#                           'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
#         df.reset_index(inplace = True)
#         filename = f"ml_prediction_data_{c}.csv"
#         file_path = Path(f"../data/ml_prediction_data/{filename}")
#         df.to_csv(file_path, index=False)
 

# for Monte Carlo simulations we need more data than is available in just the predictions dataframe
# we add in the extra data for the test period as well since it was not used to train the ML model
# only for evaluation purposes       
def MC_create_ml_prediction_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    reduced = ['balanced']
    for c in classes:
        test = pd.read_csv(Path(f"../modeling/data/X_test_full_{c}.csv"), index_col='Unnamed: 0', infer_datetime_format=True, parse_dates=True)
        train = pd.read_csv(Path(f"../data/ml_prediction_data/X_pred_full_{c}.csv"),index_col='Unnamed: 0', infer_datetime_format=True, parse_dates=True)
        df = pd.concat([train,test])
        # df = af.MC_build_ml_prediction_data(c)
        # df.drop(['performance_signal'], axis=1, inplace=True)
        if dataset_type[c] == 'reduced':
            df = df[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
        df.reset_index(inplace = True)
        filename = f"mc_ml_prediction_data_{c}.csv"
        file_path = Path(f"../MCdata/MC_ml_prediction_data/{filename}")
        df.to_csv(file_path, index=False)


# create performance data to be used for Monte Carlo simulations
# steps are the same as for creating performance data for the investment strategies
def MC_create_performance_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    nn_classes = ['balanced']
    strategies_list = {'conservative': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'balanced': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'growth': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'aggressive': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'alternative': ['sma', 'rsi', 'macd','stoch','bb']
             }
    for c in classes:
        start_date = datetime.datetime.strptime('2018-4-1', '%Y-%m-%d')
        df, ml = build_portfolio_signal_ml_df(c)
        share_size = default_share_size[c]
        df_ml = df.copy()
        # df = df.loc[af.default_test_start_date:,]
        # create performance dataframes for each strategy defined for the portfolio class
        strategies = strategies_list[c]
        for s in strategies:
            ind = s.upper() + '_signal'
            
            performance = calculate_daily_values(df, ind, share_size=share_size)
            performance = performance[['close', ind, 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                      'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns', 'Market Daily Returns', 'Market Cumulative Returns']].loc[start_date:,]
            performance = performance.loc[start_date:,]
            performance.reset_index(inplace=True)
            performance.dropna(inplace=True)
            file_name = f"mc_performance_data_{s}_{c}.csv"
            file_path = Path(f"../MCdata/MCperformance/{file_name}")
            performance.to_csv(file_path, index=False)
            
        # create performance dataframes for each strategy for the select ML model
       
        # import model
        # filepath = Path(f"../modeling/saved_models/{c}.joblib")
        # with open(filepath, 'rb') as file:
        #     model = load(file) 
        # # load data to make predictions on
        # file = Path(f"../MCdata/mc_ml_prediction_data/mc_ml_prediction_data_{c}.csv")
        # pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'index')
        # preds = model.predict(pred_data)
        if c not in nn_classes:
    
    
            # import model
            filepath = Path(f"../modeling/saved_models/{c}.joblib")
            with open(filepath, 'rb') as file:
                model = load(file) 
            # load data to make predictions on
            file = Path(f"../MCdata/mc_ml_prediction_data/mc_ml_prediction_data_{c}.csv")
            pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'index')
            preds = model.predict(pred_data)
            
        else:
            filepath = Path(f"../modeling/saved_models/{c}.h5")
            model = tf.keras.models.load_model(filepath)
            file = Path(f"../MCdata/mc_ml_prediction_data/mc_ml_prediction_data_{c}.csv")
            pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'index')
            predictions = model.predict(pred_data)
            preds = tf.cast(predictions >= 0.5, tf.int32)



        preds_df = pd.DataFrame(index=pred_data.index)
        preds_df['model_signal'] = preds



        df_ml = df_ml.loc[preds_df.index[0]:]
        df_ml = pd.concat([df_ml, preds_df], axis=1)

        performance = calculate_daily_values(df_ml, 'model_signal', share_size=share_size)

        performance = performance[['close', 'model_signal', 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                  'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns', 'Market Daily Returns', 'Market Cumulative Returns']].loc[start_date:,]
        # performance = performance.loc[af.default_test_start_date:,]
        performance.reset_index(inplace=True)
        performance.dropna(inplace=True)
        filename = f"mc_performance_data_ml_{c}.csv"
        file_path = Path(f"../MCdata/MCperformance/{filename}")
        performance.to_csv(file_path, index=False)

        
      
# create train/test/prediction datasets for each portfolio class        
def create_train_test():
    
    portfolios=['conservative', 'balanced','growth',
                                  'aggressive', 'alternative']
    # loop through portfolios and create datasets.
    # train is first 24 months
    # test is next 12 months
    # prediction is remaining
    
    for port in portfolios:
        signals_df, ml_df = build_portfolio_signal_ml_df(f'{port}')

        X = ml_df.drop('performance_signal', axis=1).copy()
        cats = X.columns
        X = X[cats].shift().dropna()

        y = ml_df['performance_signal'].copy()

        training_begin = X.index.min()
        training_end = X.index.min() + DateOffset(months=24)
        test_end = training_end + DateOffset(months=12)

        X_train = X.loc[training_begin:training_end]
        X_test = X.loc[training_end:test_end]
        X_pred = X.loc[test_end:]

        y_train = y.loc[training_begin:training_end]
        y_test = y.loc[training_end:test_end]
        y_pred = y.loc[test_end:]
        

        # # save X_train/test datasets
        X_train.to_csv(Path(f"../modeling/data/X_train_full_{port}.csv"))
        X_test.to_csv(Path(f"../modeling/data/X_test_full_{port}.csv"))
        X_pred.to_csv(Path(f"../data/ml_prediction_data/X_pred_full_{port}.csv"))

        # save y train/test datasets
        y_train.to_csv(Path(f"../modeling/data/y_train_{port}.csv"))
        y_test.to_csv(Path(f"../modeling/data/y_test_{port}.csv"))
        y_pred.to_csv(Path(f"../data/ml_prediction_data/y_pred_{port}.csv"))
        
        
        
        
        # reduce features in datasets
        X_train = X_train[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
        X_test = X_test[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
        X_pred = X_pred[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
        
        # # save X_train/test datasets with reduced features
        X_train.to_csv(Path(f"../modeling/data/X_train_reduced_{port}.csv"))
        X_test.to_csv(Path(f"../modeling/data/X_test_reduced_{port}.csv"))
        X_pred.to_csv(Path(f"../data/ml_prediction_data/X_pred_reduced_{port}.csv"))
        

# compile MC simulation plot, MC distribution plot, MC summary and MC confidence interval verbiage
def prep_strategy_MC_data(ticker_data):

    # weight_list = weights['weight'].to_list()


    simulation = MCSimulation(
        portfolio_data = ticker_data,
        weights=[1.0],
        num_simulation = 200,
        num_trading_days =252*10
    )


    simulation.calc_cumulative_return()
    invested_amount = 150000
    simulation_plot = simulation.plot_simulation()
    distribution_plot = simulation.plot_distribution()
    summary = simulation.summarize_cumulative_return()
    ci_lower_ten_cumulative_return = round(summary[8]*invested_amount,2)
    ci_upper_ten_cumulative_return = round(summary[9]*invested_amount,2)
    text = f"""
            There is a 95% chance that the final portfolio value after 10 years will be within the range of ${ci_lower_ten_cumulative_return:,.2f} and ${ci_upper_ten_cumulative_return:,.2f} based upon an initial investment of ${invested_amount:,.2f}
            """
    # hvplot.save(simulation_plot, Path("./figures/test.png"))
    # distribution_plot.savefig(Path("./figures/test2.png"))
    return simulation_plot, distribution_plot, summary, text


# loop through portfolios and strategies to call prep_strategy_MC_data
# save resulting Monte Carlo data for display on dashboard 
def create_mc_info():

    portfolios = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    strategies = ['macd', 'ml', 'rsi', 'sma', 'stoch', 'bb']

    for p in portfolios:
        for s in strategies:
            df = pd.read_csv(Path(f"../MCdata/MCperformance/mc_performance_data_{s}_{p}.csv"),infer_datetime_format=True, parse_dates=True, index_col="index")
            df['type'] = f"{p}"
            df.set_index('type', append=True, inplace=True)
            df = df.unstack()
            df = df.reorder_levels([1,0], axis=1)
            df.rename(columns = {'Portfolio Daily Returns':'daily_return'}, inplace = True)
            simulation_plot, distribution_plot, summary, text = prep_strategy_MC_data(df)
            hvplot.save(simulation_plot, Path(f"../figures/simulation_{s}_{p}.png"))
            distribution_plot.savefig(Path(f"../figures/distribution_{s}_{p}.png"))
            items = [summary, text]
            filepath = Path(f"../MCdata/mcItems_{s}_{p}.joblib")
            with open(filepath, 'wb') as file:
                dump(items, file)
        


def create_data_only():
    print("downloading historical data for portfolios")
    download_historical_data()
    print("creating train/test datasets for ML modeling\n")
    create_train_test()
    print("creating strategy performance data\n")
    create_performance_data()
    print("creating MC prediction data\n")
    MC_create_ml_prediction_data()
    print("creating MC performance data\n")
    MC_create_performance_data()
    print("creating MC graphs\n")
    create_mc_info()


if __name__ == "__main__":
    print("Preparing to refresh all data files. This may take a few mintues. Please be patient...")
    create_data_only()



    