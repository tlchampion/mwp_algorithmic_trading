import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import modules.helpers as helpers
import modules.HistoricalData as hst
import modules.MCTab as MCTab
import modules.intro as intro
import modules.profile as prf
import modules.algorithmic_functions as af
import modules.AlgoTab as at
from pandas.tseries.offsets import DateOffset
from joblib import dump, load




def create_performance_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    strategies_list = {'conservative': ['sma', 'rsi', 'macd','stoch'],
              'balanced': ['sma', 'rsi', 'macd','stoch'],
              'growth': ['sma', 'rsi', 'macd','stoch'],
              'aggressive': ['sma', 'rsi', 'macd','stoch'],
              'alternative': ['sma', 'rsi', 'macd','stoch']
             }
    for c in classes:
        start_date = af.default_test_start_date
        df, ml = af.build_portfolio_signal_ml_df(c, 2017, 12, 31)
        share_size = af.default_share_size[c]
        df_ml = df.copy()
        df = df.loc[af.default_test_start_date:,]
        # create performance dataframes for each strategy defined for the portfolio class
        strategies = strategies_list[c]
        for s in strategies:
            ind = s.upper() + '_signal'
            
            performance = af.create_portfolio_performance_data(df, ind, share_size=share_size)
            performance = performance[['close', ind, 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                      'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns']].loc[start_date:,]
            performance = performance.loc[start_date:,]
            performance.reset_index(inplace=True)
            file_name = f"performance_data_{s}_{c}.csv"
            file_path = Path(f"../data/performance/{file_name}")
            performance.to_csv(file_path, index=False)
            
        # create performance dataframes for each strategy for the select ML model
       
        # import model
        filepath = Path(f"../modeling/saved_models/{c}.joblib")
        model = load(filepath) 
        # load data to make predictions on
        file = Path(f"../data/ml_prediction_data/ml_prediction_data_{c}.csv")
        pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'index')
        preds = model.predict(pred_data)
        preds_df = pd.DataFrame(index=pred_data.index)
        preds_df['model_signal'] = preds



        df_ml = df_ml.loc[preds_df.index[0]:]
        df_ml = pd.concat([df_ml, preds_df], axis=1)

        performance = af.create_portfolio_performance_data(df_ml, 'model_signal', share_size=share_size)

        performance = performance[['close', 'model_signal', 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                  'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns']].loc[start_date:,]
        performance = performance.loc[af.default_test_start_date:,]
        performance.reset_index(inplace=True)

        filename = f"performance_data_ml_{c}.csv"
        file_path = Path(f"../data/performance/{filename}")
        performance.to_csv(file_path, index=False)
    

def create_market_data():
    market = helpers.get_stocks(['^GSPC'])
    market = market['^GSPC']
    market = market.loc[af.default_test_start_date:,]
    market['market_daily_returns'] = market['close'].pct_change()
    market.dropna(inplace=True)
    market['market_cum_returns'] = (1 + market['market_daily_returns']).cumprod() - 1
    market.reset_index(inplace = True)
    market.to_csv(Path("../data/at_market_data.csv"), index=False)
    
    
def create_ml_prediction_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    
    for c in classes:
        df = af.build_ml_prediction_data(c)
        df.drop(['performance_signal'], axis=1, inplace=True)
        df.reset_index(inplace = True)
        filename = f"ml_prediction_data_{c}.csv"
        file_path = Path(f"../data/ml_prediction_data/{filename}")
        df.to_csv(file_path, index=False)


def create_train_test():
    
    portfolios=['conservative', 'balanced','growth',
                                  'aggressive', 'alternative']
    # loop through portfolios and create datasets
    
    for port in portfolios:
        signals_df, ml_df =af. build_portfolio_signal_ml_df(f'{port}',2017,12,31)

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
        

        # # save X_train/test datasets
        X_train.to_csv(Path(f"../modeling/data/X_train_full_{port}.csv"))
        X_test.to_csv(Path(f"../modeling/data/X_test_full_{port}.csv"))


        # save y train/test datasets
        y_train.to_csv(Path(f"../modeling/data/y_train_{port}.csv"))
        y_test.to_csv(Path(f"../modeling/data/y_test_{port}.csv"))
        
        
        
        
        
        # reduce features in datasets
        X_train = X_train[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
        X_test = X_test[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]

        
        # # save X_train/test datasets with reduced features
        X_train.to_csv(Path(f"../modeling/data/X_train_reduced_{port}.csv"))
        X_test.to_csv(Path(f"../modeling/data/X_test_reduced_{port}.csv"))
        


    
def create_all_data():
    create_performance_data()
    create_market_data()
    create_ml_prediction_data()
    create_train_test()


if __name__ == "__main__":
    create_all_data()
    