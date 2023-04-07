import datetime
import pandas
import numpy
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





def create_performance_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    strategies_list = {'conservative': ['sma', 'rsi', 'macd'],
              'balanced': ['sma', 'rsi', 'macd'],
              'growth': ['sma', 'rsi', 'macd'],
              'aggressive': ['sma', 'rsi', 'macd'],
              'alternative': ['sma', 'rsi', 'macd']
             }
    for c in classes:
        start_date = af.default_test_start_date
        df, ml = af.build_portfolio_signal_ml_df(c, 2021, 6, 1)

        strategies = strategies_list[c]
        for s in strategies:
            ind = s.upper() + '_signal'
            performance = af.create_portfolio_performance_data(df, ind)
            performance = performance[['close', ind, 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                      'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns']].loc[start_date:,]
            performance.reset_index(inplace=True)
            file_name = f"performance_data_{s}_{c}.csv"
            file_path = Path(f"../data/performance/{file_name}")
            performance.to_csv(file_path, index=False)
   

def create_market_data():
    market = helpers.get_stocks(['^GSPC'])
    market = market['^GSPC']
    market['market_daily_returns'] = market['close'].pct_change()
    market.dropna(inplace=True)
    market['market_cum_returns'] = (1 + market['market_daily_returns']).cumprod() - 1
    market = market.loc[af.default_test_start_date:,]
    market.reset_index(inplace = True)
    market.to_csv(Path("../data/at_market_data.csv"), index=False)

  
if __name__ == "__main__":
    create_performance_data()
    