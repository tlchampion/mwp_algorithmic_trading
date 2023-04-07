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
        df, ml = af.build_portfolio_signal_ml_df(c, 2021, 6, 1)
        share_size = af.default_share_size[c]

        strategies = strategies_list[c]
        for s in strategies:
            ind = s.upper() + '_signal'
            performance = af.create_portfolio_performance_data(df, ind, share_size=share_size)
            performance = performance[['close', ind, 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                      'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns']].loc[start_date:,]
            performance.reset_index(inplace=True)
            file_name = f"performance_data_{s}_{c}.csv"
            file_path = Path(f"../data/performance/{file_name}")
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

    
def create_all_data():
    create_performance_data()
    create_market_data()


if __name__ == "__main__":
    create_all_data()
    