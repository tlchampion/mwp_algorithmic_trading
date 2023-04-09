import pandas as pd
from matplotlib.figure import Figure
from matplotlib import cm
from pathlib import Path
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
import modules.algorithmic_functions as af

"""
The contents of this file define what is displayed on the 'Algorithmic Trading' tab
"""

strategies_by_portfolio = {"conservative": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                           ['Machine Learning Model', 'ml']],
                           "balanced": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                           ['Machine Learning Model', 'ml']],
                           "growth": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                           ['Machine Learning Model', 'ml']],
                           "aggressive": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                           ['Machine Learning Model', 'ml']],
                           "alternative": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                           ['Machine Learning Model', 'ml']]
                          }


strategies_info = {"sma": ['SMA_signal',"This is a description of SMA"],
                  "macd": ['MACD_signal', "MACD is designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price. It is a collection of three time series that are calculated from a stock's historical closing price based upon Exponential Moving Averages. Our MACD calculations are based upon the standard 12/26/9 day periods."],
                  "rsi": ['RSI_14',"The RSI is used to chart the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period, in this case a 14-day timeframe."],
                   "ml": ['ML_signal', "This is a descripiton of the ML model used"]
                                    
                          }



# define introduction/instructions for tab
def get_intro():
    text = """
    ## Welcome to the MyWealthPlan Portfolio Alogorithmic Trading page.
    
    ### In addition to a 'Buy and Hold' strategy, MyWealthPlan offers a managed portfolio strategy that monitors daily market activity and places buy/sell orders based upon your current position in the portfolio and the current market trend.

    ### You may use the dropdown menu below to select from multiple trading strategies in order to compare their individual performance over time. Performance is modeled using recent historical data. Past performance is not indicative of future performance and results seen after investing and selecting any given strategy may not conform to the performance shown here. 

"""
    return text

# return details related to the trading strategies available based upon the portfolio class
def get_strategy_options(risk):
    
    info = strategies_by_portfolio[risk]
    return info[0], info[1], info[2], info[3]
    
    
    
def get_strategies_info(strategy):
    return strategies_info[strategy][0], strategies_info[strategy][1]

def get_performance_data(portfolio_class, strategy):
    file = f"performance_data_{strategy}_{portfolio_class}.csv"
    df = pd.read_csv(Path(f"./data/performance/{file}"),
                 index_col='index', parse_dates=True, infer_datetime_format=True)
    
    
    figure = make_performance_graph(portfolio_class, df)
    roi = calculate_roi(df)
    compare = make_comparison_graph(portfolio_class, df)
    
    return figure, roi, compare

# creat graph showing total portfolio value over time
def make_performance_graph(portfolio_class,df):

    
    text = f"{portfolio_class.title()} Portfolio (with strategy)"
    text2 = f"{portfolio_class.title()} Portfolio (without strategy)"
    title = f"{portfolio_class.title()} Value over Time"
    fig0 = Figure(figsize=(16,8))
    ax = fig0.subplots()
    chart = ax.plot(df['Portfolio Total'])

    ax.set_title(title)
    ax.legend([text])

    return fig0


def make_comparison_graph(portfolio_class, df):
    market_data = pd.read_csv(Path(f"./data/at_market_data.csv"),
                 index_col='index', parse_dates=True, infer_datetime_format=True)
    text = f"{portfolio_class.capitalize()} Portfolio (Strategy)"
    text2 = f"{portfolio_class.capitalize()} Portfolio (No Strategy)"
    title = f"{portfolio_class.capitalize()} Portfolio Cumulative Returns vs S&P 500"
    fig0 = Figure(figsize=(16,8))
    ax = fig0.subplots()
    #ax = port_cum_returns.plot(figsize=(10,5), title="Cumulative Returns of Conservative Portfolio vs S&P 500")
    #gmarket_cum_returns.plot(ax=ax)
    chart = ax.plot(df['Portfolio Cumulative Returns'])
    ax.plot(market_data['market_cum_returns'])
    ax.plot(df['Base Cumulative Returns'])
    ax.set_title(title)
    ax.legend([text,
         'S&P',
              text2])
    
    return fig0


# calculate ROI for portfolio
def calculate_roi(df): 
    market_data = pd.read_csv(Path(f"./data/at_market_data.csv"),
                 index_col='index', parse_dates=True, infer_datetime_format=True)
    # roi_strategy = (data.iloc[-1,:]['Portfolio Total'] - initial_investment) / initial_investment * 100
    roi_strategy = df.iloc[-1,]['Portfolio Cumulative Returns'] * 100
    roi_nostrategy = df.iloc[-1,]['Base Cumulative Returns'] * 100
    roi_market = market_data.iloc[-1,]['market_cum_returns'] * 100
    return [roi_strategy, roi_nostrategy, roi_market]
    
    
