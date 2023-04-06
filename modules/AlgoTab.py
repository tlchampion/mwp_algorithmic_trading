"""
The contents of this file define what is displayed on the 'Algorithmic Trading' tab
"""

strategies_by_portfolio = {"conservative": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi']],
                           "balanced": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi']],
                           "growth": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi']],
                           "aggressive": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi']],
                           "alternative": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi']]
                          }


strategies_info = {"sma": ['SMA_signal',"This is a description of SMA"],
                  "macd": ['MACD_signal', "MACD is designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price. It is a collection of three time series that are calculated from a stock's historical closing price based upon Exponential Moving Averages. Our MACD calculations are based upon the standard 12/26/9 day periods."],
                  "rsi": ['RSI_14',"The RSI is used to chart the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period, in this case a 14-day timeframe."]
                                    
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
    return info[0], info[1], info[2]
    
    
    
"MACD is designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price. It is a collection of three time series that are calculated from a stock's historical closing price based upon Exponential Moving Averages. Our MACD calculations are based upon the standard 12/26/9 day periods."