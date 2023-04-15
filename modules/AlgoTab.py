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

# setup the values available in the menu dropdown for each portfolio

strategies_by_portfolio = {"conservative": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                            ['Stochastic Oscillator', 'stoch'],
                                            ['Bollinger Bands', 'bb'],
                                           ['Machine Learning Model', 'ml']],
                           "balanced": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                        ['Stochastic Oscillator', 'stoch'],
                                            ['Bollinger Bands', 'bb'],
                                           ['Machine Learning Model', 'ml']],
                           "growth": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                      ['Stochastic Oscillator', 'stoch'],
                                            ['Bollinger Bands', 'bb'],
                                           ['Machine Learning Model', 'ml']],
                           "aggressive": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                          ['Stochastic Oscillator', 'stoch'],
                                            ['Bollinger Bands', 'bb'],
                                           ['Machine Learning Model', 'ml']],
                           "alternative": [['Simple Moving Average (SMA)','sma'],
                                            ['Moving Average Convergance/Divergence', 'macd'],
                                            ['Relative Strength Index (RSI)', 'rsi'],
                                           ['Stochastic Oscillator', 'stoch'],
                                            ['Bollinger Bands', 'bb'],
                                           ['Machine Learning Model', 'ml']]
                          }

# define the descriptions for the strategies

strategies_info = {"sma": ['SMA_signal',"""<p>The Simple Moving Average (SMAx) is the average of a security's price over x days.</p>

<p>Our SMA strategy relies on a short and long SMA, where the number of days used to calculate the short SMA is less than that used to calculate the long SMA. If the value for the short SMA becomes greater than the value for the long SMA it increases a 'buy' signal. When the value for the short SMA becomes less than that for the long SMA it generates a 'sell' signal.</p>"""],
                  "macd": ['MACD_signal', """<p>The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMA). EMA is a type of moving average that gives more weight to recent data points compared to Simple Moving Average (SMA), which gives equal weight to all data points in the specified period. </p>
<p>
For our strategy, we calculate a 12-day and 26-day EMA. The difference between these two values is taken, which proves the MACD value. 
</p>
<p>
Next, a 9-day EMA is taken of the MACD itself, which is referred to as the Moving Average Convergence Divergence Signal (MACDs)
</p>
<p>
Our MACD strategy generates a 'buy' signal if the MACD value is greater than or equal to the MACDs value. If the MACD falls below the MACDs a 'sell' signal is generated.</p>"""],
                  "rsi": ['RSI_signal', """<p>Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100 and is calculated as the average gain of up periods divided by the average loss of down periods over a specified period of time, in our case 14 periods. It is used to identify overbought or oversold conditions in an asset's price, indicating potential trend reversals or trend continuation.</p>
<p>
 Our strategy generates a 'buy' signal if the RSI is equal to or less than 30. A 'sell' signal is generated if the RSI is equal to or greater than 70.</p>"""],
                   "stoch": ['STOCH_signal', """<p>The Stochastic Oscillator (STOCH) is a momentum indicator that shows the location of a security's close price relative to its high-low range over a certain period of time. STOCHk represents the stochastic oscillator's %K line, which represents the current closing price in relation to the highest and lowest prices over a specified period of time, in our case 14 days. 
</p>
<p>
Our STOCH strategy generates a 'buy' signal when the STOCHk value is less than 20. A 'sell' signal is generated when the STOCHk value is greater than 80.</p>"""],
                   "bb": ["BB_signal",  """<p>Bollinger Bands (BB) consist of three lines - a middle line (BBM) which is the Simple Moving Average (SMA), and an upper (BBU) and lower (BBL) band that are calculated based on the standard deviation of the prices from the SMA. The factor for multiplying is typically set to 2, which means that the upper band is 2 times the standard deviation added to the SMA, and the lower band is 2 times the standard deviation subtracted from the SMA. Bollinger Bands are used to identify potential price volatility and trading opportunities.
</p>
<p>Our BB strategy generates a 'buy' signal if the closing price of the security is equal to or less than the value of the BBL. A 'sell' signal is generated if the closing price is equal to or greater than the value of the BBU.</p>"""],
                   "ml": ['ML_signal', """<p>Rather than relying on one performance-based indicator, our Machine Learning Strategy relies on the values of multiple indicators, such as and SMA, EMA and Bollinger Bands. The specific machine learning model varies depending on the portfolio you are invested in, however the fundamentals are the same for all models.</p>
<p>
The model, provided with an optimal set of indicators and information on the securities performance over a period of time, finds patterns and relationships between the indicators in relation to the securites performace. </p>
<p>
This learned information is then used by the model to make predictions about future performance based upon current values of the same indicators. This then provides us with our 'buy' or 'sell' signals.</p>"""]
                                    
                          }



# define introduction/instructions for tab
def get_intro():
    text = """
<p><h1>Welcome to MyWealthPath Portfolio Alogorithmic Trading</h1></p>

<p>In addition to a &lsquo;Buy and Hold&rsquo; strategy, MyWealthPath offers an active portfolio investment strategy that monitors daily market activity and places buy/sell orders based upon your current position in the portfolio and the current market trend.</p>
<h2>Our Strategy</h2>
<p> Our active investment strategy requires you to make an initial investment into your chosen portfolio which will become your base investment. In addition, we require you to fund a cash reserve account. The funds from the cash reserve account will be used to purchase additional shares of your portfolio and will recieve any funds from the selling of shares.</p>

<p> When our strategy signals a 'buy' order for your portfolio, a portion of your cash reserves will be used to buy a predetermined number (X) of additional shares as long as your last action was not a 'buy'.</p>

<p>If a 'sell' signal is recieved, a predetermined number (X) of shares will be sold as long as your last action was not a 'sell'.</p>

<p>If your last action matches the current signal generated, no action will be taken.</p>

<p>Effectively, you will always maintain a base investment position in your portfolio, but your amount invested may be temporarily increased at times based upon market indicators. This increase is in addition to any specific increases you request, which would impact your base investment level.</p>
<br>
<p>You may use the dropdown menu below to select from multiple trading strategies in order to compare their individual performance over time. Performance is modeled using recent historical data. Past performance is not indicative of future performance and results seen after investing and selecting any given strategy may not conform to the performance shown here.</p>

"""
    return text


# return details related to the trading strategies available based upon the portfolio class
def get_strategy_options(risk):
    
    info = strategies_by_portfolio[risk]
    return info[0], info[1], info[2], info[3], info[4], info[5]
    
    
# return the strategy description for display   
def get_strategies_info(strategy):
    return strategies_info[strategy][0], strategies_info[strategy][1]


# call functions to create comparison chart, portfolo value graph and return ROI for
# investment strategy, portfolio without the strategy and S&P 500
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

# graph comparison of strategy investment, portfolio performance with out
# applying the strategy and the S&P 500
def make_comparison_graph(portfolio_class, df):
    # market_data = pd.read_csv(Path(f"./data/at_market_data.csv"),
    #              index_col='index', parse_dates=True, infer_datetime_format=True)
    text = f"{portfolio_class.capitalize()} Portfolio (Strategy)"
    text2 = f"{portfolio_class.capitalize()} Portfolio (No Strategy)"
    title = f"{portfolio_class.capitalize()} Portfolio Cumulative Returns vs S&P 500"
    fig0 = Figure(figsize=(16,8))
    ax = fig0.subplots()
    #ax = port_cum_returns.plot(figsize=(10,5), title="Cumulative Returns of Conservative Portfolio vs S&P 500")
    #gmarket_cum_returns.plot(ax=ax)
    chart = ax.plot(df['Portfolio Cumulative Returns'])
    # ax.plot(market_data['market_cum_returns'])
    ax.plot(df['Market Cumulative Returns'])
    ax.plot(df['Base Cumulative Returns'])
    ax.set_title(title)
    ax.legend([text,
         'S&P',
              text2])
    
    return fig0


# calculate ROI for portfolio
def calculate_roi(df): 
    # market_data = pd.read_csv(Path(f"./data/at_market_data.csv"),
    #              index_col='index', parse_dates=True, infer_datetime_format=True)
    # roi_strategy = (data.iloc[-1,:]['Portfolio Total'] - initial_investment) / initial_investment * 100
    roi_strategy = df.iloc[-1,]['Portfolio Cumulative Returns'] * 100
    roi_nostrategy = df.iloc[-1,]['Base Cumulative Returns'] * 100
    roi_market = df.iloc[-1,]['Market Cumulative Returns'] * 100
    return [roi_strategy, roi_nostrategy, roi_market]
    
    
