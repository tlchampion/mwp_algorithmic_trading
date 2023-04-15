# Directory Contents

This folder contains data files used to prepare the displays on the Algorithmic Trading tab.

The 'historical' folder holds one file per portfolio containing the daily performance information for that portfolio

The 'ml_prediciton_data' contains a set of datasets used for predicting the buy/sell signals for the machine learning trading stratgy



# Data Collection and Processing

The following process was completed once for each of the five portfolio classes:



**1. Data Collection and Initial Processing**

* Used yahoo finance API to pull historical data on stocks in portfolio from 12/31/2017 forward.
* Combined Open/Close/High/Low/Volume data for individual stocks into one ```OCHLV dataset```, scaling each stock's contribution based on its weight in the portfolio.
* Added a ```performance signal``` column indicating buy or sell condition for each day based on closing price. If closing price saw a negative change a sell signal was generated. A positive change triggered a buy signal.
* Added a collection of indicators using ```pandas-ta``` library: SMA (30, 100, 200), EMA (50), MACD (12/26/9), BBANDS (20/2.0), RSI (14), HLC3, OHLC4, LINREG (14), and STOCH (14/3/3).
* Daily closing price of S&P 500 was pulled using the same API and concatenated onto the dataframe for use in performance comparisons.


At this point the process was dependent on if the strategy buy/sell signal was being trriggered by a markent indicator was or by a machine learning model.


**2a. Market Indicator Strategy Pathway**

* Converted indicators into Buy/Sell signals based on the following designed strategies: 
<br>


    | Indicator |       Buy       |      Sell      |
    |:---------:|:---------------:|:--------------:|
    | SMA       | SMA30 >= SMA100 | SMA30 < SMA100 |
    | MACD      | MACD >= MACDs   | MACD < MACDs   |
    | BBANDS    | close  <= BBL   | close  > BBU   |
    | RSI       | RSI <= 30       | RSI >= 70      |
    | STOCH     | STOCHk < 20     | STOCHk > 80    |

* A 'buy' signal has a value of 1 while a 'sell' signal has a value of 0
* Continued with Step 3 below


**2b. Machine Learning Strategy Pathway**


* Dropped unnecessary columns from data, retaining performance signal and indicator values only (not the buy/sell signals from the indicators).
* Created ```feature dataset``` by dropping the buy/sell performance indicator.
* Shifted ```feature dataset``` by one time period so that today's performance is used to determine tomorrow's buy/sell action.
* Created ```target dataset``` using the buy/sell performance indicator.
* Created ```train/test datasets``` . The first 24 months of data were used for the training dataset, the folloiwing 12 months for the test dataset. The remainder was reserved for making model-based predictions.
* Created two ```final X (feature) datasets``` with different contents: full set of 18 features and reduced set with only SMA200, EMA50, and the five Bollinger Band indicators.
* Saved train/test datasets for use in modeling process
* The optimal model per portfolio was used to make predictions using the final third of the original dataset
* Model predictions were combined onto a copy of the portfolios historical price data from step 1 above. The predictions became the Buy/Sell signal ('buy' = 1, 'sell' = 0)
* Continued with Step 3 below


**3. Investment Strategy Perfomance Data**

Performance backtesting for the investment stratgies followed the following steps:
* Each portfolio has a "share size" assigned based upon the value of one share. Portfolios with a higher value/cost have a lower share size. This allows ongoing buy/sell actions to remain below the default investment amount of $150,000 for demonstration purposes. varied between portfolio classes. 
* Added a ```Position``` column indicating the number of shares owned in the portfolio for each day. This is equal to the share size (X) plus the share size times the buy/sell signal. This gives a value of either X (if in a sell condition)) or 2X (if in a buy condition)
* Added an ```Entry/Exit Position``` column as the difference between current and previous position values, except for the first row which was equal to the position value.
* Added ```Portfolio Holdings``` as the product of number of shares held and closing price.
* Added ```Portfolio Cash``` as the initial capital invested minus cash used to buy shares plus proceeds of selling shares.
* Added ```Portfolio Total``` as the sum of 'Portfolio Holdings' and 'Portfolio Cash'.
* Calculated ```Portfolio Daily Returns``` as the percentage change over the previous day.
* Calculated ```Portfolio Cumulative Returns``` as the cumulative product of 'Portfolio Daily Returns'.
* Added ```Daily Returns``` and ```Cumulative Returns``` based on portfolio closing price to reflect performance without using buy/sell signals.
* Added ```Market Daily Returns``` and ```Market Cumulative Returns``` based upon daily closing price of S&P 500 previously added to dataframe
* Saved compiled data for use in dashboard application.






