# Data Collection and Trading Signals Creation

The following process was completed once for each of the five portfolio classes:

* Historical **Data collection** (5y, from Yahoo Finance API)
* Creation of a range of **trading indicators** (SMA, EMA, MACD, BBANDS, RSI, STOCH) to build the Trading Models.
* Creation of **trading models** with their specific buy/sell signals.
* **Signal Performance Pathway**: Backtesting of performance of the Portfolio vs the Portfolio enhanced by the trading strategy of choice and vs a Market Benchmark (S&P500).
* **Machine Learning Pathway**: Training and testing of all the Portfolios with 6 different Model algorithms and 11 variations within each Model. Selection of the best performing one for each Portfolio to show the client.

The trading strategy of choice and the predictions by the best performing Model will be displayed for the client in a new tab (**"Algorithmic Trading"**) on MyWealthPath's Platform.

---

## Analysis Detail

**1. Data Collection**

* Used 5yfinance API to pull historical data on stocks for each of the five portfolio classes from 12/31/2017 forward.
* Combined Open/Close/High/Low/Volume data for individual stocks into one ```OCHLV dataset```, scaling each stock's contribution based on its weight in the portfolio.
* Added a ```performance signal``` column indicating buy or sell condition for each day based on closing price.
* Added various indicators using ```python-ta library```: SMA (30, 100, 200), EMA (50), MACD (12/26/9), BBANDS (20/2.0), RSI (14), HLC3, OHLC4, LINREG (14), and STOCH (14/3/3).
* Converted indicators into Buy/Sell signals based on the following designed strategies:

| Indicator |       Buy       |      Sell      |
|:---------:|:---------------:|:--------------:|
| SMA       | SMA30 >= SMA100 | SMA30 < SMA100 |
| MACD      | MACD >= MACDs   | MACD < MACDs   |
| BBANDS    | close  <= BBL   | close  > BBU   |
| RSI       | RSI <= 30       | RSI >= 70      |
| STOCH     | STOCHk < 20     | STOCHk > 80    |


**2. Signal Performance Pathway**

* Added a ```Position``` column indicating the number of shares owned in the portfolio, determined by share size times the strategy signal (share size varied between portfolio classes).
* Added an ```Entry/Exit Position``` column as the difference between current and previous position values, except for the first row.
* Added ```Portfolio Holdings``` as the product of number of shares held and closing price.
* Added ```Portfolio Cash``` as the initial capital invested minus cash used to buy shares plus proceeds of selling shares.
* Added ```Portfolio Total``` as the sum of 'Portfolio Holdings' and 'Portfolio Cash'.
* Calculated ```Portfolio Daily Returns``` as the percentage change over the previous day.
* Calculated ```Portfolio Cumulative Returns``` as the cumulative product of 'Portfolio Daily Returns'.
* Added ```Daily Returns``` and ```Cumulative Returns``` based on portfolio closing price to reflect performance without using buy/sell signals.
* Saved compiled data for use in dashboard application.


**3. Machine Learning Pathway**

* Dropped unnecessary columns from data, retaining performance signal and indicators.
* Created ```feature dataset``` by dropping the buy/sell performance indicator.
* Shifted ```feature dataset``` by one time period so that today's performance is used to determine tomorrow's buy/sell action.
* Created ```target dataset``` using the buy/sell performance indicator.
* Created ```train/test datasets``` by splitting data based on a 36-month window, reserving early dates for Train and late dates for Test datasets.
* Created two ```final X (feature) datasets``` with different contents: full set of features and reduced set with only SMA200, EMA50, and the five Bollinger Band indicators.
* Saved train/test datasets for use in modeling process.

**4. Dashboard Performance Data**

(To be displayed to the User)

* Generate prediction datasets for each portfolio class (ie. Conservative, Growth, etc) by following the same steps as for model training data, using a beginning date of January 10, 2022.
* Make predictions for each portfolio class using the saved model for that class (the best performing model for each class).
* Concatenate/combine predictions with portfolio's performance data, which was compiled earlier in the process as stated above.
* Process and plot the combined dataframe for the User.

---

## Contributors

[Ahmad Takatkah](https://github.com/vcpreneur)
[Lourdes Dominguez Bengoa](https://github.com/LourdesDB)
[Patricio Gomez](https://github.com/patogogo)
[Lovedeep Singh](https://github.com/LovedeepSingh89)
[Thomas L. Champion](https://github.com/tlchampion)

---

## License

License information can be found in the included LICENSE file.

---
## Credits
* Code for generating the Monte Carlo Simulation was modified from code provided by UC Berkeley Extension FinTech Bootcamp

---

## Disclaimer

The information provided through this application is for information and educational purposes only. 
It is not intended to be, nor should it be used as, investment advice. 
Seek a duly licensed professional for investment advice.


