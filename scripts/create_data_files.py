
import questionary
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



def create_performance_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    nn_classes = ['balanced']
    strategies_list = {'conservative': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'balanced': ['sma', 'rsi', 'macd','stoch', 'bb'],
              'growth': ['sma', 'rsi', 'macd','stoch','bb'],
              'aggressive': ['sma', 'rsi', 'macd','stoch','bb'],
              'alternative': ['sma', 'rsi', 'macd','stoch','bb']
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
       
        if c not in nn_classes:
    
    
            # import model 
            filepath = Path(f"../modeling/saved_models/{c}.joblib")
            
            with open(filepath, 'rb') as file:
                model = load(file) 
            # load data to make predictions on
            file = Path(f"../data/ml_prediction_data/ml_prediction_data_{c}.csv")
            pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'index')
            preds = model.predict(pred_data)
            
        else:
            filepath = Path(f"../modeling/saved_models/{c}.h5")
            model = tf.keras.models.load_model(filepath)
            file = Path(f"../data/ml_prediction_data/ml_prediction_data_{c}.csv")
            pred_data = pd.read_csv(file, infer_datetime_format=True, parse_dates=True, index_col = 'index')
            predictions = model.predict(pred_data)
            preds = tf.cast(predictions >= 0.5, tf.int32)
        
        
        
        
        preds_df = pd.DataFrame(index=pred_data.index)
        preds_df['model_signal'] = preds



        df_ml = df_ml.loc[preds_df.index[0]:]
        df_ml = pd.concat([df_ml, preds_df], axis=1)

        performance = af.create_portfolio_performance_data(df_ml, 'model_signal', share_size=share_size)

        performance = performance[['close', 'model_signal', 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                  'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns']].loc[start_date:,]
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
    reduced = ['balanced']
    for c in classes:
        df = af.build_ml_prediction_data(c)
        df.drop(['performance_signal'], axis=1, inplace=True)
        if c in reduced:
            df = df[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
        df.reset_index(inplace = True)
        filename = f"ml_prediction_data_{c}.csv"
        file_path = Path(f"../data/ml_prediction_data/{filename}")
        df.to_csv(file_path, index=False)
        
def MC_create_ml_prediction_data():
    classes = ['conservative', 'balanced', 'growth', 'aggressive', 'alternative']
    reduced = ['balanced']
    for c in classes:
        df = af.MC_build_ml_prediction_data(c)
        df.drop(['performance_signal'], axis=1, inplace=True)
        if c in reduced:
            df = df[['SMA_200', 'EMA_50', 'BBL_20_2.0','BBM_20_2.0',
                          'BBU_20_2.0','BBB_20_2.0','BBP_20_2.0']]
        df.reset_index(inplace = True)
        filename = f"mc_ml_prediction_data_{c}.csv"
        file_path = Path(f"../MCdata/MC_ml_prediction_data/{filename}")
        df.to_csv(file_path, index=False)

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
        df, ml = af.build_portfolio_signal_ml_df(c, 2017, 12, 31)
        share_size = af.default_share_size[c]
        df_ml = df.copy()
        # df = df.loc[af.default_test_start_date:,]
        # create performance dataframes for each strategy defined for the portfolio class
        strategies = strategies_list[c]
        for s in strategies:
            ind = s.upper() + '_signal'
            
            performance = af.create_portfolio_performance_data(df, ind, share_size=share_size)
            performance = performance[['close', ind, 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                      'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns']].loc[start_date:,]
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

        performance = af.create_portfolio_performance_data(df_ml, 'model_signal', share_size=share_size)

        performance = performance[['close', 'model_signal', 'Position', 'Entry/Exit Position', 'Portfolio Holdings', 'Portfolio Cash',
                                  'Portfolio Total', 'Portfolio Daily Returns', 'Portfolio Cumulative Returns', 'Base Daily Returns', 'Base Cumulative Returns']].loc[start_date:,]
        # performance = performance.loc[af.default_test_start_date:,]
        performance.reset_index(inplace=True)
        performance.dropna(inplace=True)
        filename = f"mc_performance_data_ml_{c}.csv"
        file_path = Path(f"../MCdata/MCperformance/{filename}")
        performance.to_csv(file_path, index=False)

        
        
        
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
        training_end = X.index.min() + DateOffset(months=24)
        test_end = training_end + DateOffset(months=12)

        X_train = X.loc[training_begin:training_end]
        X_test = X.loc[training_end:test_end]
        y_train = y.loc[training_begin:training_end]
        y_test = y.loc[training_end:test_end]
        

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
    invested_amount = 100000
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
        

def save_models():
    # # fit and save best model for all portfolio classes

    # ### All portfolio classes found either SVC or GuassianNB to provide the best results,
    # ### except for the alternative class which found the neural network to provide the best results.
    # ### Both SVC and GaussianNB had the optimal performance with the reduced feature set
    # 
    # #### loading train/test data for reduced features and defining the models


    # load X_train_reduced and X_test_reduced
    X_train_full_conservative = pd.read_csv(Path("../modeling/data/X_train_full_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    X_test_full_conservative = pd.read_csv(Path("../modeling/data/X_test_full_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    X_train_reduced_balanced = pd.read_csv(Path("../modeling/data/X_train_reduced_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    X_test_reduced_balanced = pd.read_csv(Path("../modeling/data/X_test_reduced_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    X_train_full_growth = pd.read_csv(Path("../modeling/data/X_train_full_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    X_test_full_growth = pd.read_csv(Path("../modeling/data/X_test_full_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    X_train_full_aggressive = pd.read_csv(Path("../modeling/data/X_train_full_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    X_test_full_aggressive = pd.read_csv(Path("../modeling/data/X_test_full_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    X_train_full_alternative = pd.read_csv(Path("../modeling/data/X_train_full_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    X_test_full_alternative = pd.read_csv(Path("../modeling/data/X_test_full_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    #load y_train and y_test
    y_train_conservative = pd.read_csv(Path("../modeling/data/y_train_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    y_test_conservative = pd.read_csv(Path("../modeling/data/y_test_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    y_train_balanced = pd.read_csv(Path("../modeling/data/y_train_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    y_test_balanced = pd.read_csv(Path("../modeling/data/y_test_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    y_train_growth = pd.read_csv(Path("../modeling/data/y_train_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    y_test_growth = pd.read_csv(Path("../modeling/data/y_test_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    y_train_aggressive = pd.read_csv(Path("../modeling/data/y_train_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    y_test_aggressive = pd.read_csv(Path("../modeling/data/y_test_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

    y_train_alternative = pd.read_csv(Path("../modeling/data/y_train_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
    y_test_alternative = pd.read_csv(Path("../modeling/data/y_test_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)





    svc_model = SVC(random_state=42, max_iter=1000, kernel='linear', C=0.5, probability=True)
    gaussian_model = GaussianNB(var_smoothing=1.0, priors=[.3, 0.7])
    rf_model = RandomForestClassifier(n_estimators=1000, max_depth=40, min_samples_split=50, min_samples_leaf=20, max_features=None, bootstrap=True, criterion='entropy', min_impurity_decrease=0.01, class_weight={0: 1, 1: 5}, oob_score=True)


    # #### Define standard scaler to use in model pipelines




    scaler = StandardScaler()


    # ## Aggressive
    # 
    np.random.seed(8171)



    pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
    pipeline.fit(X_train_full_aggressive, y_train_aggressive)
    
    filepath = Path("../modeling/saved_models/aggressive.joblib")
    with open(filepath, 'wb') as file:
        dump(pipeline, file)


    # ## Alternative
    np.random.seed(8171)


    #update to nn
    pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
    pipeline.fit(X_train_full_alternative, y_train_alternative)
    
    filepath = Path("../modeling/saved_models/alternative.joblib")
    with open(filepath, 'wb') as file:
        dump(pipeline,file)






    # ## Balanced




#     pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
#     pipeline.fit(X_train_full_balanced, y_train_balanced)
#     dump(pipeline, Path("../modeling/saved_models/balanced.joblib"))

    
    tf.keras.utils.set_random_seed(42)
    
       # Create the scaler instance
    X_scaler = StandardScaler()

    # Fit the scaler
    X_scaler.fit(X_train_reduced_balanced)

    # Scale the data
    X_train_reduced_balanced_scaled = X_scaler.transform(X_train_reduced_balanced)
    X_test_reduced_balanced_scaled = X_scaler.transform(X_test_reduced_balanced)
    number_input_features = 7
    hidden_nodes_layer1 = 32
    hidden_nodes_layer2 = 3
    activation_1 = 'tanh'
    activation_2 = 'tanh'
    lr = 0.01

    # Create a sequential neural network model
    nn_balanced = Sequential()

    # Add the first hidden layer
    nn_balanced.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation=activation_1))

    # Add the second hidden layer
    nn_balanced.add(Dense(units=hidden_nodes_layer2, activation=activation_2))

    # Add the output layer
    nn_balanced.add(Dense(units=1, activation="sigmoid"))

    # Compile the model 
    # Set the parameters as mean_squared_error, adam, and accuracy.
    nn_balanced.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])

    # Fit the model
    deep_net_balanced_model = nn_balanced.fit(X_train_reduced_balanced_scaled, y_train_balanced, epochs=100, verbose=0)


    nn_balanced.save(Path("../modeling/saved_models/balanced.h5"))

    # ## Conservative

    np.random.seed(8171)


    pipeline = Pipeline([('scaler', scaler), ('model', svc_model)])
    pipeline.fit(X_train_full_conservative, y_train_conservative)
    
    filepath = Path("../modeling/saved_models/conservative.joblib")
    with open(filepath, 'wb') as file:
        dump(pipeline, file)



#     # Create the scaler instance
#     X_scaler = StandardScaler()

#     # Fit the scaler
#     X_scaler.fit(X_train_reduced_conservative)

#     # Scale the data
#     X_train_reduced_conservative_scaled = X_scaler.transform(X_train_reduced_conservative)
#     X_test_reduced_conservative_scaled = X_scaler.transform(X_test_reduced_conservative)
#     number_input_features = 7
#     hidden_nodes_layer1 = 36
#     hidden_nodes_layer2 = 3
#     activation_1 = 'tanh'
#     activation_2 = 'tanh'
#     lr = 0.01

#     # Create a sequential neural network model
#     nn_conservative = Sequential()

#     # Add the first hidden layer
#     nn_conservative.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation=activation_1))

#     # Add the second hidden layer
#     nn_conservative.add(Dense(units=hidden_nodes_layer2, activation=activation_2))

#     # Add the output layer
#     nn_conservative.add(Dense(units=1, activation="sigmoid"))

#     # Compile the model 
#     # Set the parameters as mean_squared_error, adam, and accuracy.
#     nn_conservative.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])

#     # Fit the model
#     deep_net_conservative_model = nn_conservative.fit(X_train_reduced_conservative_scaled, y_train_conservative, epochs=100, verbose=0)


#     nn_conservative.save(Path("../modeling/saved_models/conservative.h5"))


    # ## Growth

    np.random.seed(8171)


    pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
    pipeline.fit(X_train_full_growth, y_train_growth)
    
    filepath = Path("../modeling/saved_models/growth.joblib")
    with open(filepath, 'wb') as file:
        dump(pipeline, file)






   

    

# function to create all needed files for application functioning
def create_all_data():
    print("creating ML prediction data\n")
    create_ml_prediction_data()
    print("creating S&P 500 historical data\n")
    create_market_data()
    print("creating train/test datasets for ML modeling\n")
    create_train_test()
    print("create and save models")
    save_models()
    print("creating strategy performance data\n")
    create_performance_data()
    print("creating MC prediction data\n")
    MC_create_ml_prediction_data()
    print("creating MC performance data\n")
    MC_create_performance_data()
    print("creating MC graphs\n")
    create_mc_info()

def create_data_only():
    # print("creating ML prediction data\n")
    # create_ml_prediction_data()
    # print("creating S&P 500 historical data\n")
    # create_market_data()
    # print("creating train/test datasets for ML modeling\n")
    # create_train_test()
    # print("creating strategy performance data\n")
    # create_performance_data()
    # print("creating MC prediction data\n")
    # MC_create_ml_prediction_data()
    # print("creating MC performance data\n")
    # MC_create_performance_data()
    print("creating MC graphs\n")
    create_mc_info()


def make_selection():

    action = questionary.select(
        "Please select one of the following options:",
        choices = ["Generate all data files and saved ML models.",
        "Generate all data files but not the saved ML models."]).ask()
    if action == "Generate all data files and saved ML models.":
        print("Creating all data files and saved ML models.\n This may take some time so please be patient...\n\n")
        create_all_data()
    elif action == "Generate all data files but not the saved ML models.":
        print("Creating all data files.\n This may take some time so please be patient...\n\n")
        create_data_only()
    else:
        print("Not a valid option. Please retry again.")



if __name__ == "__main__":
    # create_all_data()
    make_selection()
    # MC_create_performance_data()
    