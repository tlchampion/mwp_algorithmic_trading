

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# import modules
import panel as pn
pn.extension('tabulator')
import pandas as pd
import numpy as np
from panel.template import FastListTemplate
from pathlib import Path
from yahoo_fin.stock_info import get_data
import datetime
from matplotlib.figure import Figure
from matplotlib import cm
import hvplot.pandas
import holoviews as hv
from holoviews import opts



import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# import modules that help build tabs
import modules.helpers as helpers
import modules.HistoricalData as hst
import modules.MCTab as MCTab
import modules.intro as intro
import modules.profile as prf
import modules.algorithmic_functions as af


import pandas_ta as ta
import yfinance as yf

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score 

import seaborn as sns

from joblib import dump, load

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# # fit and save best model for all portfolio classes

# ### All portfolio classes found either SVC or GuassianNB to provide the best results,
# ### except for the alternative class which found the neural network to provide the best results.
# ### Both SVC and GaussianNB had the optimal performance with the reduced feature set
# 
# #### loading train/test data for reduced features and defining the models




# load X_train_reduced and X_test_reduced
X_train_reduced_conservative = pd.read_csv(Path("../modeling/data/X_train_reduced_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
X_test_reduced_conservative = pd.read_csv(Path("../modeling/data/X_test_reduced_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

X_train_reduced_balanced = pd.read_csv(Path("../modeling/data/X_train_reduced_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
X_test_reduced_balanced = pd.read_csv(Path("../modeling/data/X_test_reduced_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

X_train_reduced_growth = pd.read_csv(Path("../modeling/data/X_train_reduced_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
X_test_reduced_growth = pd.read_csv(Path("../modeling/data/X_test_reduced_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

X_train_reduced_aggressive = pd.read_csv(Path("../modeling/data/X_train_reduced_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
X_test_reduced_aggressive = pd.read_csv(Path("../modeling/data/X_test_reduced_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

X_train_reduced_alternative = pd.read_csv(Path("../modeling/data/X_train_reduced_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)
X_test_reduced_alternative = pd.read_csv(Path("../modeling/data/X_test_reduced_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True)

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


# #### Define standard scaler to use in model pipelines




scaler = StandardScaler()


# ## Aggressive
# 




pipeline = Pipeline([('scaler', scaler), ('model', svc_model)])
pipeline.fit(X_train_reduced_aggressive, y_train_aggressive)
dump(pipeline,Path("../modeling/saved_models/aggressive.joblib"))


# ## Alternative




#update to nn
# pipeline = Pipeline([('scaler', scaler), ('model', gaussian_model)])
# pipeline.fit(X_train_reduced_alternative, y_train_alternative)
# dump(pipeline,Path("./saved_models/alternative.joblib"))




# Create the scaler instance
X_scaler = StandardScaler()

# Fit the scaler
X_scaler.fit(X_train_reduced_alternative)

# Scale the data
X_train_reduced_alternative_scaled = X_scaler.transform(X_train_reduced_alternative)
X_test_reduced_alternative_scaled = X_scaler.transform(X_test_reduced_alternative)
number_input_features = 7
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 3
activation_1 = 'relu'
activation_2 = 'relu'
lr = 0.001

# Create a sequential neural network model
nn_alternative = Sequential()

# Add the first hidden layer
nn_alternative.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation=activation_1))

# Add the second hidden layer
nn_alternative.add(Dense(units=hidden_nodes_layer2, activation=activation_2))

# Add the output layer
nn_alternative.add(Dense(units=1, activation="sigmoid"))

# Compile the model 
# Set the parameters as mean_squared_error, adam, and accuracy.
nn_alternative.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])

# Fit the model
deep_net_alternative_model = nn_alternative.fit(X_train_reduced_alternative_scaled, y_train_alternative, epochs=100, verbose=0)


nn_alternative.save(Path("../modeling/saved_models/alternative.h5"))


# ## Balanced




pipeline = Pipeline([('scaler', scaler), ('model', gaussian_model)])
pipeline.fit(X_train_reduced_balanced, y_train_balanced)
dump(pipeline, Path("../modeling/saved_models/balanced.joblib"))


# ## Conservative




pipeline = Pipeline([('scaler', scaler), ('model', svc_model)])
pipeline.fit(X_train_reduced_conservative, y_train_conservative)
dump(pipeline, Path("../modeling/saved_models/conservative.joblib"))


# ## Growth




pipeline = Pipeline([('scaler', scaler), ('model', gaussian_model)])
pipeline.fit(X_train_reduced_growth, y_train_growth)
dump(pipeline, Path("../modeling/saved_models/growth.joblib"))







