


# import modules

import pandas as pd
import numpy as np

from pathlib import Path

import datetime




import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)






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


# # fit and save best model for all portfolio classes based upon final evaluation. Evaluations can be found within each models individual notebook and within the 'metric_comparisons' notebook

# ### All portfolio classes found that Random Forest to provide the best results, except for the balanced class which found the neural network to provide the best results.
# ### Random Forest had the optimal performance with the full feature set while the neural network worked best with the reduced feature set
# 
# #### loading train/test data for reduced or full features depending on model being used and defining Random Forest model. 




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
y_train_conservative = pd.read_csv(Path("../modeling/data/y_train_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()
y_test_conservative = pd.read_csv(Path("../modeling/data/y_test_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()

y_train_balanced = pd.read_csv(Path("../modeling/data/y_train_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()
y_test_balanced = pd.read_csv(Path("../modeling/data/y_test_balanced.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()

y_train_growth = pd.read_csv(Path("../modeling/data/y_train_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()
y_test_growth = pd.read_csv(Path("../modeling/data/y_test_growth.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()

y_train_aggressive = pd.read_csv(Path("../modeling/data/y_train_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()
y_test_aggressive = pd.read_csv(Path("../modeling/data/y_test_aggressive.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()

y_train_alternative = pd.read_csv(Path("../modeling/data/y_train_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()
y_test_alternative = pd.read_csv(Path("../modeling/data/y_test_alternative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()





# y_train_conservative = pd.read_csv(Path("./data/y_train_conservative.csv"), index_col="Unnamed: 0", parse_dates=True, infer_datetime_format=True).values.ravel()





y_train_conservative





rf_model = RandomForestClassifier(n_estimators=1000,
                                  max_depth=40,
                                  min_samples_split=50,
                                  min_samples_leaf=20,
                                  max_features=None,
                                  bootstrap=True,
                                  criterion='entropy', 
                                  min_impurity_decrease=0.01,
                                  class_weight={0: 1, 1: 5},
                                  oob_score=True)








# #### Define standard scaler to use in model pipelines




scaler = StandardScaler()


# ## training best model for each portfolio class and saving model for future use

# ### Aggressive
# 




np.random.seed(8171)


pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
pipeline.fit(X_train_full_aggressive, y_train_aggressive)

filepath = Path("../modeling/saved_models/aggressive.joblib")
with open(filepath, 'wb') as file:
    dump(pipeline, file)


# ### Alternative




np.random.seed(8171)

pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
pipeline.fit(X_train_full_alternative, y_train_alternative)

filepath = Path("../modeling/saved_models/alternative.joblib")
with open(filepath, 'wb') as file:
    dump(pipeline,file)


# ### Balanced




# pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
# pipeline.fit(X_train_full_balanced, y_train_balanced)
# dump(pipeline, Path("./saved_models/balanced.joblib"))
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




# ### Conservative




np.random.seed(8171)

pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
pipeline.fit(X_train_full_conservative, y_train_conservative)

filepath = Path("../modeling/saved_models/conservative.joblib")
with open(filepath, 'wb') as file:
    dump(pipeline, file)



# ### Growth



np.random.seed(8171)


pipeline = Pipeline([('scaler', scaler), ('model', rf_model)])
pipeline.fit(X_train_full_growth, y_train_growth)

filepath = Path("../modeling/saved_models/growth.joblib")
with open(filepath, 'wb') as file:
    dump(pipeline, file)






