# Modeling

The following process has been followed to test and select the most suitable Models for each Portfolio class:



**1. Data Loading**

Prepared Train/Test datasets were loaded from saved files. Please see the [main section]() for details on the preparation of the datasets.



**2. Model Testing**

Machine Learning models were built using seven different algorithms from the Scikit-Learn library:
* Bagging Classifier
* GaussianNB
* Logistic Regression
* Random Forest
* SVM Classifier
* AdaBoost Classifier
* LightGBM

As the initial step in the building of all models a ```StandardScaler``` was instantiated. 

Each model class had a minimum of 10-plus models built using different parameters. Please see below for a chart depicting the parameters used for each model:

|  **Bagging Classifier**                                                                                                                                              | 
| Model|  Base Estimator |  m_estimators |  max_samples |  max_features |  bootstrap |  oob_score |  random_state
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| model1| base_classifier | 200 | 0.8 | 0.5 | True | True | 42 |
| model2| base_classifier | 50 | 0.8 | 0.5 | True | True | 42 |
| model3| base_classifier | 100 | 0.8 | 0.5 | True | True | 42 |
| model4| base_classifier | 100 | 0.9 | 0.5 | True | True | 42 |
| model5| base_classifier | 100 | 0.5 | 0.5 | True | True | 42 |
| model6| base_classifier | 100 | 0.8 | 0.7 | True | True | 42 |
| model7| base_classifier | 100 | 0.8 | 0.3 | True | True | 42 |
| model8| base_classifier | 100 | 0.8 | 0.5 | False | False | 42 | 
| model9| base_classifier | 100 | 0.8 | 0.5 | True | True | 123 | 
| model10| base_classifier | 100 | 0.8 | 0.5 | True | False | 123| 
| model11| base_classifier | 100 | 0.8 | 0.5 | True | False | 42 | 

| **GaussianNB**                                              |
| Model | var_smoothing | priors |
|-------------------------------------------------------------|
| model1 | | None |
| model2 |  1e-9 | None |
| model3 |  1e-5 | [0.2, 0.8] |
| model4 |  1e-3 | [0.5, 0.5] |
| model5 |  1e-1 | None |
| model6 |  1.0 | [.3, 0.7] |
| model7 |  10.0 | [0.6, 0.4] |
| model8 |  1e-9 | [0.4, 0.6] |
| model9 |  1e-5 | None |
| model10 |  1e-3 | [0.2, 0.8] |
| model11 |  1e-1 | [0.5, 0.5} |


| **Logistic Regression** |
|model | random_state | max_iter | solver | penatly | l1_ratio |
|-----------------------------------------------------------------------------------------------------------------|
| model1 |42 | 10000 | saga | elasticnet | 0.1 |
| model2 |42 | 10000 | saga | elasticnet | 0.3 |
| model3 |42 | 10000 | saga | elasticnet | 0.5 |
| model4 |42 | 10000 | saga | elasticnet | 0.7 |
| model5 |42 | 10000 | saga | elasticnet | 0.9 |
| model6 |42 | 10000 | saga | l1 |
| model7 |42 | 10000 | saga | None |
| model8 |42 | 10000 | lbfgs | None |
| model9 |42 | 10000 | liblinear | l1 |
| model10 |42 | 10000 | liblinear | l2 |
| model11 |42 | 10000 | sag | None |

| **Random Forest**                                                       |
| model | n_estimators | max_depth | min_samples_split | max_features | boostrap | criterion | min_impurity_decrease | class_weight | oob_score |
|--------------------------------------------------------------------------------------------------------------------------------------------------|
| model1 | 100 | 10 | 5 | 1 | sqrt | True | gini | 0.0 | None | False |
| model2 | 200 | 20 | 10 | 5 | log2 | True | entropy | 0.001 | balanced_subsample | True |
| model3 | 500 | 30 | 20 | 10 | 0.5 | True | gini | 0.005 | {0: 1, 1: 3} | True |
| model4 | 1000 | 40 | 50 | 20 | None | True | entropy | 0.01 | {0: 1, 1: 5} | True |
| model5 | 200 | 20 | 10 | 5 | 0.7 | True | gini | 0.0 | None | False |
| model6 | 500 | 30 | 20 | 10 | 0.3 | True | entropy | 0.0 | balanced | True|
| model7 | 1000 | 40 | 50 | 20 | sqrt | True | gini | 0.0 | {0: 1, 1: 10} | True |
| model8 | 2000 | 50 | 100 | 50 | log2 | True | entropy | 0.0 | None | False |
| model9 | 1000 | 30 | 20 | 10 | None | True | gini | 0.005 | balanced | True |
| model10 | 500 | 20 | 10 | 5 | 0.7 | True | entropy | 0.001 | {0: 1, 1: 5} | True |
| model11 | 1000 | 30 | 10 | 5 | 0.5 | True | entropy | 0.001 | balanced | True |

| **SVM**                                                                                |
| model | random_state | max_iter | kernel | C | probability |
|----------------------------------------------------------------------------------------|
| model1 | 42 | 1000 | linear | 0.5 | True |
| model2 | 42 | 1000 | linear | 1 | True |
| model3 | 42 | 1000 | linear | 10 | True |
| model4 | 42 | 1000 | rbf | 0.5 | True |
| model5 | 42 | 1000 | rbf | 1 | True |
| model6 | 42 | 1000 | rbf | 10 | True |
| model7 | 42 | 1000 | sigmoid | 0.5 | True |
| model8 | 42 | 1000 | sigmoid | 1 | True |
| model9 | 42 | 1000 | sigmoid | 10 | True|


| **AdaBoost Classifier** |
| model | base_estimator | n_estimators | learning_rate | algorithm |
|-----------------------------------------------------------------------------------------------------------------------------|
| model1 | DecisionTreeClassifier(max_depth=1) | 100 | 1.0 | SAMME |
| model2 | LogisticRegression(solver='lbfgs' | 50 | 0.5 | SAMME |
| model3 | SVC(kernel='linear' | 200 | 0.1 | SAMME |
| model4 | RandomForestClassifier(n_estimators=50) | 100 | 1.0 | SAMME |
| model5 | GradientBoostingClassifier(max_depth=3) | 150 | 0.2 | SAMME |
| model6 | DecisionTreeClassifier(max_depth=5) | 100 | 1.0 | SAMME |
| model7 | LogisticRegression(solver='lbfgs') | 50 | 0.25 | SAMME |
| model8 | DecisionTreeClassifier(max_depth=3) | 200 | 0.01 | SAMME |
| model9 | RandomForestClassifier(n_estimators=50) | 100 | 0.5 | SAMME |
| model10 | GradientBoostingClassifier(max_depth=10) | 150 | 0.1 | SAMME |
| model11 | LinearSVC(max_iter=10000) | 150 | 0.05 | SAMME |


| **LoghtGBM**                                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------|
| model1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)   |
| model2 = lgb.LGBMClassifier(boosting_type='dart', num_leaves=63, max_depth=5, learning_rate=0.01, n_estimators=50)    |
| model3 = lgb.LGBMClassifier(boosting_type='goss', num_leaves=15, max_depth=10, learning_rate=0.05, n_estimators=200)  |
| model4 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=127, max_depth=3, learning_rate=0.5, n_estimators=150)   |
| model5 = lgb.LGBMClassifier(boosting_type='dart', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=50)    |
| model6 = lgb.LGBMClassifier(boosting_type='goss', num_leaves=63, max_depth=5, learning_rate=0.01, n_estimators=100)   |
| model7 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, max_depth=10, learning_rate=0.05, n_estimators=200)  |
| model8 = lgb.LGBMClassifier(boosting_type='dart', num_leaves=127, max_depth=3, learning_rate=0.5, n_estimators=150)   |
| model9 = lgb.LGBMClassifier(boosting_type='goss', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=50)    |
| model10 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=63, max_depth=5, learning_rate=0.01, n_estimators=100)  |
| model11 = lgb.LGBMClassifier(boosting_type='dart', num_leaves=15, max_depth=10, learning_rate=0.05, n_estimators=200) |



For each ML algorithm a loop structure was used to build/evaluate each of the 10-plus models using the following steps, once for the full feature datasets and once for the reduced features datasets:

* A pipeline was instantiated using ```SciKit-Learn Pipeline``` consisting of the ```StandardScaler``` and the model
* The ```X_train``` data was fit to the pipeline
* The pipeline was used to predict the ```X_test``` data
* A classification report using ```sklearn.metrics.classification_report``` was generated
* The classification reports for the full feature  and the reduced feature data sets were combined into one dataframe
* Once the model with the optimal performance was selected for each model, that metrics for that model were saved to a ```csv file```.
* All csv files for metrics were then loaded and combined into one dataframe for review. **The optimal model for each individual portfolio class was then selected.**
* For each portfolio class, the optimal model was refit and saved for use in creating performance datasets for dashboard usage.



**3. Model Performance/Selection**

Models were evaluated using an ROC-AUC score and F1 score. 

For each model type the best model was determined using a voting-style method, where each highest score for a metric counted as one vote. 

The best performers for each model type were then compared against each other, and the best model for each portfolio class was selected using the same voting-style method.

---

## Contributors

[Ahmad Takatkah](https://github.com/vcpreneur)
[Lourdes Dominguez Begoa](https://github.com/LourdesDB)
[Patricio Gomez](https://github.com/patogogo)
[Lovedeep Singh](https://github.com/LovedeepSingh89)
[Thomas L. Champion](https://github.com/tlchampion)

---

## License

License information can be found in the included LICENSE file.

---

## Disclaimer

The information provided through this application is for information and educational purposes only. 
It is not intended to be, nor should it be used as, investment advice. 
Seek a duly licensed professional for investment advice.


