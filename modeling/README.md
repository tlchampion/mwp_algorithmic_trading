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

| **Bagging Classifier**                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model1 = BaggingClassifier(base_estimator=base_classifier, n_estimators=200, max_samples=0.8, max_features=0.5, bootstrap=True, oob_score=True, random_state=42)    |
| model2 = BaggingClassifier(base_estimator=base_classifier, n_estimators=50, max_samples=0.8, max_features=0.5, bootstrap=True, oob_score=True, random_state=42)     |
| model3 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.8, max_features=0.5, bootstrap=True, oob_score=True, random_state=42)    |
| model4 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.9, max_features=0.5, bootstrap=True, oob_score=True, random_state=42)    |
| model5 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.5, max_features=0.5, bootstrap=True, oob_score=True, random_state=42)    |
| model6 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.8, max_features=0.7, bootstrap=True, oob_score=True, random_state=42)    |
| model7 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.8, max_features=0.3, bootstrap=True, oob_score=True, random_state=42)    |
| model8 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.8, max_features=0.5, bootstrap=False, oob_score=False, random_state=42)  |
| model9 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.8, max_features=0.5, bootstrap=True, oob_score=True, random_state=123)   |
| model10 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.8, max_features=0.5, bootstrap=True, oob_score=False, random_state=123) |
| model11 = BaggingClassifier(base_estimator=base_classifier, n_estimators=100, max_samples=0.8, max_features=0.5, bootstrap=True, oob_score=False, random_state=42)  |


| **GaussianNB**                                              |
|-------------------------------------------------------------|
| model1 = GaussianNB(priors=None)                            |
| model2 = GaussianNB(var_smoothing=1e-9, priors=None)        |
| model3 = GaussianNB(var_smoothing=1e-5, priors=[0.2, 0.8])  |
| model4 = GaussianNB(var_smoothing=1e-3, priors=[0.5, 0.5])  |
| model5 = GaussianNB(var_smoothing=1e-1, priors=None)        |
| model6 = GaussianNB(var_smoothing=1.0, priors=[.3, 0.7])    |
| model7 = GaussianNB(var_smoothing=10.0, priors=[0.6, 0.4])  |
| model8 = GaussianNB(var_smoothing=1e-9, priors=[0.4, 0.6])  |
| model9 = GaussianNB(var_smoothing=1e-5, priors=None)        |
| model10 = GaussianNB(var_smoothing=1e-3, priors=[0.2, 0.8]) |
| model11 = GaussianNB(var_smoothing=1e-1, priors=[0.5, 0.5]) |


| **Logistic Regression**                                                                                         |
|-----------------------------------------------------------------------------------------------------------------|
| model1 = LogisticRegression(random_state=42, max_iter=10000, solver='saga', penalty='elasticnet', l1_ratio=0.1) |
| model2 = LogisticRegression(random_state=42, max_iter=10000, solver='saga', penalty='elasticnet', l1_ratio=0.3) |
| model3 = LogisticRegression(random_state=42, max_iter=10000, solver='saga', penalty='elasticnet', l1_ratio=0.5) |
| model4 = LogisticRegression(random_state=42, max_iter=10000, solver='saga', penalty='elasticnet', l1_ratio=0.7) |
| model5 = LogisticRegression(random_state=42, max_iter=10000, solver='saga', penalty='elasticnet', l1_ratio=0.9) |
| model6 = LogisticRegression(random_state=42, max_iter=10000, solver='saga', penalty='l1')                       |
| model7 = LogisticRegression(random_state=42, max_iter=10000, solver='saga', penalty=None)                       |
| model8 = LogisticRegression(random_state=42, max_iter=10000, solver='lbfgs', penalty=None)                      |
| model9 = LogisticRegression(random_state=42, max_iter=10000, solver='liblinear', penalty='l1')                  |
| model10 = LogisticRegression(random_state=42, max_iter=10000, solver='liblinear', penalty='l2')                 |
| model11 = LogisticRegression(random_state=42, max_iter=10000, solver='sag', penalty=None)                       |


| **Random Forest**                                                                                                                                                                                                                                   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model1 = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', bootstrap=True, criterion='gini', min_impurity_decrease=0.0, class_weight=None, oob_score=False)                      |
| model2 = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=5, max_features='log2', bootstrap=True, criterion='entropy', min_impurity_decrease=0.001, class_weight='balanced_subsample', oob_score=True) |
| model3 = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=20, min_samples_leaf=10, max_features=0.5, bootstrap=True, criterion='gini', min_impurity_decrease=0.005, class_weight={0: 1, 1: 3}, oob_score=True)              |
| model4 = RandomForestClassifier(n_estimators=1000, max_depth=40, min_samples_split=50, min_samples_leaf=20, max_features=None, bootstrap=True, criterion='entropy', min_impurity_decrease=0.01, class_weight={0: 1, 1: 5}, oob_score=True)          |
| model5 = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=5, max_features=0.7, bootstrap=True, criterion='gini', min_impurity_decrease=0.0, class_weight=None, oob_score=False)                        |
| model6 = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=20, min_samples_leaf=10, max_features=0.3, bootstrap=True, criterion='entropy', min_impurity_decrease=0.0, class_weight='balanced', oob_score=True)               |
| model7 = RandomForestClassifier(n_estimators=1000, max_depth=40, min_samples_split=50, min_samples_leaf=20, max_features='sqrt', bootstrap=True, criterion='gini', min_impurity_decrease=0.0, class_weight={0: 1, 1: 10}, oob_score=True)           |
| model8 = RandomForestClassifier(n_estimators=2000, max_depth=50, min_samples_split=100, min_samples_leaf=50, max_features='log2', bootstrap=True, criterion='entropy', min_impurity_decrease=0.0, class_weight=None, oob_score=False)               |
| model9 = RandomForestClassifier(n_estimators=1000, max_depth=30, min_samples_split=20, min_samples_leaf=10, max_features=None, bootstrap=True, criterion='gini', min_impurity_decrease=0.005, class_weight='balanced', oob_score=True)              |
| model10 = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=10, min_samples_leaf=5, max_features=0.7, bootstrap=True, criterion='entropy', min_impurity_decrease=0.001, class_weight={0: 1, 1: 5}, oob_score=True)           |
| model11 = RandomForestClassifier(n_estimators=1000, max_depth=30, min_samples_split=10, min_samples_leaf=5, max_features=0.5, bootstrap=True, criterion='entropy', min_impurity_decrease=0.001, class_weight='balanced', oob_score=True)            |


| **SVM**                                                                                |
|----------------------------------------------------------------------------------------|
| model1 = SVC(random_state=42, max_iter=1000, kernel='linear', C=0.5, probability=True) |
| model2 = SVC(random_state=42, max_iter=1000, kernel='linear',C=1, probability=True)    |
| model3 = SVC(random_state=42, max_iter=1000, kernel='linear',C=10, probability=True)   |
| model4 = SVC(random_state=42, max_iter=1000, kernel='rbf',C=0.5, probability=True)     |
| model5 = SVC(random_state=42, max_iter=1000, kernel='rbf',C=1, probability=True)       |
| model6 = SVC(random_state=42, max_iter=1000, kernel='rbf',C=10, probability=True)      |
| model7 = SVC(random_state=42, max_iter=1000, kernel='sigmoid',C=0.5, probability=True) |
| model8 = SVC(random_state=42, max_iter=1000, kernel='sigmoid',C=1, probability=True)   |
| model9 = SVC(random_state=42, max_iter=1000, kernel='sigmoid',C=10, probability=True)  |


| **AdaBoost Classifier**                                                                                                     |
|-----------------------------------------------------------------------------------------------------------------------------|
| model1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=1.0)        |
| model2 = AdaBoostClassifier(base_estimator=LogisticRegression(solver='lbfgs'), n_estimators=50, learning_rate=0.5)          |
| model3 = AdaBoostClassifier(base_estimator=SVC(kernel='linear'), n_estimators=200, learning_rate=0.1)                       |
| model4 = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50), n_estimators=100, learning_rate=1.0)    |
| model5 = AdaBoostClassifier(base_estimator=GradientBoostingClassifier(max_depth=3), n_estimators=150, learning_rate=0.2)    |
| model6 = AdaBoostClassifier(base_estimator=XGBClassifier(max_depth=4), n_estimators=100, learning_rate=0.5)                 |
| model7 = AdaBoostClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=50, learning_rate=1.0)         |
| model8 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=200, learning_rate=0.01)       |
| model9 = AdaBoostClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(50, 50)), n_estimators=100, learning_rate=0.1) |
| model10 = AdaBoostClassifier(base_estimator=QuadraticDiscriminantAnalysis(), n_estimators=50, learning_rate=0.5)            |
| model11 = AdaBoostClassifier(base_estimator=LinearSVC(max_iter=10000), n_estimators=150, learning_rate=0.05)                |


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


