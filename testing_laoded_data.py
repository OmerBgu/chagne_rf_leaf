from __future__ import division

import warnings
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import GridSearchCV

warnings.simplefilter(action='ignore', category=Warning)


data_sets = [load_iris,load_boston,load_diabetes,load_breast_cancer,load_digits,load_wine,fetch_olivetti_faces,
          fetch_covtype]
names = ["iris", "boston", "diabetes", "breast_cancer", "digits","wine","fetch_olivetti_faces",
      "fetch_covtype"]
#
# data_sets = [load_iris]
# names = ["iris"]


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    return accuracy,errors


#,"fetch_lfw_people", "fetch_covtype","fetch_kddcup99"]
my_mse_list = []
original_mse_list = []
my_oob_score_list = []
original_oob_score_list = []
best_parms = []
accuracies = []
errors = []
i = 0
for data in data_sets:
    # loading the data and split to  train and test
    df = data()
    X = df.data.astype(float)
    y = df.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    # hyperparameter tunining
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    rf = RandomForestClassifier(oob_score=True, warm_start=True)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    best_parms.append(grid_search.best_params_)
    best_grid = grid_search.best_estimator_

    print("best param for {} : {}".format(names[i], grid_search.best_params_))
    grid_accuracy,grid_error = evaluate(best_grid, X_test, y_test)
    accuracies.append(grid_accuracy)
    errors.append(grid_error)

    use_same_est = 1
    if not use_same_est:
        my_estimator = RandomForestClassifier(oob_score=True, warm_start=True)
        original_estimator = RandomForestClassifier(oob_score=True, warm_start=True)
    else:
        my_estimator = best_grid
        original_estimator = best_grid

    original_estimator.fit(X_train, y_train)

    y_predicted = original_estimator.predict(X_test)

    y_test[0] = int(y_test[0])

    RF_mse = mean_squared_error(y_test, y_predicted)

    my_estimator.fit_new(X_train, y_train)
    y_predicted = my_estimator.predict_new(X_test)
    print("len(X_test {})".format(len(X_test)))
    print("y_predicted {} in len {}".format(y_predicted, len(y_predicted)))
    print("y_test {} in len {}".format(y_test, len(y_test)))

    my_mse = mean_squared_error(y_test, y_predicted)

    print("my mse {} and RF mse  {}".format(my_mse, RF_mse))
    print("my OOB score {} , RF my OOB score {}".format(my_estimator.oob_score_,original_estimator.oob_score_))

    my_mse_list.append(my_mse)
    original_mse_list.append(RF_mse)
    my_oob_score_list.append(my_estimator.oob_score_)
    original_oob_score_list.append(original_estimator.oob_score_)
    i += 1

time.sleep(1)
print("\n\n\n")
for (my_mse, org_mse, my_oob,org_oob,name, best_parm,accuracy,error) in zip(my_mse_list, original_mse_list,
                                                                            my_oob_score_list,original_oob_score_list,
                                                                            names, best_parms,accuracies,errors):
    print("in {} : \nmy_mse {} , org_mse {} , my_oob {} ,org_oob {} best_parm {}".format(name,my_mse, org_mse,
                                                                                      my_oob,org_oob,best_parm))
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))


