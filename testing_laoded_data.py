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
from sklearn.model_selection import  RandomizedSearchCV

warnings.simplefilter(action='ignore', category=Warning)


#data_sets = [load_iris,load_boston,load_diabetes,load_breast_cancer,load_digits,load_wine,fetch_olivetti_faces,fetch_covtype]
#names = ["iris", "boston", "diabetes", "breast_cancer", "digits","wine","fetch_olivetti_faces","fetch_covtype"]

data_sets = [fetch_covtype]
names = ["fetch_covtype"]


my_mse_list = []
original_mse_list = []
my_oob_score_list = []
original_oob_score_list = []


if __name__ == "__main__":
    i = 0
    for data in data_sets:
        # loading the data and split to  train and test
        df = data()
        X = df.data.astype(float)
        y = df.target.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

        if i == 70:
            # hyperparameter tunining
            param_grid = {
                'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': [2, 3,7],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 300, 500]
            }

            rf = RandomForestClassifier(oob_score=True, warm_start=True)
            grid_search = RandomizedSearchCV(estimator=rf, param_distributions =param_grid,
                                       cv=3, n_jobs=-1, verbose=1, n_iter=50)

            # Fit the grid search to the data
            grid_search.fit(X_train, y_train)
            best_grid = grid_search.best_estimator_
            print("best param for {} : {}".format(names[i], grid_search.best_params_))
            f = open(r"C:\Users\omera\Downloads\parms.txt", "a")
            f.write("best param for {} : {}".format(names[i], grid_search.best_params_))
            f.close()



        #setting best params fited by CV
        if i == 10:  # iris
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True,n_estimators=100,min_samples_split=8,
                                                  min_samples_leaf=3,max_features=2,max_depth=100, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True,n_estimators=100,
                                                        min_samples_split=8,min_samples_leaf=3, max_features=2,
                                                        max_depth=100, bootstrap=True)
        elif i == 1:  # load_boston
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=100,
                                                  min_samples_split=10,min_samples_leaf=3, max_features=2,
                                                  max_depth=110, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=100,
                                                        min_samples_split=10, min_samples_leaf=3, max_features=2,
                                                        max_depth=110, bootstrap=True)
        elif i == 2:  # load_diabetes
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=300,
                                                  min_samples_split=12,min_samples_leaf=3, max_features=3,
                                                  max_depth=110, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=300,
                                                        min_samples_split=12, min_samples_leaf=3, max_features=3,
                                                        max_depth=110, bootstrap=True)
        elif i == 3:  # load_breast_cancer
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=200,
                                                  min_samples_split=10, min_samples_leaf=4, max_features=3,
                                                  max_depth=110, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=200,
                                                        min_samples_split=10, min_samples_leaf=4, max_features=3,
                                                        max_depth=110, bootstrap=True)
        elif i == 4:  # load_digits
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=500,
                                                  min_samples_split=8, min_samples_leaf=3, max_features=3,
                                                  max_depth=100, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=500,
                                                        min_samples_split=8, min_samples_leaf=3, max_features=3,
                                                        max_depth=100, bootstrap=True)
        elif i == 5:  # load_wine
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=200,
                                                  min_samples_split=10, min_samples_leaf=4, max_features=2,
                                                  max_depth=110, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=200,
                                                        min_samples_split=10, min_samples_leaf=4, max_features=2,
                                                        max_depth=110, bootstrap=True)
        elif i == 6:  # fetch_olivetti_faces
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=300, min_samples_split=8,
                                                  min_samples_leaf=3, max_features=3, max_depth=110, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, n_estimators=300,
                                                        min_samples_split=8, min_samples_leaf=3, max_features=3,
                                                        max_depth=110, bootstrap=True)
        elif i == 0: #fetch_covtype
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True,n_estimators=500, min_samples_split=8,
                                                  min_samples_leaf=3, max_features=7, max_depth=110, bootstrap=True)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True,n_estimators=500,
                                                        min_samples_split=8, min_samples_leaf=3, max_features=7,
                                                        max_depth=110, bootstrap=True)

        print("start fit original...")
        original_estimator.fit(X_train, y_train)

        print("start predict original...")
        y_predicted = original_estimator.predict(X_test)

        y_test[0] = int(y_test[0])
        RF_mse = mean_squared_error(y_test, y_predicted)

        print("start fit my model...")
        my_estimator.fit_new(X_train, y_train)
        print("start predict my model...")
        y_predicted = my_estimator.predict_new(X_test)

        my_mse = mean_squared_error(y_test, y_predicted)
        print("{}:"
              "".format(names[i]))
        print("my mse {} and RF mse  {}".format(my_mse, RF_mse))
        print("my OOB score {} , RF my OOB score {}".format(my_estimator.oob_score_,original_estimator.oob_score_))

        my_mse_list.append(my_mse)
        original_mse_list.append(RF_mse)
        my_oob_score_list.append(my_estimator.oob_score_)
        original_oob_score_list.append(original_estimator.oob_score_)
        i += 1

    time.sleep(1)
    f = open(r"C:\Users\omera\Downloads\results.txt", "a")


    print("\n\n\n")
    for (my_mse, org_mse, my_oob,org_oob,name) in zip(my_mse_list, original_mse_list,
                                                                                my_oob_score_list,original_oob_score_list,
                                                                                names):
        print("in {} : \nmy_mse {} , org_mse {} , my_oob {} ,org_oob {} ".format(name,my_mse, org_mse,
                                                                                              my_oob,org_oob))

        f.write("in {} : \nmy_mse {} , org_mse {} , my_oob {} ,org_oob {} ".format(name,my_mse,org_mse,my_oob,org_oob))
        f.write("\n")

    f.close()
