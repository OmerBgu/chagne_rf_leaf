from __future__ import division

import warnings
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import train_test_split
import  pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  RandomizedSearchCV

warnings.simplefilter(action='ignore', category=Warning)


names = ["PhishingData.csv","EEG Eye State.csv","MAGIC Gamma Telescope.csv","fertility_Diagnosis.csv"]



my_mse_list = []
original_mse_list = []
my_oob_score_list = []
original_oob_score_list = []


if __name__ == "__main__":
    i = 0
    for data in names:

        path = r"C:\Users\omera\Downloads\data_sets\\" + data
        print("read {}".format(path))
        df = pd.read_csv(path)
        # df =df.values
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        y = y.values
        X = X.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

        if i == 70:
            # hyperparameter tunining
            param_grid = {
                'bootstrap': [True],
                'max_depth': [100, 200],
                'max_features': [2, 3,7],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [250, 500]
            }


            rf = RandomForestClassifier(oob_score=True, warm_start=True)
            #grid_search = RandomizedSearchCV(estimator=rf, param_distributions =param_grid,cv=3, n_jobs=-1, verbose=1, n_iter=50)

            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                       cv=3, n_jobs=-1, verbose=1)

            # Fit the grid search to the data
            grid_search.fit(X_train, y_train)
            best_grid = grid_search.best_estimator_
            i = 0
            print("best param for {} : {}".format(names[i], grid_search.best_params_))
            f = open(r"C:\Users\omera\Downloads\parms.txt", "a")
            f.write("best param for {} : {}".format(names[i], grid_search.best_params_))
            f.close()

            exit(1)

        #setting best params fited by CV
        if i == 0:  # PhishingData
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, bootstrap=True, max_depth=90, max_features=7,
                                                  min_samples_leaf=3, min_samples_split=8, n_estimators=500)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, bootstrap=True, max_depth=90, max_features=7,
                                                  min_samples_leaf=3, min_samples_split=8, n_estimators=500)
        elif i == 1:  # EEG Eye State
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True,bootstrap=True, max_depth=90,
                                                  max_features=7, min_samples_leaf=3, min_samples_split=8, n_estimators=500)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True,bootstrap=True, max_depth=90,
                                                  max_features=7, min_samples_leaf=3, min_samples_split=8, n_estimators=500)
        elif i == 2:  # MAGIC Gamma Telescope
            my_estimator = RandomForestClassifier(oob_score=True, warm_start=True, bootstrap=True, max_depth=90, max_features=2,
                                                  min_samples_leaf=3, min_samples_split=12, n_estimators=100)
            original_estimator = RandomForestClassifier(oob_score=True, warm_start=True, bootstrap=True, max_depth=90, max_features=2,
                                                  min_samples_leaf=3, min_samples_split=12, n_estimators=100)

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
