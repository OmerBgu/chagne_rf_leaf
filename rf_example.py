
from __future__ import division


import warnings


warnings.simplefilter(action='ignore', category=Warning)

import warnings
from warnings import warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.utils.validation import *

from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_diabetes

import pandas as pd
debug =1
if not debug:
    names = ["car.csv", "iris.csv", "poker-hand-testing.csv", "wine.csv", "transfusion.csv"]
else:
    names = ["PhishingData.csv"]

# iris = load_iris()
# X = iris.data
# y = iris.target
for data in names:
    path = r"C:\Users\omera\Downloads\data_sets\\" + data
    df = pd.read_csv(path)
    #df =df.values
    y = df.iloc[:,-1]
    X = df.iloc[:,:-1]
    y= y.values
    X = X.values

    # print("df:\n {}".format(df))
    print("y {}".format(y[0:5]))
    # print("X {}".format(X))
    print("df.dtypes : \n {}".format(df.dtypes))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    print("typ x {} type y {}".format(type(X_train),type(y_train)))


    my_estimator = RandomForestClassifier(n_estimators=3)
    original_estimator = RandomForestClassifier(n_estimators=3)

    original_estimator.fit(X_train, y_train)

    y_predicted = original_estimator.predict(X_test)
    y_predicted[0] = int(y_predicted[0])


    y_test[0] = int(y_test[0])


    print("y_test {}".format(type(y_test[0])))
    RF_mse = mean_squared_error(y_test, y_predicted)


    my_estimator.fit_new(X_train, y_train)
    y_predicted = my_estimator.predict_new(X_test)
    print("len(X_test {})".format(len(X_test)))
    print("y_predicted {} in len {}".format(y_predicted,len(y_predicted)))
    print("y_test {} in len {}".format(y_test,len(y_test)))

    my_mse = mean_squared_error(y_test, y_predicted)



    print("my mse {} and RF mse  {}".format(my_mse,RF_mse ))

