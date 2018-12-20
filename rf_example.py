
from __future__ import division


import warnings

from sklearn.ensemble.forest import MAX_INT, _parallel_build_trees, _generate_sample_indices
from sklearn.externals.joblib import Parallel, delayed
from sklearn.neighbors.typedefs import DTYPE
from sklearn.tree.tree import DOUBLE
from sklearn.utils import compute_sample_weight

warnings.simplefilter(action='ignore', category=FutureWarning)

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



iris = load_iris()
X = iris.data
y = iris.target



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
estimator = RandomForestClassifier(n_estimators=3)

#print("x.shape : {} x : {} ".format(X.shape, X))
estimator.fit_new(X_train, y_train)
y_predicted = estimator.predict_new(X_test)
print("len(X_test {})".format(len(X_test)))
print("y_predicted {} in len {}".format(y_predicted,len(y_predicted)))
print("y_test {} in len {}".format(y_test,len(y_test)))

mse = mean_squared_error(y_test, y_predicted)
print("mse {}".format(mse))

