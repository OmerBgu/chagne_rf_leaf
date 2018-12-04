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


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))


X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
rf = RandomForestClassifier(max_depth=2, random_state=0)
print(X.shape)


rf.fit_GaussianNB(X, y)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                max_depth=2, max_features='auto', max_leaf_nodes=None,
#                min_impurity_decrease=0.0, min_impurity_split=None,
#                min_samples_leaf=1, min_samples_split=2,
#                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#                oob_score=False, random_state=0, verbose=0, warm_start=False)
#print(clf.feature_importances_)
print(rf.predict([[0, 0, 0, 0]]))
# print's , means change proba  in this file   File tree.py line 830, proba : [[0.04218362 0.95781638]]