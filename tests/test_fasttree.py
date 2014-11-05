from __future__ import division, print_function, absolute_import

import numpy
from sklearn.metrics import roc_auc_score
from hep_ml.commonutils import generate_sample
from hep_ml.experiments.fasttree import FastTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# for super-puper hack
from six.moves import builtins
__author__ = 'Alex Rogozhnikov'


def test_tree(n_samples=1000):
    X, y = generate_sample(n_samples=n_samples, n_features=5)
    X = numpy.array(X)
    w = numpy.ones(n_samples)
    tree = FastTreeRegressor()
    tree = tree.fit(X, y, sample_weight=w)
    prediction = tree.predict(X)
    tree.print_tree_stats()
    print(prediction)
    auc = roc_auc_score(y, prediction)
    print("AUC", auc)
    assert auc > 0.7, auc


def test_tree_speed(n_samples=100000, n_features=10):
    import timeit
    tree_regressor = DecisionTreeRegressor(max_depth=10, min_samples_split=50)
    tree_fast_regressor = FastTreeRegressor(max_depth=10, min_samples_split=50)
    X, y = generate_sample(n_samples=n_samples, n_features=n_features)
    X = numpy.array(X)
    w = numpy.ones(n_samples)
    timer1 = timeit.Timer('tree_regressor.fit(X, y, sample_weight=w)')
    timer2 = timeit.Timer('tree_fast_regressor.fit(X, y, sample_weight=w)')
    # super-puper-hack!
    builtins.__dict__.update(locals())

    time1 = timer1.timeit(1)
    time2 = timer2.timeit(1)
    print('times', time1, time2)
    assert time1 > time2

