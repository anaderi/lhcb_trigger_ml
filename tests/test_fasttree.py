from __future__ import division, print_function, absolute_import
from collections import OrderedDict

import numpy
import time
from sklearn.metrics import roc_auc_score
from hep_ml.commonutils import generate_sample
from hep_ml.experiments.fasttree import FastTreeRegressor
from sklearn.tree import DecisionTreeRegressor

__author__ = 'Alex Rogozhnikov'


def test_tree(n_samples=1000):
    X, y = generate_sample(n_samples=n_samples, n_features=5)
    X = numpy.array(X)
    w = numpy.ones(n_samples)
    tree = FastTreeRegressor()
    tree = tree.fit(X, y, sample_weight=w)
    prediction = tree.predict(X)
    tree.print_tree_stats()
    auc = roc_auc_score(y, prediction)
    print("AUC", auc)
    assert auc > 0.7, auc


def test_tree_speed(n_samples=100000, n_features=10):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features)
    X = numpy.array(X)
    w = numpy.ones(n_samples)

    regressors = OrderedDict()
    regressors['old'] = DecisionTreeRegressor(max_depth=10, min_samples_split=50)
    regressors['new'] = FastTreeRegressor(max_depth=10, min_samples_split=50)

    for name, regressor in regressors.items():
        start = time.time()
        for _ in range(3):
            regressor.fit(X, y, sample_weight=w)
        print(name, 'trains in ', time.time() - start)

    # Testing speed of prediction:
    methods = OrderedDict()
    methods['old'] = lambda: regressors['old'].predict(X)
    methods['new'] = lambda: regressors['new'].apply(X)
    methods['new-fast'] = lambda: regressors['new'].fast_apply(X)
    for name, method in methods.items():
        start = time.time()
        for _ in range(5):
            method()
        print(name, 'requires ', time.time() - start)


def tree_quality_comparison(n_samples=200000, n_features=10):
    trainX, trainY = generate_sample(n_samples=n_samples, n_features=n_features)
    testX, testY = generate_sample(n_samples=n_samples, n_features=n_features)

    # Multiplying by random matrix
    multiplier = numpy.random.normal(size=[n_features, n_features])
    trainX = numpy.dot(trainX.values, multiplier)
    testX = numpy.dot(testX.values, multiplier)
    regressors = OrderedDict()
    regressors['old'] = DecisionTreeRegressor(max_depth=10, min_samples_split=50)
    regressors['new'] = FastTreeRegressor(max_depth=10, min_samples_split=50, criterion='pvalue')
    w = numpy.ones(n_samples)

    for name, regressor in regressors.items():
        regressor.fit(trainX, trainY, sample_weight=w)
        print(name, roc_auc_score(testY, regressor.predict(testX)))

    # Testing apply method
    indices1, values1 = regressors['new'].apply(testX)
    indices2, values2 = regressors['new'].fast_apply(testX)
    assert numpy.all(values1 == values2), 'two apply methods give different results'


tree_quality_comparison(n_samples=1000)