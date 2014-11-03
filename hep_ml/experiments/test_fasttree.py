from __future__ import division, print_function, absolute_import

import numpy
from sklearn.metrics import roc_auc_score
from ..commonutils import generate_sample
from . import fasttree
__author__ = 'Alex Rogozhnikov'

def test_tree(n_samples=1000):
    X, y = generate_sample(n_samples=n_samples, n_features=5)
    w = numpy.ones(n_samples)
    tree = fasttree.FastTreeRegressor()
    tree = tree.fit(X, y, sample_weight=w)
    labels = tree.predict(X) > 0

    assert roc_auc_score(y, labels) > 0.7
