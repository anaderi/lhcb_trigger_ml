from __future__ import division, print_function, absolute_import

import numpy
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics.metrics import roc_auc_score
from hep_ml.commonutils import generate_sample
from hep_ml.ugradientboosting import AdaLossFunction
from hep_ml.experiments.fastgb import FastGradientBoostingClassifier
__author__ = 'Alex Rogozhnikov'


def test_gb_quality(n_samples=1000):
    X, y = generate_sample(n_samples=n_samples, n_features=5)
    X = numpy.array(X)
    w = numpy.ones(n_samples)
    gb = FastGradientBoostingClassifier(loss=AdaLossFunction(), n_estimators=100, min_samples_split=50, max_depth=5)
    gb = gb.fit(X, y)

    gb_old = GradientBoostingClassifier(n_estimators=100, min_samples_split=50, max_depth=5)
    gb_old = gb_old.fit(X, y)

    testX, testy = generate_sample(n_samples=n_samples + 1, n_features=5)
    testX = numpy.array(testX)

    auc = roc_auc_score(testy, gb.predict(testX))
    auc_old = roc_auc_score(testy, gb_old.predict(testX))
    print("AUC new", auc)
    print("AUC old", auc_old)
    assert auc > 0.95, auc

test_gb_quality()