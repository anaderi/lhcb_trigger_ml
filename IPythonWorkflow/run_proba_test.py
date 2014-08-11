#!/usr/bin/env python2
"""
Run proba_test on a selection of classifiers
"""
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from uboost import uBoostClassifier
from reports import ClassifiersDict
import numpy as np
import pylab as pl

from commonutils import generate_sample
from proba_test import proba_test


def main():
    np.random.seed(42)
    clf_dict = ClassifiersDict({
        'Ada_SAMME': AdaBoostClassifier(algorithm="SAMME"),
        'Ada_SAMME_R': AdaBoostClassifier(algorithm="SAMME.R"),
        'Dummy': DummyClassifier("uniform"),
        'clf_uBoost_SAMME_R': uBoostClassifier(
            uniform_variables=['column0'],
            n_neighbors=50,
            efficiency_steps=5,
            n_estimators=50,
            algorithm="SAMME.R"),
        'clf_uBoost_SAMME': uBoostClassifier(
            uniform_variables=['column0'],
            n_neighbors=50,
            efficiency_steps=5,
            n_estimators=50,
            algorithm="SAMME"),
        'GradientBoosting': GradientBoostingClassifier(),
        'Bayes': GaussianNB()
        })
    x_train, y_train = generate_sample(5000, 10, 0.42)
    x_test, y_test = generate_sample(3000, 10, 0.42)
    clf_dict.fit(x_train, y_train)
    for name, classifier in clf_dict.iteritems():
        print("%s: %f" % (
            name, proba_test(classifier, x_test, y_test, name=name)[0]))

    pl.show()

if __name__ == '__main__':
    main()
