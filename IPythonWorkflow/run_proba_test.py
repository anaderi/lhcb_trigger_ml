#!/usr/bin/env python2
"""
Run proba_test on a selection of classifiers
"""
import argparse
import numpy as np
import pylab as pl
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

from uboost import uBoostClassifier
from reports import ClassifiersDict
from commonutils import generate_sample
from proba_test import proba_test


def main():
    parser = argparse.ArgumentParser(
        description="Plot probability vs. real frequency graphs for a "
        "selection of classifiers.")
    parser.add_argument('-o', '--output-file', type=str,
                        help=r"Filename pattern with one %%s to save "
                        "the plots to. Example: classifiers_%%s.pdf")
    parser.add_argument('-s', '--random-seed', type=int,
                        help="Random generator seed to use.")
    args = parser.parse_args()
    if args.random_seed:
        np.random.seed(args.random_seed)
    else:
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
        error, ax, fig = proba_test(classifier, x_test, y_test, name=name)
        print("%s: %f" % (name, error))
        if args.output_file:
            fig.savefig(args.output_file % name, bbox="tight")

    if not args.output_file:
        pl.show()

if __name__ == '__main__':
    main()
