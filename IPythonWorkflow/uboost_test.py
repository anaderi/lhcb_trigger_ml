#!/usr/bin/env python

import pylab as pl
import numpy as np
from itertools import izip, islice
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from commonutils import generate_sample
from uboost import uBoostBDT, uBoostClassifier
from reports import Predictions, ClassifiersDict


def test_uboost_classifier_real(trainX, trainY, testX, testY):
    # We will try to get uniform distribution along this variable
    uniform_variables = ['column0']

    base_classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=12)

    for target_efficiency in [0.1, 0.3, 0.5, 0.7, 0.9]:
        bdt_classifier = uBoostBDT(
            uniform_variables=uniform_variables,
            target_efficiency=target_efficiency,
            n_neighbors=20,
            n_estimators=20,
            base_estimator=base_classifier,
            algorithm="SAMME.R")
        bdt_classifier.fit(trainX, trainY)
        filtered = np.sum(
            bdt_classifier.predict_proba(trainX[trainY > 0.5])[:, 1] >
            bdt_classifier.bdt_cut)
        assert abs(filtered - np.sum(trainY) * target_efficiency) < 5,\
            "global cut is set wrongly"


def test_uboost_classifier_discrete(trainX, trainY, testX, testY):
    # We will try to get uniform distribution along this variable
    uniform_variables = ['column0']

    base_classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=12)

    for target_efficiency in [0.1, 0.3, 0.5, 0.7, 0.9]:
        bdt_classifier = uBoostBDT(
            uniform_variables=uniform_variables,
            target_efficiency=target_efficiency,
            n_neighbors=20,
            n_estimators=20,
            base_estimator=base_classifier)
        bdt_classifier.fit(trainX, trainY)
        filtered = np.sum(
            bdt_classifier.predict_proba(trainX[trainY > 0.5])[:, 1] >
            bdt_classifier.bdt_cut)
        assert abs(filtered - np.sum(trainY) * target_efficiency) < 5,\
            "global cut wrong"

        staged_filtered_upper = [
            np.sum(pred[:, 1] > cut - 1e-7) for pred, cut in
            izip(bdt_classifier.staged_predict_proba(trainX[trainY > 0.5]),
                 bdt_classifier.bdt_cuts_)]
        staged_filtered_lower = [
            np.sum(pred[:, 1] > cut + 1e-7) for pred, cut in
            izip(bdt_classifier.staged_predict_proba(trainX[trainY > 0.5]),
                 bdt_classifier.bdt_cuts_)]

        assert bdt_classifier.bdt_cut == bdt_classifier.bdt_cuts_[-1],\
            'something wrong with computed cuts'
        for filter_lower, filter_upper in islice(
                izip(staged_filtered_lower, staged_filtered_upper), 10, 100):
            assert filter_lower - 1 <= sum(trainY) * target_efficiency <= \
                filter_upper + 1, "stage cut is set wrongly"

    uboost_classifier = uBoostClassifier(
        uniform_variables=uniform_variables,
        n_neighbors=20,
        efficiency_steps=5,
        n_estimators=20)

    bdt_classifier = uBoostBDT(
        uniform_variables=uniform_variables,
        n_neighbors=20,
        n_estimators=20,
        base_estimator=base_classifier)

    for classifier in [bdt_classifier, uboost_classifier]:
        classifier.fit(trainX, trainY)
        proba1 = classifier.predict_proba(testX)
        proba2 = list(classifier.staged_predict_proba(testX))[-1]
        assert np.allclose(proba1, proba2, atol=0.001),\
            "staged_predict doesn't coincide with the predict for proba."

    assert len(bdt_classifier.feature_importances_) == trainX.shape[1]

    uboost_classifier.fit(trainX, trainY)
    predict_proba = uboost_classifier.predict_proba(testX)
    predict = uboost_classifier.predict(testX)
    error = np.sum(np.abs(predict - testY))
    print("SAMME error %.3f" % (error / float(len(testX))))


def test_classifiers(trainX, trainY, testX, testY, output_name_patern=None):
    uniform_variables = ['column0']
    clf_Ada = AdaBoostClassifier(n_estimators=50)
    clf_uBoost_SAMME = uBoostClassifier(
        uniform_variables=uniform_variables,
        n_neighbors=50,
        efficiency_steps=5,
        n_estimators=50,
        algorithm="SAMME")
    clf_uBoost_SAMME_R = uBoostClassifier(
        uniform_variables=uniform_variables,
        n_neighbors=50,
        efficiency_steps=5,
        n_estimators=50,
        algorithm="SAMME.R")
    clf_dict = ClassifiersDict({
        "Ada": clf_Ada,
        "uSAMME": clf_uBoost_SAMME,
        "uSAMME.R": clf_uBoost_SAMME_R
        })
    clf_dict.fit(trainX, trainY)

    predictions = Predictions(clf_dict, testX, testY)
    predictions.print_mse(uniform_variables, in_html=False)
    # TODO(kazeevn)
    # Make reports save the plots. And rewrite it from using global
    # pl.* calls.
    predictions.mse_curves(uniform_variables)
    if output_name_patern is not None:
        pl.savefig(output_name_patern % "mse_curves", bbox="tight")
    figure1 = pl.figure()
    predictions.learning_curves()
    if output_name_patern is not None:
        pl.savefig(output_name_patern % "learning_curves", bbox="tight")
    predictions.efficiency(uniform_variables)
    if output_name_patern is not None:
        pl.savefig(output_name_patern % "efficiency_curves", bbox="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Run some assert-based tests, "
        "calculate MSE and plot local efficiencies for"
        " AdaBoost, uBoost.SAMME and uBoost.SAMME.R")
    parser.add_argument('-o', '--output-file', type=str,
                        help=r"Filename pattern with one %%s to save "
                        "the plots to. Example: classifiers_%%s.pdf")
    args = parser.parse_args()
    # Tests results depend significantly on the seed
    np.random.seed(42)
    testX, testY = generate_sample(10000, 10, 0.6)
    trainX, trainY = generate_sample(10000, 10, 0.6)
    test_uboost_classifier_real(trainX, trainY, testX, testY)
    test_uboost_classifier_discrete(trainX, trainY, testX, testY)
    test_classifiers(trainX, trainY, testX, testY, args.output_file)
    if args.output_file is None:
        pl.show()

if __name__ == '__main__':
    main()
