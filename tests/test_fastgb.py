from __future__ import division, print_function, absolute_import

import numpy
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics.metrics import roc_auc_score
from sklearn.tree.tree import DecisionTreeRegressor
from hep_ml.commonutils import generate_sample
from hep_ml.ugradientboosting import AdaLossFunction, BinomialDevianceLossFunction as BinomialDeviance, \
    uGradientBoostingClassifier
from hep_ml.experiments.categorical import CategoricalTreeRegressor, SimpleCategorizer, ComplexCategorizer
from hep_ml.experiments.fasttree import FastTreeRegressor, FastNeuroTreeRegressor
from hep_ml.experiments.fastgb import TreeGradientBoostingClassifier, CommonGradientBoosting, FoldingGBClassifier
import time

__author__ = 'Alex Rogozhnikov'


def test_workability(n_samples=10000, n_features=10, distance=0.5):
    trainX, trainY = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    testX, testY = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for booster in [FoldingGBClassifier, TreeGradientBoostingClassifier]:
        for loss in [BinomialDeviance(), AdaLossFunction()]:
            for update in [True, False]:
                for base in [FastTreeRegressor(max_depth=3), FastNeuroTreeRegressor(max_depth=3)]:
                    clf = booster(loss=loss, n_estimators=100,
                                  base_estimator=base, update_tree=update)
                    clf.fit(trainX, trainY)
                    print('booster', booster, loss, 'update=', update, ' base=', base.__class__,
                          ' quality=', roc_auc_score(testY, clf.predict_proba(testX)[:, 1]))


# test_workability()


def test_gb_quality(n_samples=10000, n_features=10, distance=0.5):
    trainX, trainY = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    testX, testY = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)

    # Multiplying by random matrix
    multiplier = numpy.random.normal(size=[n_features, n_features])
    shift = numpy.random.normal(size=[1, n_features]) * 5
    trainX = numpy.dot(trainX.values, multiplier) + shift
    testX = numpy.dot(testX.values, multiplier) + shift

    boosters = {
        'old_boost': GradientBoostingClassifier(n_estimators=100, min_samples_split=50, max_depth=5, subsample=0.3),
        'fast+old_tree': CommonGradientBoosting(n_estimators=100,
            base_estimator=DecisionTreeRegressor(min_samples_split=50, max_depth=5)),
        'fast+neuro': TreeGradientBoostingClassifier(n_estimators=100, update_tree=True,
                                                     base_estimator=FastNeuroTreeRegressor()),
        'fold+tree': FoldingGBClassifier(loss=BinomialDeviance(), n_estimators=10, update_tree=True,
                                         base_estimator=FastNeuroTreeRegressor()),
        'ugb': uGradientBoostingClassifier(loss=AdaLossFunction(),
            n_estimators=100, min_samples_split=50, max_depth=5, update_tree=True, subsample=0.3)
    }

    for criterion in ['mse', # 'fmse', # 'pvalue',
                      # 'significance',
                      'significance2',
                      # 'gini',
                      'entropy',
                      'poisson'
    ]:
        boosters['fast-' + criterion[:4]] = TreeGradientBoostingClassifier(n_estimators=100, update_tree=True,
            base_estimator=FastTreeRegressor(criterion=criterion))

    for name, booster in boosters.items():
        start = time.time()
        booster.fit(trainX, trainY)
        auc = roc_auc_score(testY, booster.predict_proba(testX)[:, 1])
        print(name, "spent:{:3.2f} auc:{}".format(time.time() - start, auc))

        # assert new_auc > 0.7, new_auc


test_gb_quality(n_samples=100000)


def test_categorical_gb(n_samples=100000, n_features=10, p=0.7):
    y = numpy.random.random(n_samples) > 0.5
    X = numpy.random.randint(40, size=[n_samples, n_features]) * 2
    X += numpy.random.random(size=[n_samples, n_features]) > p
    X += y[:, numpy.newaxis]

    from sklearn.cross_validation import train_test_split

    trainX, testX, trainY, testY = train_test_split(X, y)
    boosters = {
        'old': GradientBoostingClassifier(n_estimators=100, min_samples_split=50, max_depth=5),
        'cat': CommonGradientBoosting(loss=AdaLossFunction(), subsample=0.5, dtype=int,
            base_estimator=CategoricalTreeRegressor()),
        'cat2': TreeGradientBoostingClassifier(loss=BinomialDeviance(), dtype='int', update_tree=False,
            base_estimator=SimpleCategorizer(n_features=2, n_attempts=3, method='cv')),
        'cat3': TreeGradientBoostingClassifier(loss=BinomialDeviance(), dtype='int', update_tree=False,
            base_estimator=ComplexCategorizer(n_features=10, n_categories_power=5, splits=1, pfactor=0.5)),
        'cat2-2': TreeGradientBoostingClassifier(loss=BinomialDeviance(), dtype='int', update_tree=False, n_threads=2,
            base_estimator=SimpleCategorizer(n_features=2, n_attempts=1)),
    }
    for name, booster in boosters.items():
        start = time.time()
        booster.fit(trainX, trainY)
        auc = roc_auc_score(testY, booster.predict_proba(testX)[:, 1])
        print(name, "spent:{:3.2f} auc:{}".format(time.time() - start, auc))


# test_categorical_gb()