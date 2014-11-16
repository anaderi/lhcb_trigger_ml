from __future__ import division, print_function, absolute_import

import numpy
from hep_ml.experiments.metaclassifiers import FeatureSplitter, DumbSplitter, ChainClassifiers
from hep_ml import commonutils

from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from collections import OrderedDict

__author__ = 'Alex Rogozhnikov'


def test_feature_splitter(size=2000):
    X, y = commonutils.generate_sample(size, 10, distance=0.5)
    X['column0'] = numpy.clip(numpy.array(X['column0']).astype(numpy.int), -2, 2)
    trainX, testX, trainY, testY = commonutils.train_test_split(X, y)
    base_estimators = {'rf': RandomForestClassifier()}
    splitter = FeatureSplitter('column0', base_estimators=base_estimators, final_estimator=RandomForestClassifier())
    splitter.fit(trainX, trainY)

    print(splitter.score(testX, testY))
    print(RandomForestClassifier().fit(trainX, trainY).score(testX, testY))
    print(DumbSplitter('column0', base_estimator=RandomForestClassifier()).fit(trainX, trainY).score(testX, testY))
    chain = OrderedDict()
    chain['QDA'] = QDA()
    chain['LDA'] = LDA()
    chain['RF'] = RandomForestClassifier()
    print(ChainClassifiers(chain).fit(trainX, trainY).score(testX, testY))
    print(LDA().fit(trainX, trainY).score(testX, testY))

