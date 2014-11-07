from __future__ import division, print_function, absolute_import
import numpy
from unittest import TestCase
from sklearn.metrics import roc_auc_score

from hep_ml.commonutils import generate_sample
from hep_ml.meanadaboost import MeanAdaBoostClassifier
from hep_ml.experiments.triggermaxvoter import generate_max_voter

__author__ = 'Alex Rogozhnikov'


class TestMeanAdaBoostClassifier(TestCase):
    def setUp(self, n_samples=1000, n_features=5):
        self.trainX, self.trainY = generate_sample(n_samples=n_samples, n_features=n_features)
        self.testX, self.testY = generate_sample(n_samples=n_samples, n_features=n_features)
        self.trainW = numpy.ones(n_samples)
        self.testW = numpy.ones(n_samples)
        self.uniform_variables = self.trainX.columns[:1]
        self.train_variables = self.trainX.columns[1:]

    def check_clf(self, classifier):
        classifier = classifier.fit(self.trainX, self.trainY, sample_weight=self.trainW)
        pred = classifier.predict_proba(self.testX)
        auc = roc_auc_score(self.testY, pred[:, 1])
        print(auc, classifier)
        assert auc > 0.7, 'Too poor predictions'

    def test_workability(self):
        # Testing with standard voters
        for voting in ['mean', 'median', 'random-mean', 'random-percentile']:
            ada = MeanAdaBoostClassifier(voting=voting,
                                         uniform_variables=self.uniform_variables,
                                         train_variables=self.train_variables)
            self.check_clf(ada)

        # TODO testing with matrix voter

        # testing with voting
        n_events = 40
        event_indices = numpy.random.randint(0, n_events, size=len(self.trainY)) + n_events * self.trainY
        voter = generate_max_voter(event_indices)
        self.check_clf(MeanAdaBoostClassifier(voting=voter,
                                              uniform_variables=self.uniform_variables,
                                              train_variables=self.train_variables))

