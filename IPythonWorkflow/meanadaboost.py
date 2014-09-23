from __future__ import division

import numbers
import numpy
import pandas
from scipy.special import expit
import sklearn
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils.validation import column_or_1d, check_arrays
from sklearn.base import BaseEstimator, ClassifierMixin
from commonutils import check_sample_weight, computeKnnIndicesOfSameClass

# About

# In this module some very simple modifications of AdaBoost
# is presented. In weights boosting it uses mean of neighbours scores,
import commonutils


__author__ = 'Alex Rogozhnikov'

# TODO add some tests and assertions here
# TODO algorithm based on predict (SAMME)?
# TODO multiclass classification?


class MeanAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 uniform_variables=None,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=10,
                 learning_rate=1.,
                 n_neighbours=10,
                 uniform_label=1,
                 train_variables=None,
                 voting='mean'):
        self.uniform_variables = uniform_variables
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_neighbours = n_neighbours
        self.uniform_label = uniform_label
        self.train_variables = train_variables
        self.voting = voting

    def fit(self, X, y, sample_weight=None, A=None):
        # TODO add checks here
        if self.voting == 'matrix':
            assert A is not None, 'A matrix should be passed'
            assert A.shape[0] == len(X) and A.shape[1] == len(X), 'wrong shape of passed matrix'
        label = self.uniform_label
        self.uniform_label = numpy.array([label]) if isinstance(label, numbers.Number) else numpy.array(label)

        sample_weight = check_sample_weight(y, sample_weight=sample_weight).copy()
        assert numpy.all(numpy.in1d(y, [0, 1])), 'only two-class classification is supported by now'
        y = column_or_1d(y)
        y_signed = 2 * y - 1

        X = pandas.DataFrame(X)
        knn_indices = computeKnnIndicesOfSameClass(self.uniform_variables, X, y, self.n_neighbours)

        # for those events with non-uniform label we repeat it's own index several times
        for label in [0, 1]:
            if label not in self.uniform_label:
                knn_indices[y == label, :] = numpy.arange(len(y))[y == label][:, numpy.newaxis]

        X = self.get_train_vars(X)
        cumulative_score = numpy.zeros(len(X))
        self.estimators = []

        for stage in range(self.n_estimators):
            knn_scores = numpy.take(cumulative_score, knn_indices)
            if self.voting == 'mean':
                voted_score = numpy.mean(knn_scores, axis=1)
            elif self.voting == 'median':
                voted_score = numpy.median(knn_scores, axis=1)
            elif self.voting == 'percentile':
                voted_score = numpy.percentile(knn_scores, numpy.random.random(), axis=1)
            elif self.voting == 'random-mean':
                n_feats = numpy.random.randint(self.n_neighbours//2, self.n_neighbours)
                voted_score = numpy.mean(knn_scores[:, :n_feats], axis=1)
            elif self.voting == 'matrix':
                voted_score = A.dot(cumulative_score)
            else: # self.voting is callable
                voted_score = self.voting(cumulative_score, knn_scores)

            weight = sample_weight * numpy.exp(- y_signed * voted_score)
            weight = self.normalize_weights(y=y, sample_weight=weight)

            classifier = sklearn.clone(self.base_estimator)
            classifier.fit(X, y, sample_weight=weight)
            cumulative_score += self.learning_rate * self.compute_score(classifier, X=X)
            self.estimators.append(classifier)

    @staticmethod
    def normalize_weights(y, sample_weight):
        sample_weight[y == 0] /= numpy.sum(sample_weight[y == 0])
        sample_weight[y == 1] /= numpy.sum(sample_weight[y == 1])
        return sample_weight

    def predict_score(self, X):
        X = self.get_train_vars(X)
        return self.learning_rate * sum(self.compute_score(clf, X) for clf in self.estimators)

    def staged_predict_score(self, X):
        X = self.get_train_vars(X)
        score = numpy.zeros(len(X))
        for clf in self.estimators:
            score += self.learning_rate * self.compute_score(clf, X)
            yield score

    def score_to_proba(self, score):
        """Compute class probability estimates from decision scores."""
        proba = numpy.zeros((score.shape[0], 2), dtype=numpy.float64)
        proba[:, 1] = expit(numpy.clip(score / self.n_estimators, -500, 500))
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

    @staticmethod
    def compute_score(clf, X):
        """X should include only train vars"""
        p = clf.predict_proba(X)
        p[p <= 1e-5] = 1e-5
        return numpy.log(p[:, 1] / p[:, 0])

    def get_train_vars(self, X):
        """Gets the DataFrame and returns only columns
           that should be used in fitting / predictions"""
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]

    def predict_proba(self, X):
        return self.score_to_proba(self.predict_score(X))

    def staged_predict_proba(self, X):
        for score in self.staged_predict_score(X):
            yield self.score_to_proba(score=score)


def generate_max_voter(event_indices):
    """
    This voter is prepared specially for using in triggers.
    Voting returns max(svr predictions),
    :param event_indices: array, each element is

    """
    groups = commonutils.indices_of_values(event_indices)
    def voter(cumulative_score, knn_scores):
        result = numpy.zeros(len(cumulative_score))
        for key, group in groups:
            result[group] = numpy.max(cumulative_score[group])
    return voter


