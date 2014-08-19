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


__author__ = 'Alex Rogozhnikov'

# TODO add some tests and assertions here
# TODO based on predict algorithm (SAMME)?
# TODO multiclass classification?


class MeanAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 uniform_variables=None,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=10,
                 learning_rate=1.,
                 n_neighbours=10,
                 uniform_label=1,
                 separate_reweighting=False,
                 train_variables=None):
        self.uniform_variables = uniform_variables
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_neighbours = n_neighbours
        self.uniform_label = uniform_label
        self.separate_reweighting = separate_reweighting
        self.train_variables = train_variables

    def fit(self, X, y, sample_weight=None):
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
            classifier = sklearn.clone(self.base_estimator)
            classifier.fit(X, y, sample_weight=sample_weight)
            score = self.learning_rate * self.compute_score(classifier, X=X)
            cumulative_score += score
            sample_weight *= numpy.exp(- y_signed * numpy.take(score, knn_indices).mean(axis=1))
            sample_weight = self.normalize_weights(y=y, sample_weight=sample_weight)
            self.estimators.append(classifier)

    def normalize_weights(self, y, sample_weight):
        sample_weight += 1e-10
        if self.separate_reweighting:
            sample_weight[y == 0] /= numpy.sum(sample_weight[y == 0])
            sample_weight[y == 1] /= numpy.sum(sample_weight[y == 1])
        else:
            sample_weight /= numpy.sum(sample_weight)

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
        proba[:, 1] = expit(score / self.n_estimators)
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


