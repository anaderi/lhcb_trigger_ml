from __future__ import print_function, division, absolute_import

from math import sqrt
import numpy
import pandas
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import column_or_1d

from .commonutils import sigmoid_function, check_sample_weight

__author__ = "Alex Rogozhnikov"


class HidingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, train_variables=None, base_estimator=None):
        """This is a simple meta-classifier, which uses only subset of variables to train
        base classifier and to estimate result.

        If you train alone classifier, you can of course remove the features you don't
        want classifier to use from data set. But for simplicity of comparing different classifiers
        using this meta-classifier is more appropriate way:
        >>> from sklearn.ensemble import AdaBoostClassifier
        to train on all features
        >>> classifier = AdaBoostClassifier(n_estimators=100)
        to train only on 'momentum' and 'charge'
        >>> classifier = HidingClassifier(train_variables=['momentum', 'charge'],
        >>>                               base_estimator=AdaBoostClassifier(n_estimators=100))
        """
        self.train_variables = train_variables
        self.base_estimator = base_estimator

    def fit(self, X, y, **kwargs):
        assert self.base_estimator is not None, "base estimator was not set"
        self._trained_estimator = clone(self.base_estimator)
        self._trained_estimator.fit(X[self.train_variables], y, **kwargs)
        return self

    def predict(self, X):
        return self._trained_estimator.predict(X[self.train_variables])

    def predict_proba(self, X):
        return self._trained_estimator.predict_proba(X[self.train_variables])

    def staged_predict_proba(self, X):
        return self._trained_estimator.staged_predict_proba(X[self.train_variables])


class AbstractBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    The base class to incorporate routines frequently used in Boosting Classifiers.
    This stub abstract class supports:

    In general, we expect boosting classifier:
      * to have `n_estimators`, `estimators` properties
      * to use `train_variables` property to know which features to use
      * to use score, this means that score is represented as sum of
            score(event) = sum_{classifiers} classifier.score(event)
        the score is 'translated' to probabilities by sigmoid function.
      * two classes

    Of course, this all can be overriden.
    """

    @staticmethod
    def check_input(X, y, sample_weight, check_two_classes=True):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        assert len(X) == len(y), 'Different lengths'
        X = pandas.DataFrame(X)
        y = column_or_1d(y)

        if check_two_classes:
            assert numpy.all(numpy.in1d(y, [0, 1])), \
                'only two-class classification is supported by now'
        return X, y, sample_weight

    def staged_predict_score(self, X):
        X = self.get_train_vars(X)
        score = numpy.zeros(len(X))
        for clf in self.estimators:
            score += self.learning_rate * self.compute_score(clf, X)
            yield score

    def predict_score(self, X):
        X = self.get_train_vars(X)
        return self.learning_rate * sum(self.compute_score(clf, X) for clf in self.estimators)

    def score_to_proba(self, score):
        """Compute class probability estimates from decision scores."""
        proba = numpy.zeros((score.shape[0], 2), dtype=numpy.float)
        proba[:, 1] = sigmoid_function(score, sqrt(self.n_estimators))
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

    @staticmethod
    def compute_score(clf, X):
        """X should include only train vars"""
        p = clf.predict_proba(X)
        p += 1e-5
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
            yield self.score_to_proba(score)


