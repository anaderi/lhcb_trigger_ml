from __future__ import print_function, division

from sklearn.base import BaseEstimator, ClassifierMixin, clone

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



