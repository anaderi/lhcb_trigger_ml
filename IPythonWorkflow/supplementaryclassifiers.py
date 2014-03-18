from sklearn.base import BaseEstimator, ClassifierMixin, clone


class HidingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, train_variables=None, base_estimator=None):
        """This is a dumb meta-classifier, which uses only subset of variables to train
        base classifier and to estimate result"""
        self.train_variables = train_variables
        self.base_estimator = base_estimator

    def fit(self, X, y):
        assert self.base_estimator is not None, "base estimator was not set"
        self._trained_estimator = clone(self.base_estimator)
        self._trained_estimator.fit(X[self.train_variables], y)

    def predict(self, X):
        return self._trained_estimator.predict(X[self.train_variables])

    def predict_proba(self, X):
        return self._trained_estimator.predict_proba(X[self.train_variables])

    def staged_predict_proba(self, X):
        return self._trained_estimator.staged_predict_proba(X[self.train_variables])
