from sklearn.base import BaseEstimator, ClassifierMixin, clone


class HidingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, train_variables, base_estimator):
        """This is a metaclassifier, which uses only subset of variables to train
        base classifier"""
        self.train_variables = train_variables
        self.base_estimstor = base_estimator

    def fit(self, X, y):
        self._trained_estimator = clone(self.base_estimstor)\
            .fit(X[self.train_variables], y)

    def predict(self, X):
        return self._trained_estimator.predict(X[self.train_variables])

    def predict_proba(self, X):
        return self._trained_estimator.predict_proba(X[self.train_variables])

    def staged_predict_proba(self, X):
        return self._trained_estimator.staged_predict_proba(X[self.train_variables])
