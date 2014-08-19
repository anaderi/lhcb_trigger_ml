import numpy
import pandas
from scipy.special import expit
import sklearn
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.base import BaseEstimator, ClassifierMixin
from commonutils import check_sample_weight, computeKnnIndicesOfSameClass

# About

# In this module some very simple modifications of AdaBoost
# is presented. In weights boosting it uses mean of neighbours scores,


__author__ = 'Alex Rogozhnikov'

# TODO add some tests and assertions here
# TODO based on predict algorithm?


class MeanAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 uniform_variables=None,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=10,
                 learning_rate=0.5,
                 n_neighbours=10,
                 train_variables=None):
        self.uniform_variables = uniform_variables
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_neighbours = n_neighbours
        self.train_variables = train_variables

    def fit(self, X, y, sample_weight=None):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        assert len(X) == len(y)
        y = column_or_1d(y)
        y_signed = 2 * y - 1

        X = pandas.DataFrame(X)
        knn_indices = computeKnnIndicesOfSameClass(self.uniform_variables, X, y, self.n_neighbours)
        X = self.get_train_vars(X)
        cumulative_score = numpy.zeros(len(X))
        self.estimators = []
        for stage in range(self.n_estimators):
            classifier = sklearn.clone(self.base_estimator)
            classifier.fit(X, y, sample_weight=sample_weight)
            score = self.learning_rate * self.compute_score(classifier, X=X)
            cumulative_score += score
            sample_weight *= numpy.exp(- y_signed * numpy.take(score, knn_indices).mean(axis=1))
            self.estimators.append(classifier)

        assert numpy.allclose(cumulative_score, self.predict_score(X), atol=1e-5, rtol=1e-5)

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


