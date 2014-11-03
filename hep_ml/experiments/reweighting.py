import numpy

from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree.tree import DecisionTreeClassifier

from .. import commonutils
from ..metrics import compute_group_efficiencies

__author__ = "Alex Rogozhnikov"


"""
About

A simple way of uniforming is tried here:
Each time we simply increase weights in the
regions with lower-than-average local efficiency

Pay attention that here we don't use boosting of any kind,
each time we just train new classifier with new weights.

This approach turned out to be very inefficient.

"""

class ReweightClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, uniform_variables, knn=50, iterations=10,
                 base_estimator=DecisionTreeClassifier(max_depth=6),
                 train_variables=None, learning_rate=10, efficiencies_as_sum=True):
        """This classifier tries to obtain flat efficiency in signal by
        changing the weights of training sample. Doesn't use boosting or whatever

        :type base_estimator: BaseEstimator
        """
        self.base_estimator = base_estimator
        self.uniform_variables = uniform_variables
        self.knn = knn
        self.iterations = iterations
        self.train_variables = train_variables
        self.learning_rate = learning_rate
        self.efficiencies_as_sum = efficiencies_as_sum

    def fit(self, X, y):
        assert len(X) == len(y), 'different length'
        assert self.iterations > 0, 'number of iterations should be positive'
        self.debug_dict = defaultdict(list)
        y = numpy.array(y > 0.5)
        knn_all_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, X,
                                                              is_signal=(y > -1), n_neighbors=self.knn)
        self.debug_dict['knn_all_indices'] = knn_all_indices
        weights = 1.0 / (numpy.take(y * 1.0, knn_all_indices).mean(axis=1) + 1e-8)
        bg_weight = numpy.mean(weights[y])
        weights[~y] = bg_weight
        weights /= numpy.sum(weights)

        knn_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, X, is_signal=y, n_neighbors=self.knn)
        X_train = self.get_train_variables(X)

        self.debug_dict['knn_indices'] = knn_indices

        estimator = None
        for iteration in range(1, self.iterations + 1):
            estimator = clone(self.base_estimator)
            estimator.fit(X_train, y, sample_weight=weights)
            predict_proba = estimator.predict_proba(X_train)

            if self.efficiencies_as_sum:
                # here we compute local efficiency as mean probability of signal among knn
                local_efficiencies = numpy.take(predict_proba[:, 1], knn_indices).mean(axis=1)
            else:
                # here we compute local efficiency at the cut, corresponding to global_efficiency=0.5
                global_cut = commonutils.compute_bdt_cut(0.5, y, predict_proba[:, 1])
                local_efficiencies = compute_group_efficiencies(predict_proba[:, 1], knn_indices,
                                                                cut=global_cut, sample_weight=None)
            mse = numpy.std(numpy.log(local_efficiencies))

            weights *= numpy.exp(- local_efficiencies * y * self.learning_rate * mse)
            bg_weight = numpy.mean(weights[y])
            weights[~y] = bg_weight
            weights /= numpy.sum(weights)
            self.debug_dict['weights'].append(weights.copy())
            self.debug_dict['local_efficiencies'].append(local_efficiencies.copy())
            self.debug_dict['estimators'].append(estimator)

        self.trained_estimator = estimator
        return self

    def get_train_variables(self, X):
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]

    def predict(self, X):
        X = self.get_train_variables(X)
        return self.trained_estimator.predict(X)

    def predict_proba(self, X):
        X = self.get_train_variables(X)
        return self.trained_estimator.predict_proba(X)

    def staged_predict_proba(self, X):
        X = self.get_train_variables(X)
        for estimator in self.debug_dict['estimators']:
            yield estimator.predict_proba(X)

    def inner_staged_predict_proba(self, X):
        X = self.get_train_variables(X)
        return self.trained_estimator.predict_proba(X)


def test_reweighting():
    trainX, trainY = commonutils.generate_sample(4000, 5, 2.0)
    testX, testY = commonutils.generate_sample(4000, 5, 2.0)

    reweighting = ReweightClassifier(uniform_variables=trainX.columns[:1],
                                     base_estimator=RandomForestClassifier(n_estimators=10),
                                     iterations=10, learning_rate=100)
    reweighting.fit(trainX, trainY)
    reweighting.predict(testX)
    reweighting.predict_proba(testX)
    reweighting.staged_predict_proba(testX)
    print "reweighting is ok"


test_reweighting()