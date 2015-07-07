from __future__ import print_function, division, absolute_import

import copy
import numpy
import pandas

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree.tree import DecisionTreeRegressor, DTYPE
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import column_or_1d

from .commonutils import check_sample_weight, sigmoid_function, check_arrays
from .losses import AbstractLossFunction, AdaLossFunction, AbstractFlatnessLossFunction, \
    KnnFlatnessLossFunction, BinFlatnessLossFunction, AbstractMatrixLossFunction, \
    SimpleKnnLossFunction, BinomialDevianceLossFunction


__author__ = 'Alex Rogozhnikov'


def score_to_proba(score):
    result = numpy.zeros([len(score), 2], dtype=float)
    result[:, 1] = sigmoid_function(score, width=1.)
    result[:, 0] = 1. - result[:, 1]
    return result


class uGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss=None,
                 n_estimators=10,
                 learning_rate=0.1,
                 subsample=1.,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 max_leaf_nodes=None,
                 max_depth=3,
                 init_estimator=None,
                 update_tree=False,
                 criterion='mse',
                 splitter='best',
                 train_variables=None,
                 random_state=None):
        """This version of gradient boosting supports only two-class classification and only special losses
        derived from AbstractLossFunction.
        :type loss: AbstractLossFunction
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.init_estimator = init_estimator
        self.update_tree = update_tree
        self.train_variables = train_variables
        self.random_state = random_state
        self.criterion = criterion
        self.splitter = splitter

    def check_params(self):
        assert isinstance(self.loss, AbstractLossFunction), \
            'LossFunction should be derived from AbstractLossFunction'
        assert self.n_estimators > 0, 'n_estimators should be positive'
        assert 0 < self.subsample <= 1., 'subsample should be in (0, 1]'
        self.random_state = check_random_state(self.random_state)

    def fit(self, X, y, sample_weight=None):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        assert len(X) == len(y), 'Different lengths of X and y'
        X = pandas.DataFrame(X)
        y = numpy.array(column_or_1d(y), dtype=int)
        assert numpy.all(numpy.in1d(y, [0, 1])), 'Only two-class classification supported'
        self.check_params()

        self.estimators = []
        self.scores = []

        n_samples = len(X)
        n_inbag = int(self.subsample * len(X))
        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)

        # preparing for fitting in trees
        X = self.get_train_vars(X)
        self.n_features = X.shape[1]
        X, y = check_arrays(X, y)
        X = X.astype(DTYPE)
        y_pred = numpy.zeros(len(X), dtype=float)

        if self.init_estimator is not None:
            y_signed = 2 * y - 1
            self.init_estimator.fit(X, y_signed, sample_weight=sample_weight)
            y_pred += numpy.ravel(self.init_estimator.predict(X))

        for stage in range(self.n_estimators):
            # tree creation
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes)

            # tree learning
            residual = self.loss.negative_gradient(y_pred)
            train_indices = self.random_state.choice(n_samples, size=n_inbag, replace=False)

            tree.fit(X[train_indices], residual[train_indices],
                     sample_weight=sample_weight[train_indices], check_input=False)
            # update tree leaves
            if self.update_tree:
                self.loss.update_tree(tree.tree_, X=X, y=y, y_pred=y_pred, sample_weight=sample_weight,
                                      update_mask=numpy.ones(len(X), dtype=bool), residual=residual)

            y_pred += self.learning_rate * tree.predict(X)
            self.estimators.append(tree)
            self.scores.append(self.loss(y_pred))
        return self

    def get_train_vars(self, X):
        if self.train_variables is None:
            return X
        else:
            return X.loc[:, self.train_variables]

    def staged_predict_score(self, X):
        X = self.get_train_vars(X)
        y_pred = numpy.zeros(len(X))
        if self.init_estimator is not None:
            y_pred += numpy.ravel(self.init_estimator.predict(X))

        for estimator in self.estimators:
            y_pred += self.learning_rate * estimator.predict(X)
            yield y_pred

    def predict_score(self, X):
        result = None
        for score in self.staged_predict_score(X):
            result = score
        return result

    def staged_predict_proba(self, X):
        for score in self.staged_predict_score(X):
            yield score_to_proba(score)

    def predict_proba(self, X):
        return score_to_proba(self.predict_score(X))

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)

    @property
    def feature_importances_(self):
        total_sum = sum(tree.feature_importances_ for tree in self.estimators)
        return total_sum / len(self.estimators)