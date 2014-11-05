from __future__ import division, print_function, absolute_import
import copy
import pandas
from sklearn.tree.tree import DTYPE
from sklearn.utils.validation import check_random_state, column_or_1d, check_arrays

from hep_ml.commonutils import check_sample_weight, sigmoid_function
from hep_ml.ugradientboosting import AbstractLossFunction

from hep_ml.experiments.fasttree import FastTreeRegressor

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin

__author__ = 'Alex Rogozhnikov'


class FastGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 min_samples_split=2,
                 max_features=None,
                 max_depth=3,
                 max_events_used=1000,
                 update_tree=False,
                 criterion='mse',
                 train_variables=None,
                 random_state=None):
        """This version of gradient boosting supports only two-class classification and only special losses
        derived from AbstractLossFunction.
        :type loss: AbstractLossFunction
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_depth = max_depth
        self.max_events_used = max_events_used
        self.update_tree = update_tree
        self.criterion = criterion
        self.train_variables = train_variables
        self.random_state = random_state

    def check_params(self):
        assert isinstance(self.loss, AbstractLossFunction), \
            'LossFunction should be derived from AbstractLossFunction'
        assert self.n_estimators > 0, 'n_estimators should be positive'
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

        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)

        # preparing for fitting in trees
        X = self.get_train_vars(X)
        X, y = check_arrays(X, y, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        y_pred = numpy.zeros(len(X), dtype=float)

        for stage in range(self.n_estimators):
            # tree creation
            tree = FastTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state,
                max_events_used=self.max_events_used)

            # tree learning
            residual = self.loss.negative_gradient(y_pred)
            tree.fit(X, residual, sample_weight=sample_weight, check_input=False)

            # update tree leaves (not working ar this moment)
            # if self.update_tree:
            # self.loss.update_tree(tree.tree_, X=X, y=y, y_pred=y_pred, sample_weight=sample_weight,
            #                           update_mask=numpy.ones(len(X), dtype=bool), residual=residual)

            y_pred += self.learning_rate * tree.predict(X)
            self.estimators.append(tree)
            self.scores.append(self.loss(y_pred))
        return self

    def get_train_vars(self, X):
        if self.train_variables is None:
            return numpy.array(X)
        else:
            return numpy.array(X.loc[:, self.train_variables])

    def score_to_proba(self, score):
        result = numpy.zeros([len(score), 2], dtype=float)
        result[:, 1] = sigmoid_function(score, width=1.)
        result[:, 0] = 1. - result[:, 1]
        return result

    def staged_predict_score(self, X):
        X = self.get_train_vars(X)
        y_pred = numpy.zeros(len(X))
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
            yield self.score_to_proba(score)

    def predict_proba(self, X):
        return self.score_to_proba(self.predict_score(X))

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)
