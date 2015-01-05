from __future__ import division, print_function, absolute_import
import copy
import numpy
import pandas
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree.tree import DTYPE
from sklearn.utils.validation import check_random_state, column_or_1d, check_arrays
from sklearn.base import clone, BaseEstimator, ClassifierMixin

from ..commonutils import check_sample_weight, sigmoid_function
from ..losses import AbstractLossFunction

from .fasttree import FastTreeRegressor, FastNeuroTreeRegressor


__author__ = 'Alex Rogozhnikov'

# TODO where to include sample_weight - in the loss, or in the tree.fit, introduce special parameter
# TODO strategies

# Possible strategies
# subsample + update on all / update on same / update on other,
# kFold (fitting on all but one fold, prediction on the one wasn't used in training),


class AbstractGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 train_variables=None,
                 random_state=None):
        """This version of gradient boosting supports only two-class classification and only special losses
        derived from AbstractLossFunction.
        There are some methods that should be overriden in descendants.
        :type loss: AbstractLossFunction
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.train_variables = train_variables
        self.random_state = random_state
        self.initial_prediction = 0.

    def _check_params(self):
        # Losses from sklearn are nor allowed
        assert isinstance(self.loss, AbstractLossFunction), \
            'LossFunction should be derived from AbstractLossFunction'
        assert self.n_estimators > 0, 'n_estimators should be positive'
        self.random_state = check_random_state(self.random_state)
        assert 0 < self.subsample <= 1.0, 'subsample should be in the interval (0, 1]'

    def _create_estimator(self, stage):
        raise NotImplementedError('Should be overriden in descendants')

    def _fit_estimator(self, estimator, X, y, sample_weight, residual, mask):
        """ mask - which events to use in training """
        estimator.fit(X[mask, :], residual[mask], sample_weight=sample_weight[mask])

    def _update_estimator(self, estimator, X, y, sample_weight, residual, y_pred, mask):
        pass

    def _prepare_data_for_fitting(self, X, y, sample_weight):
        """By default the same format used as for trees """
        X = self.get_train_vars(X)
        X, y = check_arrays(X, y, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        return X, y, sample_weight

    @staticmethod
    def _initial_data_check(X, y, sample_weight):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        assert len(X) == len(y), 'Different lengths of X and y'
        X = pandas.DataFrame(X)
        y = numpy.array(column_or_1d(y), dtype=int)
        assert numpy.all(numpy.in1d(y, [0, 1])), 'Only two-class classification supported'
        return X, y, sample_weight

    def _prepare_initial_predictions(self, X, y, sample_weight):
        self.initial_prediction = 0.

    def _generate_mask(self, length, subsample):
        if subsample == 1.0:
            return slice(None, None, None)
        else:
            n_sampled_events = int(subsample * length)
            return self.random_state.choice(length, n_sampled_events, replace=True)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self._initial_data_check(X, y, sample_weight)
        self._check_params()

        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)

        X, y, sample_weight = self._prepare_data_for_fitting(X, y, sample_weight)

        self._prepare_initial_predictions(X, y, sample_weight)
        y_pred = numpy.zeros(len(X), dtype=float) + self.initial_prediction
        self.estimators = []
        self.scores = []

        for stage in range(self.n_estimators):
            # estimator creation
            estimator = self._create_estimator(stage)

            # estimator learning
            residual = self.loss.negative_gradient(y_pred)
            train_mask = self._generate_mask(len(X), subsample=self.subsample)
            self._fit_estimator(estimator, X, y, sample_weight, residual, mask=train_mask)

            # update estimator
            update_mask = numpy.ones(len(y), dtype=bool)
            self._update_estimator(estimator, X, y, sample_weight, residual, y_pred, mask=update_mask)

            # updating training state
            y_pred += self.learning_rate * estimator.predict(X)
            self.estimators.append(estimator)
            self.scores.append(self.loss(y_pred))
        return self

    def get_train_vars(self, X):
        if self.train_variables is None:
            return numpy.array(X)
        else:
            return numpy.array(X.loc[:, self.train_variables])

    @staticmethod
    def score_to_proba(score):
        result = numpy.zeros([len(score), 2], dtype=float)
        result[:, 1] = sigmoid_function(score, width=1.)
        result[:, 0] = 1. - result[:, 1]
        return result

    def staged_predict_score(self, X):
        X = self.get_train_vars(X)
        y_pred = numpy.zeros(len(X)) + self.initial_prediction
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


class CommonGradientBoosting(AbstractGradientBoostingClassifier):
    """
    This classifier can use gradient boosting upon any scikit-learn regressor
    """

    def __init__(self, loss=None,
                 base_estimator=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 train_variables=None,
                 dtype='float',
                 random_state=None):
        self.base_estimator = base_estimator
        self.dtype = dtype
        AbstractGradientBoostingClassifier.__init__(self, loss=loss,
                                                    n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    subsample=subsample,
                                                    train_variables=train_variables,
                                                    random_state=random_state)

    def _prepare_data_for_fitting(self, X, y, sample_weight):
        """By default the same format used as for trees """
        X = self.get_train_vars(X)
        X, y = check_arrays(X, y, dtype=self.dtype, sparse_format="dense", check_ccontiguous=True)
        return X, y, sample_weight

    def _create_estimator(self, stage):
        return clone(self.base_estimator)


class AbstractFoldingGBClassifier(AbstractGradientBoostingClassifier):
    """
    Uses kFolding, built upon any classifier.
    """

    def __init__(self, loss=None,
                 base_estimator=None,
                 n_folds=3,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 train_variables=None,
                 random_state=None):
        self.base_estimator = base_estimator
        self.n_folds = n_folds
        AbstractGradientBoostingClassifier.__init__(self, loss=loss,
                                                    n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    subsample=subsample,
                                                    train_variables=train_variables,
                                                    random_state=random_state)

    def _create_estimator(self, stage):
        return clone(self.base_estimator)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self._initial_data_check(X, y, sample_weight=sample_weight)
        self._check_params()

        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)

        X, y, sample_weight = self._prepare_data_for_fitting(X, y, sample_weight)

        y_pred = numpy.zeros(len(X), dtype=float)
        self.estimators = []
        self.scores = []

        for stage in range(self.n_estimators):
            stage_estimators = []
            for fold, (train_indices, test_indices) in enumerate(
                    StratifiedKFold(y, n_folds=self.n_folds, shuffle=True, random_state=stage)):
                # estimator creation
                estimator = self._create_estimator(stage)

                # estimator learning
                residual = self.loss.negative_gradient(y_pred)
                train_mask = self._generate_mask(len(X), self.subsample)
                self._fit_estimator(estimator, X, y, sample_weight, residual, mask=train_mask)

                # update estimator
                self._update_estimator(estimator, X, y, sample_weight, residual, y_pred, mask=train_indices)

                # updating training state
                y_pred[test_indices] += self.learning_rate * estimator.predict(X[test_indices, :])

                stage_estimators.append(estimator)
            self.estimators.append(stage_estimators)
            self.scores.append(self.loss(y_pred))
        return self

    def staged_predict_score(self, X):
        """ Uses mean predictions"""
        X = self.get_train_vars(X)
        y_pred = numpy.zeros(len(X))
        for stage_estimators in self.estimators:
            for estimator in stage_estimators:
                y_pred += (self.learning_rate - self.n_folds) * estimator.predict(X)
            yield y_pred


class FastGradientBoostingClassifier(AbstractGradientBoostingClassifier):
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
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_depth = max_depth
        self.max_events_used = max_events_used
        self.update_tree = update_tree
        self.criterion = criterion

        AbstractGradientBoostingClassifier.__init__(self, loss=loss,
                                                    n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    subsample=1.0,
                                                    train_variables=train_variables,
                                                    random_state=random_state)

    def check_params(self):
        assert isinstance(self.loss, AbstractLossFunction), \
            'LossFunction should be derived from AbstractLossFunction'
        assert self.n_estimators > 0, 'n_estimators should be positive'
        self.random_state = check_random_state(self.random_state)

    def _create_estimator(self, stage):
        return FastTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            max_events_used=self.max_events_used)

    def _fit_estimator(self, estimator, X, y, sample_weight, residual, mask):
        # mask is ignored
        estimator.fit(X, residual, sample_weight=sample_weight, check_input=False)

    def _update_estimator(self, estimator, X, y, sample_weight, residual, y_pred, mask):
        # mask is ignored
        if self.update_tree:
            self.loss.update_fast_tree(fast_tree=estimator,
                                       X=X, y=y, y_pred=y_pred, sample_weight=sample_weight,
                                       update_mask=numpy.ones(len(X), dtype=bool), residual=residual)


class FastNeuralGradientBoostingClassifier(FastGradientBoostingClassifier):
    def __init__(self, loss=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 min_samples_split=2,
                 max_features=None,
                 max_depth=3,
                 max_events_used=1000,
                 n_lincomb=2,
                 update_tree=False,
                 criterion='mse',
                 train_variables=None,
                 random_state=None):
        """This version of gradient boosting supports only two-class classification and only special losses
        derived from AbstractLossFunction.
        :type loss: AbstractLossFunction
        """
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_depth = max_depth
        self.max_events_used = max_events_used
        self.update_tree = update_tree
        self.criterion = criterion
        self.n_lincomb = n_lincomb

        AbstractGradientBoostingClassifier.__init__(self, loss=loss,
                                                    n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    subsample=subsample,
                                                    train_variables=train_variables,
                                                    random_state=random_state)

    def _create_estimator(self, stage):
        return FastNeuroTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            max_events_used=self.max_events_used,
            n_lincomb=self.n_lincomb)


class CategoricalGradientBoosting(AbstractGradientBoostingClassifier):
    """
    Specially for Avazu
    """
    def __init__(self, loss=None,
                 base_estimator=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 train_variables=None,
                 dtype='int',
                 update_tree=True,
                 random_state=None):
        self.base_estimator = base_estimator
        self.dtype = dtype
        self.update_tree = update_tree
        AbstractGradientBoostingClassifier.__init__(self, loss=loss,
                                                    n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    subsample=subsample,
                                                    train_variables=train_variables,
                                                    random_state=random_state)

    def _prepare_initial_predictions(self, X, y, sample_weight):
        from scipy.special import logit
        self.initial_prediction = logit(numpy.average(y, weights=sample_weight))

    def _prepare_data_for_fitting(self, X, y, sample_weight):
        """By default the same format used as for trees """
        X = self.get_train_vars(X)
        X, y = check_arrays(X, y, dtype=self.dtype, sparse_format="dense", check_ccontiguous=True, )
        X = numpy.array(X, dtype=self.dtype, order='F')
        return X, y, sample_weight

    def _create_estimator(self, stage):
        return clone(self.base_estimator)

    def _update_estimator(self, estimator, X, y, sample_weight, residual, y_pred, mask):
        # mask is ignored
        if self.update_tree:
            self.loss.update_fast_tree(fast_tree=estimator,
                                       X=X, y=y, y_pred=y_pred, sample_weight=sample_weight,
                                       update_mask=numpy.ones(len(X), dtype=bool), residual=residual)
