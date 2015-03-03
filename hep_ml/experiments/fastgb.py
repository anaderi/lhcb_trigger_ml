from __future__ import division, print_function, absolute_import
import copy
import numpy
import pandas
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.tree.tree import DTYPE
from sklearn.utils.validation import check_random_state, column_or_1d, check_arrays
from sklearn.base import clone, BaseEstimator, ClassifierMixin

from ..commonutils import check_sample_weight, sigmoid_function
from hep_ml.losses import AdaLossFunction
from ..losses import AbstractLossFunction

from .fasttree import FastTreeRegressor, FastNeuroTreeRegressor
from scipy.special import logit
from multiprocessing.pool import ThreadPool
from threading import Lock


__author__ = 'Alex Rogozhnikov'

# TODO where to include sample_weight - in the loss, or in the tree.fit, introduce special parameter


def _train_one_classifier(train_params):
    """
    Supplementary function to train one classifier, used in parallel versions of boosting.
    """
    self, X, y, sample_weight, y_pred, lock = train_params

    # estimator creation
    estimator = self._create_estimator(len(self.estimators))

    # estimator learning
    residual = self.loss.negative_gradient(y_pred)
    train_mask = self._generate_mask(len(X), subsample=self.subsample)
    self._fit_estimator(estimator, X, y, sample_weight, residual, mask=train_mask)

    # update estimator
    update_mask = numpy.ones(len(y), dtype=bool)
    self._update_estimator(estimator, X, y, sample_weight, residual, y_pred, mask=update_mask)

    # updating training state
    stage_pred = self.learning_rate * estimator.predict(X)
    with lock:
        y_pred += stage_pred
        self.estimators.append(estimator)
    self.scores.append(self.loss(y_pred))


def _train_kfold_classifier(train_params):
    self, X, y, sample_weight, y_pred, residual, train_indices, test_indices, lock = train_params

    # splitting train on real train and update
    if self.subsample < 0.5:
        train_train_indices, train_update_indices = train_test_split(train_indices, train_size=self.subsample)
    else:
        train_train_indices = train_indices[self._generate_mask(len(train_indices), subsample=self.subsample)]
        train_update_indices = train_indices

    # estimator creation
    estimator = self._create_estimator(len(self.estimators))

    # estimator learning
    self._fit_estimator(estimator, X, y, sample_weight, residual, mask=train_train_indices)

    # update estimator
    self._update_estimator(estimator, X, y, sample_weight, residual, y_pred, mask=train_update_indices)

    # updating training state
    test_pred = estimator.predict(X[test_indices])

    # rest is done in classifier
    return estimator, test_indices, test_pred


class AbstractGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 train_variables=None,
                 random_state=None,
                 n_threads=1,
                 dtype=DTYPE):
        """This version of gradient boosting supports only two-class classification and only special losses
        derived from AbstractLossFunction.
        There are some methods that should be overriden in descendants.
        :type loss: AbstractLossFunction, by default AdaLossFunction is used
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.train_variables = train_variables
        self.random_state = random_state
        self.initial_prediction = 0.
        self.dtype = dtype
        self.n_threads = n_threads

    def _check_params(self):
        if self.loss is None:
            self.loss = AdaLossFunction()
        # Losses from sklearn are not allowed
        assert isinstance(self.loss, AbstractLossFunction), \
            'LossFunction should be derived from AbstractLossFunction'
        assert self.n_estimators > 0, 'n_estimators should be positive'
        self.random_state = check_random_state(self.random_state)
        assert 0 < self.subsample <= 1.0, 'subsample should be in the interval (0, 1]'

    def _create_estimator(self, stage):
        raise NotImplementedError('Should be overriden in descendants')

    def _fit_estimator(self, estimator, X, y, sample_weight, residual, mask):
        """ mask - which events to use in training """
        # TODO do we need check_input=false for trees?
        estimator.fit(X[mask, :], residual[mask], sample_weight=sample_weight[mask])

    def _update_estimator(self, estimator, X, y, sample_weight, residual, y_pred, mask):
        pass

    def _prepare_data_for_fitting(self, X, y, sample_weight):
        """By default the same format used as for trees """
        X = self.get_train_vars(X)
        X, y = check_arrays(X, y, dtype=self.dtype, sparse_format="dense", check_ccontiguous=True)
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
        self.initial_prediction = logit(numpy.average(y, weights=sample_weight))

    def _compute_initial_predictions(self, X):
        return numpy.zeros(len(X), dtype='float') + self.initial_prediction

    def _generate_mask(self, length, subsample):
        if subsample == 1.0:
            return slice(None, None, None)
        else:
            n_sampled_events = int(subsample * length)
            return self.random_state.choice(length, n_sampled_events, replace=True)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self._initial_data_check(X, y, sample_weight)
        self._check_params()

        loss_weight = numpy.ones(len(sample_weight))
        tree_weight = sample_weight

        if False:
            loss_weight, tree_weight = tree_weight, loss_weight


        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=loss_weight)

        X, y, sample_weight = self._prepare_data_for_fitting(X, y, sample_weight)

        self._prepare_initial_predictions(X, y, sample_weight)
        y_pred = self._compute_initial_predictions(X)
        self.estimators = []
        self.scores = []

        # pool = ThreadPool(processes=self.n_threads)

        lock = Lock()
        train_params = [self, X, y, tree_weight, y_pred, lock]
        # TODO use threading
        # pool.map(_train_one_classifier, [train_params] * self.n_estimators, chunksize=1)
        map(_train_one_classifier, [train_params] * self.n_estimators)

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
        y_pred = self._compute_initial_predictions(X)
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
    This classifier runs gradient boosting upon any sklearn regressor
    (passed as base_estimator)
    """

    def __init__(self, loss=None,
                 base_estimator=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 train_variables=None,
                 dtype='float',
                 n_threads=1,
                 random_state=None):
        self.base_estimator = base_estimator
        AbstractGradientBoostingClassifier.__init__(self, loss=loss,
                                                    n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    subsample=subsample,
                                                    train_variables=train_variables,
                                                    random_state=random_state,
                                                    n_threads=n_threads,
                                                    dtype=dtype)

    def _create_estimator(self, stage):
        return clone(self.base_estimator)


class FoldingGBClassifier(CommonGradientBoosting):
    def __init__(self, loss=None,
                 base_estimator=None,
                 n_folds=2,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.0,
                 train_variables=None,
                 n_threads=1,
                 update_tree=False,
                 random_state=None):
        """
        :param loss: loss function used
        :param base_estimator: BaseEstimator
        :param n_folds:
        :param n_estimators:
        :param subsample: used in fitting classifiers
        :param train_variables:
        :param n_threads:
        :param update_tree: bool,
        """
        self.n_folds = n_folds
        self.update_tree = update_tree
        CommonGradientBoosting.__init__(self, loss=loss,
                                        base_estimator=base_estimator,
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        subsample=subsample,
                                        train_variables=train_variables,
                                        n_threads=n_threads,
                                        random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self._initial_data_check(X, y, sample_weight=sample_weight)
        self._check_params()

        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)

        X, y, sample_weight = self._prepare_data_for_fitting(X, y, sample_weight)

        y_pred = numpy.zeros(len(X), dtype=float)
        self.estimators = []
        self.scores = []

        pool = ThreadPool(processes=self.n_threads)
        lock = Lock()

        for stage in range(self.n_estimators):
            stage_estimators = []

            train_params = []
            residual = self.loss.negative_gradient(y_pred)
            for fold, (train_indices, test_indices) in enumerate(
                    StratifiedKFold(y, n_folds=self.n_folds, shuffle=True, random_state=stage)):
                train_params.append([self, X, y, sample_weight, y_pred,
                                     residual, train_indices, test_indices, lock])
            # TODO use multithreading
            # result = pool.map(_train_kfold_classifier, train_params, chunksize=1)
            result = map(_train_kfold_classifier, train_params)
            for estimator, test_indices, test_prediction in result:
                stage_estimators.append(estimator)
                y_pred[test_indices] += self.learning_rate * test_prediction

            self.estimators.append(stage_estimators)
            self.scores.append(self.loss(y_pred))

        return self

    def staged_predict_score(self, X):
        """ Uses mean predictions, TODO use same folding """
        X = self.get_train_vars(X)
        y_pred = numpy.zeros(len(X))
        for stage_estimators in self.estimators:
            for estimator in stage_estimators:
                y_pred += self.learning_rate * estimator.predict(X) / self.n_folds
            yield y_pred

    def _update_estimator(self, estimator, X, y, sample_weight, residual, y_pred, mask):
        if self.update_tree:
            self.loss.update_fast_tree(fast_tree=estimator,
                                       X=X, y=y, y_pred=y_pred, sample_weight=sample_weight,
                                       update_mask=mask, residual=residual)


class TreeGradientBoostingClassifier(CommonGradientBoosting):
    def __init__(self, loss=None,
                 base_estimator=FastTreeRegressor(),
                 n_estimators=100,
                 subsample=1.0,
                 learning_rate=0.1,
                 update_tree=True,
                 train_variables=None,
                 n_threads=1,
                 dtype=DTYPE,
                 random_state=None):
        '''
        :param base_estimator: descendant of FastTreeRegressor
        :param update_tree: if True, will update values in leaves to minimize loss function
        '''
        self.update_tree = update_tree
        CommonGradientBoosting.__init__(self, loss=loss, base_estimator=base_estimator,
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        train_variables=train_variables,
                                        dtype=dtype,
                                        n_threads=n_threads,
                                        subsample=subsample,
                                        random_state=random_state)

    def _fit_estimator(self, estimator, X, y, sample_weight, residual, mask):
        estimator.fit(X, residual, sample_weight=sample_weight, check_input=False)

    def _update_estimator(self, estimator, X, y, sample_weight, residual, y_pred, mask):
        if self.update_tree:
            self.loss.update_fast_tree(fast_tree=estimator,
                                       X=X, y=y, y_pred=y_pred, sample_weight=sample_weight,
                                       update_mask=mask, residual=residual)

    def refit_trees(self, X, y, sample_weight=None, loss=None, learning_rate=None,
                    subsample=0.5, forgetting_noise=0.1):
        X, y, sample_weight = self._initial_data_check(X, y, sample_weight)
        if learning_rate is None:
            learning_rate = self.learning_rate

        if loss is None:
            loss = self.loss
        loss = copy.copy(loss)
        loss.fit(X, y, sample_weight=sample_weight)

        X, y, sample_weight = self._prepare_data_for_fitting(X, y, sample_weight)

        y_pred = self._compute_initial_predictions(X)
        for estimator in self.estimators:
            y_pred += learning_rate * estimator.predict(X)

        for estimator in self.estimators:
            y_pred -= learning_rate * estimator.predict(X)
            visual_pred = y_pred * numpy.exp(- numpy.random.exponential(size=len(y_pred)) * forgetting_noise)
            loss.update_fast_tree(fast_tree=estimator,
                                  X=X, y=y, y_pred=visual_pred, sample_weight=sample_weight,
                                  update_mask=self._generate_mask(len(y), subsample=subsample),
                                  residual=loss.negative_gradient(visual_pred))
            y_pred += learning_rate * estimator.predict(X)

