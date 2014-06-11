# About
# this module is implementation of optimized grid_search

from __future__ import division
from __future__ import print_function
from itertools import islice
import numpy
import pandas
import sklearn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import _check_param_grid
from sklearn.metrics.metrics import roc_auc_score
from sklearn.utils.random import check_random_state
import commonutils

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


__author__ = 'Alex Rogozhnikov'


def estimate_classifier(params_dict, base_estimator, X, y, folds, fold_checks,
                        score_function, sample_weight=None):
    kFolds = StratifiedKFold(y=y, n_folds=folds)
    score = 0.
    for train_indices, test_indices in islice(kFolds, fold_checks):
        trainX, trainY = X.irow(train_indices), y[train_indices]
        testX, testY = X.irow(test_indices), y[test_indices]
        estimator = sklearn.clone(base_estimator).set_params(**params_dict)
        if sample_weight is None:
            estimator.fit(X=trainX, y=trainY)
            proba = estimator.predict_proba(testX)
            score += score_function(testY, proba[:, 1])
        else:
            train_weights, test_weights = sample_weight[train_indices], sample_weight[test_indices]
            estimator.fit(X=trainX, y=trainY, sample_weight=train_weights)
            proba = estimator.predict_proba(testX)
            score += score_function(testY, proba[:, 1], sample_weight=test_weights)
    return score / fold_checks




class GridOptimalSearchCV(BaseEstimator, ClassifierMixin):
    """Optimal search over specified parameter values for an estimator. Metropolis-like algorithm is used

    Important members are fit, predict.

    GridSearchCV implements a "fit" method and a "predict" method like
    any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values. The closest values are considered

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, optional
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : integer or cross-validation generator, optional
        If an integer is passed, it is the number of folds (default 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    refit : boolean
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    n_evaluations : integer
        The number of attempts of evaluations


    Attributes
    ----------
    `grid_scores_` : list of named tuples
        Contains scores for all parameter combinations in param_grid.
        Each entry corresponds to one parameter setting.
        Each named tuple has the attributes:

            * ``parameters``, a dict of parameter settings
            * ``mean_validation_score``, the mean score over the
              cross-validation folds
            * ``cv_validation_scores``, the list of scores for each fold

    `best_estimator_` : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data.

    `best_score_` : float
        Score of best_estimator on the left out data.

    `best_params_` : dict
        Parameter setting that gave the best results on the hold out data.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    """

    def __init__(self, estimator, param_grid, score_function=None, folds=3, fold_checks=1, n_evaluations=40,
                 complete_cv=False, random_state=None, ipc_profile=None):
        """ipc_profile is name of IPython cluster profile, None for local training """
        self.base_estimator = estimator
        self.param_grid = param_grid
        self.dimensions = list([len(param_values) for param, param_values in param_grid.iteritems()])
        self.n_evaluations = n_evaluations
        self.grid_scores_ = OrderedDict()
        self.queued_tasks_ = set()
        self.score_function = score_function
        self.folds = folds
        self.fold_checks = fold_checks
        self.random_generator = random_state
        self.ipc_profile = ipc_profile

    def _check_params(self):
        assert isinstance(self.param_grid, OrderedDict), 'the passed param_grid should be of OrderedDict class'
        _check_param_grid(self.param_grid)
        size = numpy.prod(self.dimensions)
        assert size > 1, 'The space of parameters contains only %i points' % size
        # results on different parameters
        self.grid_scores_ = OrderedDict()
        # all the tasks that are being computed or already computed
        self.queued_tasks_ = set()
        self.dimensions = list([len(param_values) for param, param_values in self.param_grid.iteritems()])
        if self.score_function is None:
            self.score_function = roc_auc_score
        assert self.fold_checks <= self.folds, "We cannot have more checks than folds"
        self.random_generator = check_random_state(self.random_generator)
        self.n_evaluations = min(self.n_evaluations, size // 2)

    def _generate_start_point(self):
        while True:
            result = tuple([self.random_generator.randint(0, size) for size in self.dimensions])
            if result not in self.queued_tasks_:
                self.queued_tasks_.add(result)
                return result

    def _indices_to_parameters(self, state_indices):
        return OrderedDict([(name, values[i]) for i, (name, values) in zip(state_indices, self.param_grid.iteritems())])

    def _generate_next_point(self):
        """Generating next point in parameters space"""
        if len(self.grid_scores_) <= 2:
            return self._generate_start_point()
        results = numpy.array(self.grid_scores_.values())
        std = numpy.std(results) + 1e-5
        probabilities = numpy.exp(numpy.clip( (results - numpy.mean(results)) * 3. / std, -5, 5))
        probabilities /= numpy.sum(probabilities)
        while True:
            start = numpy.random.choice(len(probabilities), p=probabilities)
            start_indices = self.grid_scores_.keys()[start]
            axis = self.random_generator.randint(len(self.dimensions))
            new_state_indices = list(start_indices)[:] # copy
            new_state_indices[axis] += 1 if self.random_generator.uniform() > 0.5 else -1
            if new_state_indices[axis] < 0 or new_state_indices[axis] >= self.dimensions[axis]:
                continue
            new_state_indices = tuple(new_state_indices)
            if new_state_indices in self.queued_tasks_:
                continue
            self.queued_tasks_.add(new_state_indices)
            return new_state_indices

    @property
    def best_score_(self):
        return numpy.max(self.grid_scores_.values())

    @property
    def best_params_(self):
        return self._indices_to_parameters(max(self.grid_scores_.iteritems(), key=lambda x: x[1])[0])

    def _fit_best_estimator(self, X, y, sample_weight=None):
        # Training classifier once again
        self.best_estimator_ = sklearn.clone(self.base_estimator).set_params(**self.best_params_)
        if sample_weight is None:
            self.best_estimator_.fit(X, y)
        else:
            self.best_estimator_.fit(X, y, sample_weight=sample_weight)

    def fit(self, X, y, sample_weight=None, **params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
        """
        self._check_params()
        self.evaluations_done = 0
        X = pandas.DataFrame(X)
        if self.ipc_profile is None:
            while self.evaluations_done < self.n_evaluations:
                state_indices = self._generate_next_point()
                state_dict = self._indices_to_parameters(state_indices)
                self.grid_scores_[state_indices] = estimate_classifier(params_dict=state_dict,
                    base_estimator=self.base_estimator, X=X, y=y, folds=self.folds,
                    fold_checks=self.fold_checks, score_function=self.score_function, sample_weight=sample_weight)
                self.evaluations_done += 1
        else:
            from IPython.parallel import Client
            dview = Client(profile=self.ipc_profile).direct_view()
            portion = len(dview)
            print("There are {0} cores in cluster, the portion is equal {1}".format(len(dview), portion))
            while self.evaluations_done < self.n_evaluations:
                state_indices_array = [self._generate_next_point() for _ in range(portion)]
                state_dict_array = [self._indices_to_parameters(indices) for indices in state_indices_array]
                result = dview.map_sync(estimate_classifier, state_dict_array,
                    [self.base_estimator] * portion, [X]*portion, [y]*portion,
                    [self.folds] * portion, [self.fold_checks] * portion,
                    [self.score_function] * portion, [sample_weight]*portion)
                assert len(result) == portion, "The length of result is very strange"
                for state_indices, score in zip(state_indices_array, result):
                    self.grid_scores_[state_indices] = score
                self.evaluations_done += portion

        self._fit_best_estimator(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def print_results(self):
        for state_indices, value in self.grid_scores_.iteritems():
            state_string = ", ".join([d[0] + '=' + str(d[1]) for d in self._indices_to_parameters(state_indices).iteritems()])
            print("{0:.3f}:  {1}".format(value, state_string))







class TestClassifier(BaseEstimator, ClassifierMixin):
    """This classifier is created specially for testing optimization"""
    def __init__(self, a=1., b=1., c=1., d=1.):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def fit(self, X, y, sample_weight=None):
        pass

    def predict_proba(self, X):
        return numpy.zeros([len(X), 2]) + self.a * self.b * self.c * self.d


def MeanMetrics(y, pred, sample_weight=None):
    """ This metrics was created for testing purposes"""
    return numpy.mean(pred)


def test_optimization(size=10, n_evaluations=150):
    grid_1d = numpy.linspace(0.1, 1, num=size)
    grid = {'a': grid_1d, 'b': grid_1d, 'c': grid_1d, 'd': grid_1d}
    grid = OrderedDict(grid)

    grid_cv = GridOptimalSearchCV(TestClassifier(), grid, n_evaluations=n_evaluations, score_function=MeanMetrics)
    trainX, trainY = commonutils.generateSample(2000, 10, distance=0.5)
    grid_cv.fit(trainX, trainY)

    assert 0.8 <= grid_cv.best_score_ <= 1., 'Too poor optimization : %.2f' % grid_cv.best_score_
    assert MeanMetrics(trainY, grid_cv.predict_proba(trainX)[:, 1]) == grid_cv.best_score_, 'something is wrong'

test_optimization()


def test_grid_search():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    grid = {'base_estimator': [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4),
                               ExtraTreeClassifier(max_depth=4)],
            'learning_rate': [0.01, 0.1, 0.5, 1.],
            'n_estimators': [5, 10, 15, 20, 30, 40, 50, 75, 100, 125],
            'algorithm': ['SAMME', 'SAMME.R']}
    grid = OrderedDict(grid)

    trainX, trainY = commonutils.generateSample(2000, 10, distance=0.5)
    grid_cv = GridOptimalSearchCV(AdaBoostClassifier(), grid, n_evaluations=10)
    grid_cv.fit(trainX, trainY)
    grid_cv.predict_proba(trainX)
    grid_cv.predict(trainX)


test_grid_search()