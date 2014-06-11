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
    from collections import OrderedDict, defaultdict
except ImportError:
    from ordereddict import OrderedDict


__author__ = 'Alex Rogozhnikov'




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

    Examples
    --------
    >>> from sklearn import svm, grid_search, datasets
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = grid_search.GridSearchCV(svr, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=..., degree=..., gamma=...,
       kernel='rbf', max_iter=-1, probability=False, random_state=None,
       shrinking=True, tol=..., verbose=False),
           fit_params={}, iid=..., loss_func=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=..., score_func=...,
           scoring=..., verbose=...)


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

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a an hyperparameter grid.

    :func:`sklearn.cross_validation.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    """

    def __init__(self, estimator, param_grid, score_function=None, folds=3, fold_checks=1, n_evaluations=40,
                 complete_cv=False, random_state=None, pareto=False):
        """If complete_cv is False, only.
        If pareto = True, it optimizes in pareto style """
        self.base_estimator = estimator
        self.param_grid = param_grid
        self.dimensions = list([len(param_values) for param, param_values in param_grid.iteritems()])
        self.n_evaluations = n_evaluations
        # results on different parameters
        self.grid_scores_ = OrderedDict()
        self.score_function = score_function
        self.folds = folds
        self.fold_checks = fold_checks
        self.random_generator = random_state

    def check_params(self):
        assert isinstance(self.param_grid, OrderedDict), 'the passed param_grid should be of OrderedDict class'
        _check_param_grid(self.param_grid)
        size = numpy.prod(self.dimensions)
        assert size > 1, 'The space of parameters contains only %i points' % size
        # results on different parameters
        self.grid_scores_ = OrderedDict()
        self.dimensions = list([len(param_values) for param, param_values in self.param_grid.iteritems()])
        if self.score_function is None:
            self.score_function = roc_auc_score
        assert self.fold_checks <= self.folds, "We cannot have more checks than folds"
        self.random_generator = check_random_state(self.random_generator)
        self.n_evaluations = min(self.n_evaluations, size // 2)

    def _generate_first_point(self):
        return tuple([self.random_generator.randint(0, size - 1) for size in self.dimensions])

    def _indices_to_parameters(self, state_indices):
        return OrderedDict([(name, values[i]) for i, (name, values) in zip(state_indices, self.param_grid.iteritems())])

    def _generate_next_point(self):
        """Generating next point in parameters space"""
        results = numpy.array(self.grid_scores_.values())
        std = numpy.std(results)
        probabilities = numpy.exp(-results * 3. / std)
        probabilities /= numpy.sum(probabilities)
        start = numpy.random.choice(a=self.grid_scores_.keys(), p=probabilities)
        while True:
            axis = self.random_generator.randint(len(self.dimensions))
            new_state_indices = list(start)[:] # copy
            new_state_indices[axis] += 1 if self.random_generator.uniform() > 0.5 else -1
            if new_state_indices[axis] < 0 or new_state_indices[axis] >= self.dimensions[axis]:
                continue
            return tuple(new_state_indices)

    def _indices_after_jump(self, current_indices, new_indices):
        """Decides whether to jump in the new point or not, after we had computed the loss there """
        if not self.grid_scores_.has_key(current_indices):
            # this happens at first iteration
            return new_indices
        val = self.grid_scores_[current_indices]
        new_val = self.grid_scores_[new_indices]
        std = numpy.std(self.grid_scores_.values()) + 1e-10
        if self.random_generator.uniform() < numpy.exp((new_val - val) * 4. / std):
            return new_indices
        else:
            return current_indices

    @property
    def best_score_(self):
        return numpy.max(self.grid_scores_.values())

    @property
    def best_params_(self):
        return self._indices_to_parameters(max(self.grid_scores_.iteritems(), key=lambda x: x[1])[0])

    def fit(self, X, y, sample_weight=None, **params):
        """Run fit with all sets of parameters.
        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        """
        self.check_params()
        self.evaluations_done = 0
        state_indices = self._generate_first_point()
        X = pandas.DataFrame(X)
        while self.evaluations_done < self.n_evaluations:
            new_state_indices = self._generate_next_point()
            if self.grid_scores_.has_key(new_state_indices):
                state_indices = self._indices_after_jump(state_indices, new_state_indices)
                continue
            state_dict = self._indices_to_parameters(new_state_indices)
            self.evaluations_done += 1
            kFolds = StratifiedKFold(y=y, n_folds=self.folds)
            for train_indices, test_indices in islice(kFolds, self.fold_checks):
                trainX, trainY = X.irow(train_indices), y[train_indices]
                testX, testY = X.irow(test_indices), y[test_indices]
                estimator = sklearn.clone(self.base_estimator).set_params(**state_dict)
                score = 0.
                if sample_weight is None:
                    estimator.fit(X=trainX, y=trainY)
                    proba = estimator.predict_proba(testX)
                    score += self.score_function(testY, proba[:, 1])
                else:
                    train_weights, test_weights = sample_weight[train_indices], sample_weight[test_indices]
                    estimator.fit(X=trainX, y=trainY, sample_weight=train_weights)
                    proba = estimator.predict_proba(testX)
                    score += self.score_function(testY, proba[:, 1], sample_weight=test_weights)
                self.grid_scores_[new_state_indices] = score / self.fold_checks
                state_indices = self._indices_after_jump(state_indices, new_state_indices)

        # Training classifier once again
        self.best_estimator_ = sklearn.clone(self.base_estimator).set_params(**self.best_params_)
        if sample_weight is None:
            self.base_estimator.fit(X, y)
        else:
            self.base_estimator.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X, y, sample_weight=None):
        raise NotImplementedError()

    def predict(self, X, y, sample_weight=None):
        raise NotImplementedError()

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


def metric_functions(y, pred, sample_weight=None):
    return numpy.sum(pred[:, 1])




def test_optimization(size=10, n_evaluations=100):
    grid_1d = numpy.linspace(0, 1, num=size)
    grid = {'a': grid_1d, 'b': grid_1d, 'c': grid_1d, 'd': grid_1d}
    grid = OrderedDict(grid)

    grid_cv = GridOptimalSearchCV(TestClassifier(), grid, n_evaluations=n_evaluations)
    trainX, trainY = commonutils.generateSample(2000, 10, distance=0.5)
    grid_cv.fit(trainX, trainY)

    print(len(grid_cv.grid_scores_))
    print(grid_cv.best_params_)
    print(grid_cv.best_score_)
    grid_cv.print_results()
    print('Success!')
    # TODO assertions here

test_optimization()



def test_grid_search():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    grid = {'base_estimator': [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4),
                               ExtraTreeClassifier(max_depth=4)],
            'learning_rate': [0.01, 0.1, 0.5, 1.],
            'n_estimators': [5, 10, 15, 20, 30, 40, 50]}
    grid = OrderedDict(grid)

    trainX, trainY = commonutils.generateSample(2000, 10, distance=0.5)
    grid_cv = GridOptimalSearchCV(AdaBoostClassifier(), grid)
    grid_cv.fit(trainX, trainY)
    print(len(grid_cv.grid_scores_))
    print(grid_cv.best_params_)
    print(grid_cv.best_score_)
    grid_cv.print_results()
    print('Success!')




test_grid_search()