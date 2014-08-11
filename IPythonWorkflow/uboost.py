"""
The module contains an implementation of uBoost algorithm in sklearn-way

- ``uBoostBDT`` is a modified version of AdaBoost, that targets to
obtain efficiency uniformity at the specified level
- ``uBoostClassifier`` - a combination of uBoostBDTs for different efficiencies
"""

# Authors:
# Alex Rogozhnikov <axelr@yandex-team.ru>
# Nikita Kazeev <kazeevn@yandex-team.ru>

from collections import defaultdict
from itertools import izip
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble.weight_boosting import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_arrays

from commonutils import compute_groups_efficiencies,\
    sigmoid_function, computeSignalKnnIndices, compute_bdt_cut, map_on_cluster


__author__ = "Alex Rogozhnikov, Nikita Kazeev"
__copyright__ = "Copyright 2014, Yandex"

__all__ = ["uBoostBDT", "uBoostClassifier"]


# TODO (Alex) update interface of knn function
# TODO (Alex) take weights into account when computing efficiencies



class uBoostBDT:
    def __init__(self,
                 uniform_variables,
                 target_efficiency=0.5,
                 n_neighbors=50,
                 bagging=True,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50,
                 learning_rate=1.,
                 uniforming_rate=1.,
                 separate_normalization=False,
                 train_variables=None,
                 smoothing=0.0,
                 keep_debug_info=False,
                 random_state=None,
                 algorithm="SAMME"):
        """
        uBoostBDT is AdaBoostClassifier, which is modified to have flat
        efficiency of signal (class=1) along some variables.
        Efficiency is only guaranteed at the cut,
        corresponding to global efficiency == target_efficiency.

        Can be used alone, without uBoost.

        Parameters
        ----------
        uniform_variables: list of strings, names of variables, along which
         flatness is desired

        target_efficiency: float, the flatness is obtained at global BDT cut,
            corresponding to global efficiency

        n_neighbours: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        bagging: float or bool (default=True), bagging usually speeds up the
            convergence and prevents overfitting
            (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
            if True, usual bootstrap aggregating is used
            (sampling with replacement at each iteration, size=len(X))
            if float, used sampling with replacement, the size of generated set
             is bagging * len(X)
            if False, usual boosting is used

        base_estimator : object, optional (default=DecisionTreeClassifier)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper
            `classes_` and `n_classes_` attributes.

        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by
            ``learning_rate``. There is a trade-off between ``learning_rate``
            and ``n_estimators``.

        uniforming_rate: float, optional (default=1.)
            how much do we take into account the uniformity of signal,
             there is a trade-off between uniforming_rate and the speed of
             uniforming, zero value corresponds to plain AdaBoost

        separate_normalization: bool (default=False),
            if True, the sum of weights of both signal and background
            will be normalized after each iteration to 0.5

        train_variables: list of strings, names of variables used in
           fit/predict. If None, all the variables are used
           (including uniform_variables)

        smoothing: float, default=(0.), used to smooth computing of local
           efficiencies, 0.0 corresponds to usual uBoost

        random_state: int, RandomState instance or None, (default=None)
            If int, random_state is the seed used by the
                random number generator;
            If RandomState instance, random_state is the
                random number generator;
            If None, the random number generator is the RandomState
            instance used by `np.random`.

        Attributes
        ----------
        `estimators_` : list of classifiers
            The collection of fitted sub-estimators.

        `estimator_weights_` : array of floats
            Weights for each estimator in the boosted ensemble.

        `estimator_errors_` : array of floats
            Classification error for each estimator in the boosted
            ensemble.

        Reference
        ----------
        .. [1] Justin Stevens, Mike Williams 'uBoost: A boosting method for
            producing uniform
            selection efficiencies from multivariate classifiers'
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.uniforming_rate = uniforming_rate
        self.uniform_variables = uniform_variables
        self.target_efficiency = target_efficiency
        self.n_neighbors = n_neighbors
        self.bagging = bagging
        self.separate_normalization = separate_normalization
        self.train_variables = train_variables
        self.smoothing = smoothing
        self.keep_debug_info = keep_debug_info
        self.random_state = random_state
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None, neighbours_matrix=None):
        """Build a boosted classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            The target values (integers that correspond to classes).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        neighbours_matrix: array-like of shape [n_samples, n_neighbours],
            each row contains indices of signal neighbours
            (neighbours should be computed for background too),
            if None, this matrix is computed.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.smoothing < 0:
            raise ValueError("Smoothing must be non-negative")
        if not isinstance(self.base_estimator, BaseEstimator):
            raise TypeError("estimator must be a subclass of BaseEstimator")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported"
                             % self.algorithm)

        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator, 'predict_proba'):
                raise TypeError(
                    "uBoostBDT with algorithm='SAMME.R' requires "
                    "that the weak learner have a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")

        assert np.in1d(y, [0, 1]).all(), \
            "only two-class classification is implemented"

        if neighbours_matrix is not None:
            assert np.shape(neighbours_matrix) == (len(X), self.n_neighbors), \
                "Wrong shape of neighbours_matrix"
            self.knn_indices = neighbours_matrix
        else:
            assert self.uniform_variables is not None,\
                "uniform_variables should be set"
            self.knn_indices = computeSignalKnnIndices(
                self.uniform_variables, X, y > 0.5, self.n_neighbors)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.ones(len(X), dtype=np.float) / len(X)
        else:
            # Normalize existing weights
            assert np.all(sample_weight >= 0.),\
                'the weights should be non-negative'
            sample_weight /= np.sum(sample_weight)

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = []
        # score cuts correspond to
        # global efficiency == target_efficiency on each iteration.
        self.score_cuts_ = []

        X_train_variables = self.get_train_vars(X)
        y = np.ravel(y)
        X_train_variables, y = check_arrays(
            X_train_variables, y, sparse_format="dense")

        # A dictionary to keep all intermediate weights, efficiencies and so on
        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)

        self.random_generator = check_random_state(self.random_state)

        self._boost(X_train_variables, y, sample_weight)

        self.score_cut = compute_bdt_cut(
            self.target_efficiency, y, self.predict_score(X))
        assert abs(self.score_cut - self.score_cuts_[-1] < 1e-4),\
            "score cut doesn't appear to coincide with the staged one"
        assert len(self.estimators_) == len(self.estimator_weights_) == len(self.score_cuts_)
        return self

    def _make_estimator(self):
        estimator = clone(self.base_estimator)
        # self.estimators_.append(estimator)
        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass
        return estimator

    def _estimator_score(self, estimator, X):
        if self.algorithm == "SAMME":
            return 2 * estimator.predict(X) - 1.
        else:
            p = estimator.predict_proba(X)
            p[p <= 0.] = 1e-5
            return np.log(p[:, 1] / p[:, 0])

    def _normalize_weight(self, y, weight):
        weight += 1e-30
        if self.separate_normalization:
            weight[y == 0] /= np.sum(weight[y == 0]) * 2
            weight[y == 1] /= np.sum(weight[y == 1]) * 2
        else:
            weight /= np.sum(weight)
        return weight

    def get_uboost_weights(self, sample_weight, score, y):
        """Returns uBoost multipliers to sample_weight
        and computed global cut"""
        global_score_cut = compute_bdt_cut(self.target_efficiency, y, score)

        local_efficiencies = compute_groups_efficiencies(
            global_score_cut, self.knn_indices, y, score,
            smoothing_width=self.smoothing * self.n_estimators)

        e_prime = np.average(np.abs(local_efficiencies - self.target_efficiency),
                             weights=sample_weight)

        # beta = np.log((1.0 - e_prime) / e_prime)
        # log(1. / e_prime), otherwise this can lead to the situation
        # where beta is negative (which is a disaster).
        # Mike (uboost author) said he didn't take that into account.
        beta = np.log(1. / e_prime)
        boost_weights = np.exp((self.target_efficiency - local_efficiencies) * y *
            (beta * self.uniforming_rate))

        if self.keep_debug_info:
            self.debug_dict['local_efficiencies'].append(
                local_efficiencies.copy())

        return boost_weights, global_score_cut

    def _boost(self, X, y, sample_weight):
        """Implement a single boost using the SAMME or SAMME.R algorithm,
        which is modified in uBoost way"""
        cumulative_score = np.zeros(len(X))
        y_signed = 2 * y - 1
        for iboost in xrange(self.n_estimators):
            estimator = self._make_estimator()
            mask = generate_mask(len(X), self.bagging, self.random_generator)
            estimator.fit(X, y, sample_weight=sample_weight * mask)

            # computing estimator weight
            if self.algorithm == 'SAMME':
                y_pred = estimator.predict(X)

                # Error fraction
                estimator_error = np.average(y_pred != y, weights=sample_weight)
                estimator_error = np.clip(estimator_error, 1e-6, 1 - 1e-6)

                estimator_weight = self.learning_rate * 0.5 * (
                    np.log((1. - estimator_error) / estimator_error))

                score = estimator_weight * (2 * y_pred - 1)
            else:
                estimator_weight = self.learning_rate * 0.5
                score = estimator_weight * self._estimator_score(estimator, X)

            # correcting the weights and score according to predictions
            sample_weight *= np.exp(- y_signed * score)
            cumulative_score += score

            uboost_multipliers, global_score_cut = \
                self.get_uboost_weights(sample_weight, cumulative_score, y)
            sample_weight *= uboost_multipliers
            sample_weight = self._normalize_weight(y, sample_weight)

            self.score_cuts_.append(global_score_cut)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)

            # assert np.allclose(cumulative_score, self.predict_score(X), atol=1e-4), \
            #     "wrong prediction"

            if self.keep_debug_info:
                self.debug_dict['weights'].append(sample_weight.copy())

        self.classes_ = estimator.classes_
        if not self.keep_debug_info:
            self.knn_indices = None

    def get_train_vars(self, X):
        """Gets the DataFrame and returns only columns
           that should be used in fitting / predictions"""
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]

    def predict_score(self, X):
        X = self.get_train_vars(X)
        score = np.zeros(len(X))
        for classifier, weight in zip(self.estimators_, self.estimator_weights_):
            score += self._estimator_score(classifier, X) * weight
        return score

    def staged_predict_score(self, X):
        X = self.get_train_vars(X)
        score = np.zeros(len(X))
        for classifier, weight in zip(self.estimators_, self.estimator_weights_):
            score += self._estimator_score(classifier, X) * weight
            yield score

    def score_to_proba(self, score):
        """Compute class probability estimates from decision scores."""
        proba = np.empty((score.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(score / self.n_estimators)
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

    def predict(self, X):
        """Predict classes for X.
        Parameters:
            X : array-like of shape = [n_samples, n_features],
            the input samples.

        Returns:
            y : array of shape = [n_samples], the predicted classes.
        """
        return np.array(self.predict_score(X) > self.score_cut, dtype=int)

    def _uboost_predict_score(self, X):
        """Method added specially for uBoostClassifier"""
        return sigmoid_function(self.predict_score(X) - self.score_cut,
                                self.smoothing)

    def _uboost_staged_predict_score(self, X):
        """Method added specially for uBoostClassifier"""
        for cut, score in zip(self.score_cuts_, self.staged_predict_score(X)):
            yield sigmoid_function(score - cut, self.smoothing)

    def predict_proba(self, X):
        return self.score_to_proba(self.predict_score(X))

    def staged_predict_proba(self, X):
        for score in self.staged_predict_score(X):
            yield self.score_to_proba(score)

    @property
    def feature_importances_(self):
        """Return the feature importances
           (the higher, the more important the feature).
        Returns:
            array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted,"
                             " call `fit` before `feature_importances_`.")

        return sum(tree.feature_importances_ * weight for tree, weight
                   in zip(self.estimators_, self.estimator_weights_))


def _train_classifier(classifier, X_train_vars, y, sample_weight, neighbours_matrix):
    return classifier.fit(X_train_vars, y, sample_weight=sample_weight,
                          neighbours_matrix=neighbours_matrix)


class uBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, uniform_variables=None,
                 n_neighbors=50,
                 efficiency_steps=20,
                 n_estimators=40,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 bagging=True,
                 train_variables=None,
                 algorithm="SAMME",
                 separate_normalization=False,
                 smoothing=None,
                 ipc_profile=None,
                 random_state=None):
        """uBoost classifier, am algorithm of boosting targeted to obtain
        flat efficiency in signal along some variables. See [1] for details.

        Parameters
        ----------
        uniform_variables: list of strings, names of variables,
            along which flatness is desired

        n_neighbours: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        efficiency_steps: integer, optional (default=20),
            How many uBoostBDTs should be trained
            (each with its own target_efficiency)

        base_estimator : object, optional (default=DecisionTreeClassifier)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required,
            as well as proper `classes_` and `n_classes_` attributes.

        bagging: float or bool (default=True), bagging usually
            speeds up the convergence and prevents overfitting
            (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
            if True, usual bootstrap aggregating is used
            (sampling with replacement at each iteration, size=len(X))
            if float, used sampling with replacement, the size of generated
               set is bagging * len(X)
            if False, usual boosting is used

        train_variables: list of strings,
            names of variables used in fit/predict.
            if None, all the variables are used (including uniform_variables)

        separate_normalization: bool (default=False),
            if True, the normalization of weight for signal and bg
            events is done independently

        smoothing: float, default=(0.), used to smooth computing of
            local efficiencies, 0.0 corresponds to usual uBoost

        random_state : int, RandomState instance or None, (default=None)
            optional
            If int, random_state is the seed used by
                the random number generator;
            If RandomState instance, random_state is
                the random number generator;
            If None, the random number generator is the RandomState
            instance used by `np.random`.

        parallel_profile: profile (name of cluster) in IPython
            to parallelize computations

        Reference
        ----------
        .. [1] Justin Stevens, Mike Williams 'uBoost: A boosting method
            for producing uniform
            selection efficiencies from multivariate classifiers'
        """
        self.uniform_variables = uniform_variables
        self.knn = n_neighbors
        self.efficiency_steps = efficiency_steps
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.bagging = bagging
        self.train_variables = train_variables
        self.separate_normalization = separate_normalization
        self.smoothing = smoothing
        self.ipc_profile = ipc_profile
        self.algorithm = algorithm

    def get_train_variables(self, X):
        if self.train_variables is not None:
            return X[self.train_variables]
        else:
            return X

    def fit(self, X, y, sample_weight=None):
        if self.uniform_variables is None:
            raise ValueError("Please set uniform variables")
        if len(self.uniform_variables) == 0:
            raise ValueError("The set of uniform variables cannot be empty")
        if len(X) != len(y):
            raise ValueError("Different size of X and y")

        X_train_vars = self.get_train_variables(X)

        if self.smoothing is None:
            self.smoothing = 0.2 / self.efficiency_steps

        neighbours_matrix = computeSignalKnnIndices(
            self.uniform_variables, X, y > 0.5, n_neighbors=self.knn)
        # TODO(Alex) select some other targets ?
        self.target_efficiencies = [(i + 1.0) / (self.efficiency_steps + 1.0)
                                    for i in range(self.efficiency_steps)]
        self.classifiers = []

        for efficiency in self.target_efficiencies:
            classifier = uBoostBDT(
                uniform_variables=self.uniform_variables, train_variables=None,
                target_efficiency=efficiency, n_neighbors=self.knn,
                n_estimators=self.n_estimators,
                base_estimator=self.base_estimator,
                random_state=self.random_state, bagging=self.bagging,
                smoothing=self.smoothing, algorithm=self.algorithm,
                separate_normalization=self.separate_normalization)
            self.classifiers.append(classifier)

        map_on_cluster(self.ipc_profile,
                       _train_classifier,
                       self.classifiers,
                       self.efficiency_steps * [X_train_vars],
                       self.efficiency_steps * [y],
                       self.efficiency_steps * [sample_weight],
                       self.efficiency_steps * [neighbours_matrix])

        self.classes_ = self.classifiers[0].classes_
        return self

    def predict(self, X):
        # TODO(kazeevn) Shall we sync behaviour with predict_proba and
        # return a list of predictions for all target_efficiencies
        return self.predict_proba(X).argmax(axis=1)

    def score_to_proba(self, score):
        proba = np.empty((len(score), 2), dtype=float)
        proba[:, 1] = expit(score / self.efficiency_steps)
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

    def predict_proba(self, X):
        X = self.get_train_variables(X)
        score = sum(clf._uboost_predict_score(X) for clf in self.classifiers)
        return self.score_to_proba(score)

    def staged_predict_proba(self, X):
        X = self.get_train_variables(X)
        for scores in izip(*[clf._uboost_staged_predict_score(X)
                             for clf in self.classifiers]):
            yield self.score_to_proba(sum(scores))


def generate_mask(n_samples, bagging=True, random_generator=np.random):
    """bagging: float or bool (default=True), bagging usually
        speeds up the convergence and prevents overfitting
        (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
        if True, usual bootstrap aggregating is used
           (sampling with replacement at each iteration, size=len(X))
        if float, used sampling with replacement, the size of generated
           set is bagging * len(X)
        if False, returns 1."""
    if bagging is True:
        indices = random_generator.randint(0, n_samples, n_samples)
        mask = np.bincount(indices, minlength=n_samples)
    elif isinstance(bagging, float):
        mask = random_generator.rand(n_samples) > 1 - bagging
    elif bagging is False:
        mask = 1.
    else:
        raise ValueError("something wrong was passed as bagging")
    return mask
