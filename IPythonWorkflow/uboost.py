"""
The module contains an implementation of uBoost algorithm in sklearn-way

- The ``uBoostBDT`` class is a modified version of AdaBoost, that targets to
obtain efficency uniformity at the specified level
- ``uBoostClassifier`` - a combination of uBoostBDTs for different efficiencies
"""

# Authors:
# Alex Rogozhnikov <axelr@yandex-team.ru>
# Nikita Kazeev <kazeevn@yandex-team.ru>

from collections import defaultdict
import math
from itertools import izip, islice
import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble.weight_boosting import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_arrays

from commonutils import compute_groups_efficiencies,\
    sigmoid_function, computeSignalKnnIndices, compute_bdt_cut


__author__ = "Alex Rogozhnikov, Nikita Kazeev"
__copyright__ = "Copyright 2014, Yandex"

__all__ = [
    "uBoostBDT",
    "uBoostClassifier"
    ]


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
                 boost_only_signal=True,
                 train_variables=None,
                 smoothing=0.0,
                 keep_debug_info=False,
                 random_state=None,
                 algorithm="SAMME",
                 uniforming_algorithm="uBoost"):
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

        target_efficiency: float/list of floats, the flatness is obtained at global BDT cut,
            corresponding to global efficiency. WARNING: list of floats makes sense only for
            uniforming_algorithm == 'mBoost'

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

        boost_only_signal: bool (default=True),
            if True, only weights of signal are changed depending on local
            efficiency (as in uBoost) if False, both weights of
             signal and background are changed

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

        algorithm: 'SAMME'/'SAMME.R' boosting algorithm to use

        uniforming_algorithm: 'uBoost'/'mBoost'. The algorithm to achieve
            efficency uniformity. uBoost is implemented as described in [1].
            mBoost is a modification, that substitutes target efficiency
            with median probability for signal, aiming to achive uniformity
            for all efficiencies with a single boosted classifier.


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
        self.boost_only_signal = boost_only_signal
        self.train_variables = train_variables
        self.smoothing = smoothing
        self.keep_debug_info = keep_debug_info
        self.random_state = random_state
        self.algorithm = algorithm
        self.uniforming_algorithm = uniforming_algorithm

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

        if self.uniforming_algorithm not in ('uBoost', 'mBoost'):
            raise ValueError("Unforming algorithm %s is not supported"
                             % self.uniforming_algorithm)

        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator, 'predict_proba'):
                raise TypeError(
                    "uBoostBDT with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")

        assert np.all((y == 0) | (y == 1)),\
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
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)
        # BDT cuts correspond to
        # global efficiency == target_efficiency on each iteration.
        self.bdt_cuts_ = []

        X_train_variables = self.get_train_vars(X)
        y = np.ravel(y)
        X_train_variables, y = check_arrays(
            X_train_variables, y, sparse_format="dense")

        # A dictionary to keep all intermediate weights, efficiencies and so on
        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)

        self.random_generator = check_random_state(self.random_state)

        if self.algorithm == "SAMME":
            self._boost_discrete(X_train_variables, y, sample_weight)
        elif self.algorithm == "SAMME.R":
            self._boost_real(X_train_variables, y, sample_weight)

        self.bdt_cut = compute_bdt_cut(
            self.target_efficiency, y, self.predict_proba(X)[:, 1])
        if self.uniforming_algorithm == 'uBoost':
            assert self.bdt_cut == self.bdt_cuts_[-1],\
                "BDT cut doesn't appear to coincide with the staged one"
        else:
            assert np.array_equal(self.bdt_cut, self.bdt_cuts_[-1]),\
                "BDT cut doesn't appear to coincide with the staged one"
        return self

    def _make_estimator(self, append=True):
        estimator = clone(self.base_estimator)
        self.estimators_.append(estimator)
        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass
        return estimator

    def get_uboost_weights(self, sample_weight, cumulative_score, y):
        """Returns uBoost multipliers to sample_weight"""
        # TODO(alex) think of weights,
        #   we should take weights into account when computing efficiencies
        # TODO(alex) separate normalization for classes?
        # global_cut2 = commonutils.compute_cut_for_efficiency(
        #     self.target_efficiency, y, cumulative_score)
        # cumulative_score2 = np.zeros([len(cumulative_score), 2])
        # cumulative_score2[:, 1] = cumulative_score
        # assert self.score_to_proba(
        #     np.array([global_cut2]))[0, 1] == global_cut, \
        #         ' cuts are different '
        # local_efficiencies2 = compute_groups_efficiencies(
        #    global_cut2, self.knn_indices, y, cumulative_score2,
        #    smoothing_width=self.smoothing)
        # assert np.all(local_efficiencies == local_efficiencies2),\
        #    'The computed efficiencies are different'
        real_proba = self.score_to_proba(
                cumulative_score, n_estimators=len(self.estimators_))

        if self.uniforming_algorithm == 'uBoost':
            target_efficiency = self.target_efficiency
        else:  # mBoost
            target_efficiency = np.median(real_proba[:, 1][y == 1])

        global_cut = compute_bdt_cut(
            target_efficiency, y, real_proba[:, 1])
        local_efficiencies = compute_groups_efficiencies(
            global_cut, self.knn_indices, y, real_proba,
            smoothing_width=self.smoothing)

        e_prime = np.sum(sample_weight * np.abs(
            local_efficiencies - target_efficiency))
        # beta = np.log((1.0 - e_prime) / e_prime)
        # log(1. / e_prime), otherwise this can lead to the situation
        # where beta is negative (which is a disaster).
        # Mike (uboost author) said he didn't take that into account.
        beta = np.log(1. / e_prime)
        if self.boost_only_signal:
            boost_weights = np.exp((
                target_efficiency - local_efficiencies) * y * (
                beta * self.uniforming_rate))
        else:
            boost_weights = np.exp((
                target_efficiency - local_efficiencies) * (
                beta * self.uniforming_rate))

        real_cut = compute_bdt_cut(
            self.target_efficiency, y, real_proba[:, 1])
        return boost_weights, real_cut

    def _boost_discrete(self, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm,
        which is modified in uBoost way"""
        cumulative_score = np.zeros(len(X))
        for iboost in xrange(self.n_estimators):
            estimator = self._make_estimator()
            #TODO(kazeevn) avoid memory waste if no bagging is needed
            estimator.fit(X, y, sample_weight=sample_weight * generate_mask(
                          len(X), self.bagging, self.random_generator))

            y_predict = estimator.predict(X)

            # Instances incorrectly classified
            incorrect = y_predict != y

            # Error fraction
            estimator_error = np.average(incorrect, weights=sample_weight)
            self.estimator_errors_[iboost] = estimator_error

            # Stop if classification is perfect
            if estimator_error <= 0:
                self.estimator_weights_[iboost] = 1
                return

            # Stop if the error is at least as bad as random guessing
            # (Removed by Alex)

            # Boost weight using AdaBoost SAMME algorithm
            estimator_weight = self.learning_rate * (
                np.log((1. - estimator_error) / estimator_error))
            self.estimator_weights_[iboost] = estimator_weight

            # correcting the weights according to predictions
            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight += 1e-30
            sample_weight /= np.sum(sample_weight)

            # computing score
            cumulative_score += (2 * y_predict - 1) * estimator_weight
            # assert np.all(cumulative_score == self.predict_score(X)), \
            # "wrong prediction"
            uboost_multipliers, global_cut = self.get_uboost_weights(
                    sample_weight, cumulative_score, y)

            sample_weight *= uboost_multipliers
            self.bdt_cuts_.append(global_cut)
            sample_weight /= np.sum(sample_weight)

            if self.keep_debug_info:
                self.debug_dict['weights'].append(sample_weight.copy())
                self.debug_dict['local_efficiencies'].append(
                    local_efficiencies.copy())

        if not self.keep_debug_info:
            self.knn_indices = None

    def _boost_real(self, X, y, sample_weight):
        """A single boost using the SAMME.R algorithm"""
        # We have only two classes and know that beforehand
        self.classes_ = np.array((0, 1))
        self.n_classes_ = 2
        score = np.zeros(len(X))
        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])
        for iboost in xrange(self.n_estimators):
            estimator = self._make_estimator()
            self.estimator_weights_[iboost] = 1.
            estimator.fit(X, y, sample_weight=sample_weight * generate_mask(
                len(X), self.bagging, self.random_generator))
            current_proba = estimator.predict_proba(X)
            samme_proba = self._samme_r_proba(current_proba, self.n_classes_)

            current_proba[current_proba <= 0] = 1e-5

            y_predict = self.classes_.take(
                np.argmax(current_proba, axis=1), axis=0)
            incorrect = y_predict != y
            estimator_error = np.mean(
                np.average(incorrect, weights=sample_weight, axis=0))

            boost_weight = (-1. * self.learning_rate
                                * (((self.n_classes_ - 1.) / self.n_classes_) *
                                   inner1d(y_coding, np.log(current_proba))))
            if not iboost == self.n_estimators - 1:
                sample_weight *= np.exp(boost_weight)
            sample_weight /= np.sum(sample_weight)
            score += 0.5 * samme_proba[:, 1]

            uboost_multipliers, global_cut = \
                self.get_uboost_weights(sample_weight, score, y)
            sample_weight *= uboost_multipliers
            self.bdt_cuts_.append(global_cut)
            sample_weight /= np.sum(sample_weight)

            if self.keep_debug_info:
                self.debug_dict['weights'].append(sample_weight.copy())
                self.debug_dict['local_efficiencies'].append(
                    local_efficiencies.copy())

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
        if self.algorithm == 'SAMME':
            score = np.zeros(len(X))
            for classifier, weight in zip(
                    self.estimators_, self.estimator_weights_):
                score += (2 * classifier.predict(X) - 1) * weight
        else:  # SAMME.R
            score = 0.5 * np.sum(self._samme_r_proba(
                estimator.predict_proba(X), self.n_classes_)[:, 1]
                for estimator in self.estimators_)
        return score

    def staged_predict_score(self, X):
        score = np.zeros(len(X))
        X = self.get_train_vars(X)
        for classifier, weight in zip(
                self.estimators_, self.estimator_weights_):
            if self.algorithm == "SAMME":
                score += (2 * classifier.predict(X) - 1) * weight
            else:
                score += 0.5 * self._samme_r_proba(
                    classifier.predict_proba(X), self.n_classes_)[:, 1]
            yield score

    def score_to_proba(self, score, old_result=None, n_estimators=None):
        """Compute class probability estimates from decision scores."""
        assert n_estimators is not None, \
            "Number of estimators is needed for normalization."
        if old_result is None:
            proba = np.empty((score.shape[0], 2), dtype=np.float64)
        else:
            proba = old_result

        proba[:, 1] = 1.0 / (1.0 + np.exp(
            -score.ravel() / self.estimator_weights_[:n_estimators].sum()))
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

    def predict(self, X):
        # TODO(kazeevn) CHECK
        """Predict classes for X.
        Parameters:
            X : array-like of shape = [n_samples, n_features],
            the input samples.

        Returns:
            y : array of shape = [n_samples], the predicted classes.
        """
        return np.array(self.predict_proba(X) > self.global_cut, dtype=int)

    @staticmethod
    def _samme_r_proba(proba,  n_classes):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

        References
        ----------
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/weight_boosting.py
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie,
        "Multi-class AdaBoost", 2009.
        """

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba <= 1e-5] = 1e-5
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])

    def predict_proba(self, X):
        return self.score_to_proba(
            self.predict_score(X), n_estimators=len(self.estimators_))

    def staged_predict_proba(self, X):
        for n_esimators, score in enumerate(self.staged_predict_score(X)):
            # +1 because enumerate begins with 0
            yield self.score_to_proba(score, n_estimators=(n_esimators+1))

    @property
    def feature_importances_(self):
        """Return the feature importances
           (the higher, the more important the feature).
        Returns:
            array, shape = [n_features]
        """
        assert self.algorithm == 'SAMME', \
            "Not implemneted for SAMME.R"

        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted,"
                             " call `fit` before `feature_importances_`.")

        return np.array(sum(tree.feature_importances_ * weight for tree, weight
                            in zip(self.estimators_, self.estimator_weights_))
                        / self.n_estimators)


def _train_classifier(
        classifier, X_train_vars, y, sample_weight, neighbours_matrix):
    return classifier.fit(
        X_train_vars, y, sample_weight=sample_weight,
        neighbours_matrix=neighbours_matrix)


class uBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, uniform_variables=None,
                 n_neighbors=50,
                 efficiency_steps=20,
                 n_estimators=40,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 bagging=True,
                 train_variables=None,
                 boost_only_signal=True,
                 smoothing=None,
                 random_state=None,
                 ipc_profile=None,
                 algorithm="SAMME",
                 uniforming_algorithm="uBoost"):
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

        boost_only_signal: bool (default=True),
            if True, only weights of signal are changed depending
                on local efficiency (as in uBoost)
            if False, both weights of signal and background are changed

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

        algorithm: 'SAMME'/'SAMME.R' - see uBoostBDT

        uniforming_algorithm: 'uBoost'/'mBoost' - see uBoostBDT. If
            mBoost is used, only one classifer is trained and its dbt_cuts_ contain
            lists of cuts for given target efficiencies.

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
        self.boost_only_signal = boost_only_signal
        self.smoothing = smoothing
        self.ipc_profile = ipc_profile
        self.algorithm = algorithm
        self.uniforming_algorithm=uniforming_algorithm

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

        if self.uniforming_algorithm == 'uBoost':
            self.classifiers = []

            for efficiency in self.target_efficiencies:
                classifier = uBoostBDT(
                    uniform_variables=self.uniform_variables, train_variables=None,
                    target_efficiency=efficiency, n_neighbors=self.knn,
                    n_estimators=self.n_estimators,
                    base_estimator=self.base_estimator,
                    random_state=self.random_state, bagging=self.bagging,
                    boost_only_signal=self.boost_only_signal,
                    smoothing=self.smoothing,
                    algorithm=self.algorithm,
                    uniforming_algorithm=self.uniforming_algorithm)
                self.classifiers.append(classifier)

            if self.ipc_profile is not None:
                from IPython.parallel import Client
                lb_view = Client(profile=self.ipc_profile).load_balanced_view()
                self.classifiers = lb_view.map_sync(
                    _train_classifier,
                    self.classifiers,
                    [X_train_vars] * self.efficiency_steps,
                    [y] * self.efficiency_steps,
                    [sample_weight] * self.efficiency_steps,
                    [neighbours_matrix] * self.efficiency_steps)
            else:
                self.classifiers = map(
                    _train_classifier,
                    self.classifiers,
                    [X_train_vars] * self.efficiency_steps,
                    [y] * self.efficiency_steps,
                    [sample_weight] * self.efficiency_steps,
                    [neighbours_matrix] * self.efficiency_steps)
        else:  # mBoost
            classifier = uBoostBDT(
                    uniform_variables=self.uniform_variables, train_variables=None,
                    target_efficiency=self.target_efficiencies,
                    n_neighbors=self.knn,
                    n_estimators=self.n_estimators,
                    base_estimator=self.base_estimator,
                    random_state=self.random_state, bagging=self.bagging,
                    boost_only_signal=self.boost_only_signal,
                    smoothing=self.smoothing,
                    algorithm=self.algorithm,
                    uniforming_algorithm=self.uniforming_algorithm)
            self.classifier = _train_classifier(
                classifier, X_train_vars, y, sample_weight, neighbours_matrix)
        return self

    def predict(self, X):
        # TODO(kazeevn) Shall we sync behaviour with predict_proba and
        # return a list of predictions for all target_efficiencies?
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        X_train_vars = self.get_train_variables(X)
        result = np.zeros([len(X), 2])
        if self.uniforming_algorithm == 'uBoost':
            for efficiency, classifier in izip(
                    self.target_efficiencies, self.classifiers):
                result[:, 1] += sigmoid_function(
                    classifier.predict_proba(
                        X_train_vars)[:, 1] - classifier.bdt_cut,
                    self.smoothing)
        else:  # mBoost
            for efficiency, bdt_cut in izip(
                    self.target_efficiencies, self.classifier.bdt_cut):
                result[:, 1] += sigmoid_function(
                self.classifier.predict_proba(
                    X_train_vars)[:, 1] - bdt_cut,
                self.smoothing)

        result[:, 1] /= self.efficiency_steps
        result[:, 0] = 1.0 - result[:, 1]
        return result

    def staged_predict_proba(self, X):
        X = self.get_train_variables(X)
        if self.uniforming_algorithm == 'uBoost':
            staged_probas = izip(* [
                bdt.staged_predict_proba(X) for bdt in self.classifiers])
            staged_cuts = izip(* [
                bdt.bdt_cuts_ for bdt in self.classifiers])
        else: #mBoost
            staged_probas = self.classifier.staged_predict_proba(X)
            staged_cuts = self.classifier.bdt_cuts_

        result = np.zeros([len(X), 2])

        for predictions, cuts in izip(staged_probas, staged_cuts):
            result[:, :] = 0.
            if self.uniforming_algorithm == 'uBoost':
                for proba, cut in izip(predictions, cuts):
                    result[:, 1] += sigmoid_function(proba[:, 1] - cut,
                                                    self.smoothing)
            else:  # mBoost
                for cut in cuts:
                    result[:, 1] += sigmoid_function(predictions[:, 1] - cut,
                                                    self.smoothing)
            result[:, 1] /= self.efficiency_steps
            result[:, 0] = 1.0 - result[:, 1]
            yield result


def generate_mask(n_samples, bagging=True, random_generator=np.random):
    """bagging: float or bool (default=True), bagging usually
        speeds up the convergence and prevents overfitting
        (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
        if True, usual bootstrap aggregating is used
        (sampling with replacement at each iteration, size=len(X))
        if float, used sampling with replacement, the size of generated
           set is bagging * len(X)
        if False, does nothing."""

    if bagging is True:
        indices = random_generator.randint(0, n_samples, n_samples)
        mask = np.bincount(indices, minlength=n_samples)
    elif isinstance(bagging, float):
        mask = random_generator.rand(n_samples) > 1 - bagging
    elif bagging is False:
        mask = np.ones(n_samples)
    else:
        assert False, "something wrong was passed as bagging"
    return mask
