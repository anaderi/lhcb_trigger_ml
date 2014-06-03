from collections import defaultdict
import math
from matplotlib.cbook import Null

import numpy
from numpy.lib._compiled_base import bincount
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble.weight_boosting import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_arrays

try:
    import cPickle as pickle
except:
    import pickle

from commonutils import computeLocalEfficiencies, computeBDTCut, sigmoidFunction, \
    generateSample, computeSignalKnnIndices

__author__ = 'Alex Rogozhnikov'



class uBoostBDT:
    def __init__(self,
                 uniform_variables,
                 target_efficiency=0.5,
                 n_neighbors=50,
                 bagging=True,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50,
                 learning_rate=1.,
                 uboosting_rate=1.,
                 boost_only_signal=True,
                 train_variables=None,
                 smoothing=0.0,
                 keep_debug_info=False,
                 random_state=None):
        """
        uBoostBDT is AdaBoostClassifier, which is modified to have flat efficiency
        of signal (class=1) along some variables. Efficiency is only guaranteed at the cut,
        corresponding to global efficiency == target_efficiency.

        Can be used alone, without uBoost.

        Parameters
        ----------
        uniform_variables: list of strings, names of variables, along which flatness is desired

        target_efficiency: float, the flatness is obtained at global BDT cut,
            corresponding to global efficiency

        n_neighbours: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        bagging: float or bool (default=True), bagging usually speeds up the convergence
            and prevents overfitting (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
            if True, usual bootstrap aggregating is used (sampling with replacement at each iteration, size=len(X))
            if float, used sampling with replacement, the size of generated set is bagging * len(X)
            if False, usual boosting is used

        base_estimator : object, optional (default=DecisionTreeClassifier)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper `classes_`
            and `n_classes_` attributes.

        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by
            ``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``.

        uboosting_rate: float, optional (default=1.)
            how much do we take into account the uniformity of signal,
             there is a trade-off between uboosting_rate and the speed of uniforming, zero
             value corresponds to plain AdaBoost

        boost_only_signal: bool (default=True),
            if True, only weights of signal are changed depending on local efficiency (as in uBoost)
            if False, both weights of signal and background are changed

        train_variables: list of strings, names of variables used in fit/predict.
            if None, all the variables are used (including uniform_variables)

        smoothing: float, default=(0.), used to smooth computing of local efficiencies
            0.0 corresponds to usual uBoost

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

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
        .. [1] Justin Stevens, Mike Williams 'uBoost: A boosting method for producing uniform
            selection efficiencies from multivariate classifiers'
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.uboosting_rate = uboosting_rate
        self.uniform_variables = uniform_variables
        self.target_efficiency = target_efficiency
        self.n_neighbors = n_neighbors
        self.bagging = bagging
        self.boost_only_signal = boost_only_signal
        self.train_variables = train_variables
        self.smoothing = smoothing
        self.keep_debug_info = keep_debug_info
        self.random_state = random_state

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
        if (sample_weight is not None) and sample_weight.sum() <= 0:
            raise ValueError("Attempting to fit with a non-positive weighted number of samples.")

        assert numpy.all((y == 0) | (y == 1)), "only two-class classification is possible"

        if neighbours_matrix is not None:
            assert numpy.shape(neighbours_matrix) == (len(X), self.n_neighbors), "Wrong shape of neighbours_matrix"
            self.knn_indices = neighbours_matrix
        else:
            assert self.uniform_variables is not None, "uniform_variables must be set"
            # computing knn matrix
            self.knn_indices = computeSignalKnnIndices(self.uniform_variables, X, y > 0.5, self.n_neighbors)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = numpy.zeros(len(X), dtype=numpy.float) + 1. / len(X)
        else:
            # Normalize existing weights
            sample_weight = numpy.copy(sample_weight) / sample_weight.sum()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = numpy.zeros(self.n_estimators, dtype=numpy.float)
        self.estimator_errors_ = numpy.ones(self.n_estimators, dtype=numpy.float)
        # BDT cuts, which correspond to global efficiency == target_efficiency on each iteration
        self.bdt_cuts_ = []

        X_train_variables = self.get_train_vars(X)
        y = numpy.ravel(y)
        X_train_variables, y = check_arrays(X_train_variables, y, sparse_format="dense")

        # Some dictionary to keep all intermediate weights, efficiencies and so on
        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)
        # Setting up random generator
        self.random_generator = check_random_state(self.random_state)
        # Boosting itself
        self._boost_discrete(X_train_variables, y, sample_weight)
        # compute BDT cut
        self.bdt_cut = computeBDTCut(self.target_efficiency, y, self.predict_proba(X))
        assert self.bdt_cut == self.bdt_cuts_[-1], "BDT cut doesn't appear to coincide with staged one"

        return self


    def _boost_discrete(self, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm,
        which is modified in uBoost way"""
        cumulative_score = numpy.zeros(len(X))
        for iboost in xrange(self.n_estimators):
            # creating new estimator
            estimator = clone(self.base_estimator)
            self.estimators_.append(estimator)

            try:
                estimator.set_params(random_state=self.random_state)
            except ValueError:
                pass

            # generating bagging, mask is to prevent overfitting
            masked_sample_weight = sample_weight.copy()
            n_samples = len(X)
            if isinstance(self.bagging, bool) and self.bagging is True:
                indices = self.random_generator.randint(0, n_samples, n_samples)
                sample_counts = bincount(indices, minlength=n_samples)
                masked_sample_weight *= sample_counts
            elif isinstance(self.bagging, float):
                masked_sample_weight *= (self.random_generator.rand(len(X)) > 1 - self.bagging)
            else:
                assert isinstance(self.bagging, bool) and self.bagging is False, "something wrong was passed as bagging"

            estimator.fit(X, y, sample_weight=masked_sample_weight)

            y_predict = estimator.predict(X)

            # Instances incorrectly classified
            incorrect = y_predict != y

            # Error fraction
            estimator_error = numpy.average(incorrect, weights=sample_weight)
            self.estimator_errors_[iboost] = estimator_error

            # Stop if classification is perfect
            if estimator_error <= 0:
                self.estimator_weights_[iboost] = 1
                return

            # Stop if the error is at least as bad as random guessing <-- I've deleted that

            # Boost weight using multi-class AdaBoost SAMME alg
            estimator_weight = self.learning_rate * (numpy.log((1. - estimator_error) / estimator_error))
            self.estimator_weights_[iboost] = estimator_weight

            # Default SAMME -- boosting only positive weights
            # sample_weight *= numpy.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

            sample_weight *= numpy.exp(estimator_weight * incorrect)
            sample_weight += 1e-30
            sample_weight /= numpy.sum(sample_weight)

            cumulative_score += (2 * y_predict - 1) * estimator_weight
            # assert numpy.all(cumulative_score == self.predict_score(X)), "wrong prediction"
            predict_proba = self.score_to_proba(cumulative_score)

            global_cut = computeBDTCut(self.target_efficiency, y, predict_proba)
            self.bdt_cuts_.append(global_cut)
            local_efficiencies = computeLocalEfficiencies(global_cut, self.knn_indices,
                                                          y, predict_proba, self.smoothing)
            e_prime = numpy.sum(sample_weight * numpy.abs(local_efficiencies - self.target_efficiency))
            # TODO why do we have nominator here?
            # beta = math.log((1.0 - e_prime) / e_prime)
            beta = math.log(1. / e_prime)
            if self.boost_only_signal:
                sample_weight *= numpy.exp((self.target_efficiency - local_efficiencies) * y * (beta * self.uboosting_rate))
            else:
                sample_weight *= numpy.exp((self.target_efficiency - local_efficiencies) * (beta * self.uboosting_rate))

            sample_weight_sum = numpy.sum(sample_weight)
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                return
            sample_weight /= sample_weight_sum

            if self.keep_debug_info:
                self.debug_dict['weights'].append(sample_weight.copy())
                self.debug_dict['local_efficiencies'].append(local_efficiencies.copy())


    def get_train_vars(self, X):
        """Gets the DataFrame and returns only columns that should be used in fitting / predictions"""
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]

    def predict_score(self, X):
        score = numpy.zeros(len(X))
        X_train_vars = self.get_train_vars(X)
        for classifier, weight in zip(self.estimators_, self.estimator_weights_):
            score += (2 * classifier.predict(X_train_vars) - 1) * weight
        return score

    def staged_predict_score(self, X):
        score = numpy.zeros(len(X))
        X_train_vars = self.get_train_vars(X)
        for classifier, weight in zip(self.estimators_, self.estimator_weights_):
            score += (2 * classifier.predict(X_train_vars) - 1) * weight
            yield score

    @staticmethod
    def score_to_proba(score):
        """Compute class probability estimates from decision scores. """
        proba = numpy.ones((score.shape[0], 2), dtype=numpy.float64)
        proba[:, 1] = 1.0 / (1.0 + numpy.exp(-score.ravel()))
        proba[:, 0] -= proba[:, 1]
        return proba

    def predict(self, X):
        """Predict classes for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features], the input samples.

        Returns
        -------
        y : array of shape = [n_samples], the predicted classes.
        """
        # return AdaBoostClassifier.predict(self, self.get_train_vars(X))
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        return self.score_to_proba(self.predict_score(X))

    def staged_predict_proba(self, X):
        for score in self.staged_predict_score(X):
            yield self.score_to_proba(score)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        return sum(tree.feature_importances_ * weight
                   for tree, weight in zip(self.estimators_, self.estimator_weights_)) / self.n_estimators





def trainClassifier(classifier, X_train_vars, y, sample_weight, neighbours_matrix):
    return classifier.fit(X_train_vars, y, sample_weight=sample_weight, neighbours_matrix=neighbours_matrix)



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
                 ipc_profile=None):
        """uBoost classifier, am algorithm of boosting targeted to obtain
        flat efficiency in signal along some variables. See [1] for details.

        Parameters
        ----------
        uniform_variables: list of strings, names of variables, along which flatness is desired

        n_neighbours: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        efficiency_steps: integer, optional (default=20),
            How many uBoostBDTs should be trained (each with its own target_efficiency)

        base_estimator : object, optional (default=DecisionTreeClassifier)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper `classes_`
            and `n_classes_` attributes.

        bagging: float or bool (default=True), bagging usually speeds up the convergence
            and prevents overfitting (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
            if True, usual bootstrap aggregating is used (sampling with replacement at each iteration, size=len(X))
            if float, used sampling with replacement, the size of generated set is bagging * len(X)
            if False, usual boosting is used

        train_variables: list of strings, names of variables used in fit/predict.
            if None, all the variables are used (including uniform_variables)

        boost_only_signal: bool (default=True),
            if True, only weights of signal are changed depending on local efficiency (as in uBoost)
            if False, both weights of signal and background are changed

        smoothing: float, default=(0.), used to smooth computing of local efficiencies
            0.0 corresponds to usual uBoost

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `numpy.random`.

        parallel_profile: profile (name of cluster) in IPython to parallelize computations

        Reference
        ----------
        .. [1] Justin Stevens, Mike Williams 'uBoost: A boosting method for producing uniform
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

    def get_train_variables(self, X):
        if self.train_variables is not None:
            return X[self.train_variables]
        else:
            return X


    def fit(self, X, y, sample_weight=None):
        if self.uniform_variables is None:
            raise ValueError("Please set uniformVariables")
        if len(self.uniform_variables) == 0:
            raise ValueError("The set of uniform variables cannot be empty")
        if len(X) != len(y):
            raise ValueError("Different size of X and y")

        X_train_vars = self.get_train_variables(X)

        if self.smoothing is None:
            self.smoothing = 0.2 / self.efficiency_steps

        neighbours_matrix = computeSignalKnnIndices(self.uniform_variables, X, y > 0.5, n_neighbors=self.knn)
        # TODO select some other targets
        self.target_efficiencies = [(i + 1.0) / (self.efficiency_steps + 1.0) for i in range(self.efficiency_steps)]
        self.classifiers = []

        for efficiency in self.target_efficiencies:
            classifier = uBoostBDT(uniform_variables=self.uniform_variables, train_variables=None,
                                   target_efficiency=efficiency, n_neighbors=self.knn,
                                   n_estimators=self.n_estimators, base_estimator=self.base_estimator,
                                   random_state=self.random_state, bagging=self.bagging,
                                   boost_only_signal=self.boost_only_signal, smoothing=self.smoothing)
            self.classifiers.append(classifier)

        if self.ipc_profile is not None:
            from IPython.parallel import Client
            client = Client(profile=self.ipc_profile)
            self.classifiers  = client.load_balanced_view().map_sync(trainClassifier, self.classifiers,
                                    [X_train_vars] * self.efficiency_steps, [y] * self.efficiency_steps,
                                    [sample_weight] * self.efficiency_steps, [neighbours_matrix] * self.efficiency_steps)
        else:
            self.classifiers = map(trainClassifier, self.classifiers,
                                   [X_train_vars] * self.efficiency_steps, [y] * self.efficiency_steps,
                                   [sample_weight] * self.efficiency_steps, [neighbours_matrix] * self.efficiency_steps)


        return self

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X_train_vars = self.get_train_variables(X)
        signal_proba = numpy.zeros(len(X))
        result = numpy.zeros([len(X), 2])
        for efficiency, classifier in zip(self.target_efficiencies, self.classifiers):
            signal_proba += sigmoidFunction(classifier.predict_proba(X_train_vars)[:, 1] - classifier.bdt_cut,
                                            self.smoothing)

        signal_proba /= self.efficiency_steps
        result[:, 1] = signal_proba
        result[:, 0] = 1.0 - signal_proba
        return result

    def staged_predict_proba(self, X):
        X = self.get_train_variables(X)
        signal_proba_stages = numpy.zeros([self.n_estimators, len(X)])
        for classifier in self.classifiers:
            staged_predicitions = list(classifier.staged_predict_proba(X))
            for i, stage_prediction in enumerate(staged_predicitions):
                signal_proba_stages[i, :] += sigmoidFunction(stage_prediction[:, 1] - classifier.bdt_cuts_[i],
                                                             self.smoothing)
        signal_proba_stages /= self.efficiency_steps
        result = []
        for signal_proba in signal_proba_stages:
            staged_prediction = numpy.zeros([len(X), 2])
            staged_prediction[:, 1] = signal_proba
            staged_prediction[:, 0] = 1.0 - signal_proba
            result.append(staged_prediction)
        return result


def test_uboost_classifier():
    # Generating some samples correlated with first variable
    testX, testY = generateSample(2000, 10, 0.6)
    trainX, trainY = generateSample(2000, 10, 0.6)
    # We will try to get uniform distribution along this variable
    uniform_variables = ['column0']

    base_classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=12)

    for target_efficiency in [0.1, 0.3, 0.5, 0.7, 0.9]:
        bdt_classifier = uBoostBDT(uniform_variables=uniform_variables, target_efficiency=target_efficiency,
                                   n_neighbors=20, n_estimators=20, base_estimator=base_classifier)
        bdt_classifier.fit(trainX, trainY)
        filtered = numpy.sum(bdt_classifier.predict_proba(trainX[trainY > 0.5])[:, 1] > bdt_classifier.bdt_cut)
        assert abs(filtered - numpy.sum(trainY) * target_efficiency) < 5, "global cut is set wrongly"

        staged_filtered_upper = [numpy.sum(pred[:, 1] > cut - 1e-7) for pred, cut in \
                                 zip(bdt_classifier.staged_predict_proba(trainX[trainY > 0.5]),
                                     bdt_classifier.bdt_cuts_)]
        staged_filtered_lower = [numpy.sum(pred[:, 1] > cut + 1e-7) for pred, cut in \
                                 zip(bdt_classifier.staged_predict_proba(trainX[trainY > 0.5]),
                                     bdt_classifier.bdt_cuts_)]

        assert bdt_classifier.bdt_cut == bdt_classifier.bdt_cuts_[-1], 'something wrong with computed cuts'
        for filter_lower, filter_upper in zip(staged_filtered_lower, staged_filtered_upper)[10:]:
            assert filter_lower - 1 <= sum(trainY) * target_efficiency <= filter_upper + 1, "stage cut is set wrongly"

    uboost_classifier = uBoostClassifier(uniform_variables=uniform_variables, n_neighbors=20, efficiency_steps=3,
                                         n_estimators=20)

    bdt_classifier = uBoostBDT(uniform_variables=uniform_variables, n_neighbors=20, n_estimators=20,
                               base_estimator=base_classifier)

    for classifier in [bdt_classifier, uboost_classifier]:
        classifier.fit(trainX, trainY)
        proba1 = classifier.predict_proba(testX)
        proba2 = list(classifier.staged_predict_proba(testX))[-1]
        assert numpy.all(abs(proba1 - proba2) < 0.001), "something wrong with predictions"

    assert len(bdt_classifier.feature_importances_) == trainX.shape[1]

    print 'uboost is ok'


test_uboost_classifier()

