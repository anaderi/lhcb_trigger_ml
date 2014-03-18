from collections import defaultdict
import math

import numpy
import numpy as np
from numpy.lib._compiled_base import bincount
from sklearn.base import BaseEstimator
from sklearn.ensemble.weight_boosting import ClassifierMixin, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import check_random_state

__author__ = 'Alex Rogozhnikov'

from commonutils import computeLocalEfficiencies, computeBDTCut, sigmoidFunction, \
    generateSample, computeSignalKnnIndices


class uBoostBDT(AdaBoostClassifier):
    def __init__(self,
                 uniform_variables,
                 target_efficiency=0.5,
                 n_neighbors=50,
                 bagging=True,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50,
                 learning_rate=1.,
                 boost_only_signal=True,
                 train_variables=None,
                 smoothing=0.0,
                 random_state=None):
        """
        uBoostBDT is AdaBoostClassifier, which is modified to have flat efficiency
        of signal (class=1) along some variables. Efficiency is only guaranteed at the cut,
        corresponding to global efficiency == target_efficiency

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

        `classes_` : array of shape = [n_classes]
            The classes labels.

        `n_classes_` : int
            The number of classes.

        `estimator_weights_` : array of floats
            Weights for each estimator in the boosted ensemble.

        `estimator_errors_` : array of floats
            Classification error for each estimator in the boosted
            ensemble.

        `feature_importances_` : array of shape = [n_features]
            The feature importances if supported by the ``base_estimator``.

        Reference
        ----------
        .. [1] Justin Stevens, Mike Williams 'uBoost: A boosting method for producing uniform
            selection efficiencies from multivariate classifiers'
        """
        AdaBoostClassifier.__init__(self,
                                    base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    algorithm='SAMME',
                                    random_state=random_state)

        self.uniform_variables = uniform_variables
        self.target_efficiency = target_efficiency
        self.n_neighbors = n_neighbors
        self.bagging = bagging
        self.boost_only_signal = boost_only_signal
        self.train_variables = train_variables
        self.smoothing = smoothing

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
        if neighbours_matrix is not None:
            assert numpy.shape(neighbours_matrix) == (len(X), self.n_neighbors), "Wrong shape of neighbours_matrix"
            self.knn_indices = neighbours_matrix
        else:
            assert self.uniform_variables is not None, "uniform_variables must be set"
            # computing knn matrix
            self.knn_indices = computeSignalKnnIndices(self.uniform_variables, X, y > 0.5, self.n_neighbors)

        assert self.smoothing >= 0., "Smoothing can not be negative"
        X_train_variables = self.get_train_vars(X)
        # Some dictionary to keep all intermediate weights, efficiencies and so on
        self.debug_dict = defaultdict(list)
        # BDT cuts, which correspond to global efficiency == target_efficiency on each iteration
        self.staged_bdt_cut = []
        self.global_random_generator = check_random_state(self.random_state)
        self.cumulative_proba = numpy.zeros((len(X_train_variables), 2))
        # run the adaBoost fitting (which will use _boost_discrete)
        AdaBoostClassifier.fit(self, X_train_variables, y, sample_weight)
        # compute BDTcut
        self.BDTCut = computeBDTCut(self.target_efficiency, y, self.predict_proba(X))
        assert self.BDTCut == self.staged_bdt_cut[-1], "BDT cut doesn't appear to coincide with staged one"
        return self

    def _boost_real(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME.R real algorithm."""
        raise NotImplemented("Classification based on SAMME.R is not implemented for uBoost")

    def _boost_discrete(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm,
        which is modified in uBoost way"""

        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        # generating bagging, mask is to prevent overfitting
        masked_sample_weight = sample_weight.copy()
        n_samples = len(X)
        if isinstance(self.bagging, bool) and self.bagging is True:
            indices = self.global_random_generator.randint(0, n_samples, n_samples)
            sample_counts = bincount(indices, minlength=n_samples)
            masked_sample_weight *= sample_counts
        elif isinstance(self.bagging, float):
            masked_sample_weight *= (self.global_random_generator.rand(len(X)) > 1 - self.bagging)
        else:
            assert isinstance(self.bagging, bool) and self.bagging is False, "something wrong was passed as bagging"

        estimator.fit(X, y, sample_weight=masked_sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # I switched this off, because uBoost should not terminate in this case
        # Stop if the error is at least as bad as random guessing
        # if estimator_error >= 1. - (1. / n_classes):
        #     self.estimators_.pop(-1)
        #     return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * \
                           (np.log((1. - estimator_error)/estimator_error) + np.log(n_classes - 1.))

        # Default SAMME -- boosting only positive weights
        # sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        sample_weight *= np.exp(estimator_weight * incorrect)
        # uboost changes
        sample_weight /= numpy.sum(sample_weight)

        # cumulative proba, needed to
        self.cumulative_proba += self.estimators_[-1].predict_proba(X) * estimator_weight
            # = sum(estimator.predict_proba(X) * w for estimator, w in zip(self.estimators_,
            #                         self.estimator_weights_))

        self.estimator_weights_[iboost] = estimator_weight
        bdt_prediction_proba = self.cumulative_proba / self.estimator_weights_.sum()
        bdt_prediction_proba = np.exp((1. / (n_classes - 1)) * bdt_prediction_proba)
        normalizer = bdt_prediction_proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        bdt_prediction_proba /= normalizer
        # assert numpy.all(bdt_prediction_proba == self.predict_proba(X)), "the predictions are different"

        global_cut = computeBDTCut(self.target_efficiency, y, bdt_prediction_proba)
        self.staged_bdt_cut.append(global_cut)
        local_efficiencies = computeLocalEfficiencies(global_cut, self.knn_indices,
                                                      y, bdt_prediction_proba, self.smoothing)
        e_prime = numpy.sum(sample_weight * numpy.abs(local_efficiencies - self.target_efficiency))
        # TODO why do we have nominator here?
        beta = math.log((1.0 - e_prime) / e_prime)
        if self.boost_only_signal:
            sample_weight *= numpy.exp((self.target_efficiency - local_efficiencies) * y * (beta * self.learning_rate))
        else:
            sample_weight *= numpy.exp((self.target_efficiency - local_efficiencies) * (beta * self.learning_rate))

        # not needed, will be done outside this function automatically
        # sample_weight /= sum(sample_weight)
        return sample_weight, estimator_weight, estimator_error

    def get_train_vars(self, X):
        """Gets the DataFrame and returns only columns that should be used in fitting / predictions"""
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]

    def predict(self, X):
        """Predict classes for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features], the input samples.

        Returns
        -------
        y : array of shape = [n_samples], the predicted classes.
        """
        return AdaBoostClassifier.predict(self, self.get_train_vars(X))

    def predict_proba(self, X):
        return AdaBoostClassifier.predict_proba(self, self.get_train_vars(X))

    def staged_predict_proba(self, X):
        return AdaBoostClassifier.staged_predict_proba(self, self.get_train_vars(X))






class uBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, uniform_variables=None,
                 knn=50,
                 efficiency_steps=100,
                 random_state=None,
                 n_estimators=40,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 bagging=True,
                 train_variables=None,
                 boost_only_signal=True,
                 smoothing=None):
        self.uniform_variables = uniform_variables
        self.knn = knn
        self.efficiency_steps = efficiency_steps
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.bagging = bagging
        self.train_variables = train_variables
        self.boost_only_signal = boost_only_signal
        self.smoothing = smoothing

    def get_train_variables(self, X):
        if self.train_variables is not None:
            return X[self.train_variables]
        else:
            return X

    def fit(self, X, y):
        if self.uniform_variables is None:
            raise ValueError("Please set uniformVariables")
        if len(self.uniform_variables) == 0:
            raise ValueError("The set of uniform variables cannot be empty")
        assert len(X) == len(y), "different size"

        X_train_vars = self.get_train_variables(X)

        if self.smoothing is None:
            self.smoothing = 0.2 / self.efficiency_steps

        knn_indices = computeSignalKnnIndices(self.uniform_variables, X, y > 0.5, knn=self.knn)


        self.target_efficiencies = [(i + 1.0) / (self.efficiency_steps + 1.0) for i in range(self.efficiency_steps)]
        self.classifiers = []
        for efficiency in self.target_efficiencies:
            classifier = uBoostBDT(self.uniform_variables, efficiency, neighbours=knn_indices,
                                   n_estimators=self.n_estimators, base_estimator=self.base_estimator,
                                   random_state=self.random_state, bagging=self.bagging,
                                   boost_only_signal=self.boost_only_signal, smoothing=self.smoothing)
            classifier.fit(X_train_vars, y)
            self.classifiers.append(classifier)
        return self

    def print_bdt_cuts_efficiency(self, X, y):
        X_uniform_vars = X[self.uniform_variables]
        X_train_vars = self.get_train_variables(X)
        for efficiency, classifier in zip(self.target_efficiencies, self.classifiers):
            signal_probas = classifier.predict_proba(X_train_vars)
            plotScoreVariableCorrelation(y, signal_probas, X_uniform_vars, thresholds=classifier.BDTCut)

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X_train_vars = self.get_train_variables(X)

        signal_proba = np.zeros(len(X))
        result = numpy.zeros((len(X), 2))
        for efficiency, classifier in zip(self.target_efficiencies, self.classifiers):
            signal_proba += sigmoidFunction(classifier.predict_proba(X_train_vars)[:, 1] - classifier.BDTCut,
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
                signal_proba_stages[i, :] += sigmoidFunction(stage_prediction[:, 1] - classifier.staged_bdt_cut[i],
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
                                   neighbours=20, n_estimators=20, base_estimator=base_classifier)
        bdt_classifier.fit(trainX, trainY)
        filtered = numpy.sum(bdt_classifier.predict_proba(trainX[trainY > 0.5])[:, 1] > bdt_classifier.BDTCut)
        assert abs(filtered - numpy.sum(trainY) * target_efficiency) < 5, "global cut is set wrongly"

        staged_filtered = [numpy.sum(pred[:, 1] > cut) for pred, cut in \
                    zip(bdt_classifier.staged_predict_proba(trainX[trainY > 0.5]), bdt_classifier.staged_bdt_cut)]
        assert bdt_classifier.BDTCut == bdt_classifier.staged_bdt_cut[-1], 'something wrong with computed cuts'
        for stage_filter in staged_filtered[10:]:
            assert abs(stage_filter - sum(trainY) * target_efficiency) < 5, "stage cut is set wrongly"

    uboost_classifier = uBoostClassifier(uniform_variables=uniform_variables, knn=20, efficiency_steps=3,
                                         n_estimators=20)
    bdt_classifier = uBoostBDT(uniform_variables=uniform_variables, neighbours=20, n_estimators=20,
                               base_estimator=base_classifier)

    for classifier in [bdt_classifier, uboost_classifier]:
        classifier.fit(trainX, trainY)
        proba1 = classifier.predict_proba(testX)
        proba2 = list(classifier.staged_predict_proba(testX))[-1]
        assert numpy.all(abs(proba1 - proba2) < 0.001), "something wrong with predictions"

    print 'uboost is ok'


test_uboost_classifier()
