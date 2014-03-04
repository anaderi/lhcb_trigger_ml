from collections import defaultdict
import numpy as np
import numpy
from numpy.core.umath_tests import inner1d
from numpy.lib._compiled_base import bincount
import pandas
from sklearn.base import BaseEstimator
from sklearn.ensemble.weight_boosting import BaseWeightBoosting, ClassifierMixin, _samme_proba, AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.utils.random import check_random_state


from commonutils import computeLocalEfficiencies, computeBDTCut, sigmoidFunction, plotScoreVariableCorrelation, \
    generateSample






class uBoostBDT(AdaBoostClassifier):
    """
    uBoostBDT is based on AdaBoostClassifier, below is AdaBoost description

    An AdaBoost classifier.

    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm known as AdaBoost-SAMME [2].

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.

    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

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

    See also
    --------
    AdaBoostRegressor, GradientBoostingClassifier, DecisionTreeClassifier

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """

    def __init__(self,
                 uniform_variables,
                 target_efficiency=0.5,
                 neighbours=100,
                 bagging=True,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50,
                 learning_rate=1.,
                 # overriden, default was "SAMME.R"
                 algorithm='SAMME',
                 random_state=None,
                 boost_only_signal=True):
        """uBoostBDT is a modification of AdaBoost algorithm, which ensures that efficiency of the signal
         is uniform at target_efficiency along the uniform_variables"""
        AdaBoostClassifier.__init__(self,
                                    base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    algorithm=algorithm,
                                    random_state=random_state)

        self.uniform_variables = uniform_variables
        self.target_efficiency = target_efficiency
        self.neighbours = neighbours
        self.bagging = bagging
        self.boost_only_signal = boost_only_signal

    def fit(self, X, y, sample_weight=None):
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

        Returns
        -------
        self : object
            Returns self.
        """
        if isinstance(self.neighbours, int):
            if self.uniform_variables is None:
                raise ValueError("uniformVariables must be set")
            # computing knn matrix
            # see http://scikit-learn.org/stable/modules/neighbors.html for algorithms of knn-ing
            signal_indices = numpy.where(y)[0]
            uniforming_features_of_signal = X.ix[signal_indices, self.uniform_variables]

            neighbours = NearestNeighbors(n_neighbors=self.neighbours, algorithm='kd_tree') \
                .fit(uniforming_features_of_signal)
            _, knnSignalIndices = neighbours.kneighbors(X[self.uniform_variables])

            # knn_indices = numpy.zeros((len(X), self.neighbours), dtype=numpy.int32)
            knn_indices = numpy.take(signal_indices, knnSignalIndices)

            # for index, signalNeigh in zip(signal_indices, knnSignalIndices):
            #     knn_indices[index,:] = signal_indices[signalNeigh]
            self.knnIndices = knn_indices
        else:
            # assuming that knn_indices were passed in neighbours
            assert isinstance(self.neighbours, numpy.ndarray)
            self.knnIndices = self.neighbours

        self.debugdict = defaultdict(list)

        self.debug_dict = defaultdict(list)
        self.staged_bdt_cut = []
        self.global_random_generator = check_random_state(self.random_state)
        self.cumulative_proba = numpy.zeros((len(X),2))

        result = AdaBoostClassifier.fit(self, X, y, sample_weight)
        # compute BDTcut
        self.BDTCut = computeBDTCut(self.target_efficiency, y, self.predict_proba(X))

        return result


    def _boost_real(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME.R real algorithm."""
        raise NotImplemented("this works strange")

    #         # not used at this moment, _boost_discrete is preffered
    #         estimator = self._make_estimator()
    #
    #         try:
    #             estimator.set_params(random_state=self.random_state)
    #         except ValueError:
    #             pass
    #
    #         estimator.fit(X, y, sample_weight=sample_weight)
    #
    #         y_predict_proba = estimator.predict_proba(X)
    #
    #         if iboost == 0:
    #             self.classes_ = getattr(estimator, 'classes_', None)
    #             self.n_classes_ = len(self.classes_)
    #
    #         y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
    #                                        axis=0)
    #
    #         # Instances incorrectly classified
    #         incorrect = y_predict != y
    #
    #         # Error fraction
    #         estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
    #
    #         # Stop if classification is perfect
    #         if estimator_error <= 0:
    #             return sample_weight, 1., 0.
    #
    #         # Construct y coding as described in Zhu et al [2]:
    #         #
    #         #    y_k = 1 if c == k else -1 / (K - 1)
    #         #
    #         # where K == n_classes_ and c, k in [0, K) are indices along the second
    #         # axis of the y coding with c being the index corresponding to the true
    #         # class label.
    #         n_classes = self.n_classes_
    #         classes = self.classes_
    #         y_codes = np.array([-1. / (n_classes - 1), 1.])
    #         y_coding = y_codes.take(classes == y[:, np.newaxis])
    #
    #         # Displace zero probabilities so the log is defined.
    #         # Also fix negative elements which may occur with
    #         # negative sample weights.
    #         y_predict_proba[y_predict_proba <= 0] = 1e-5
    #         y_total_prediction_proba = AdaBoostClassifier.predict_proba(self, X)
    #
    #         # Boost weight using multi-class AdaBoost SAMME.R alg
    #         estimator_weight = (-1. * self.learning_rate
    #                                 * (((n_classes - 1.) / n_classes) *
    #                                    inner1d(y_coding, np.log(y_predict_proba))))
    #         self.debug_dict['estimator_weight'].append(numpy.sum(estimator_weight))
    #         self.debug_dict['estimator_error'].append(estimator_error)
    #
    #         # Only boost the weights if it will fit again
    #         if not iboost == self.n_estimators - 1:
    #             # Only boost positive weights
    #             sample_weight *= np.exp(estimator_weight *
    #                                     ((sample_weight > 0) |
    #                                      (estimator_weight < 0)))
    #             # here some uBoost-like code change
    #             globalCut = ComputeBDTCut(self.targetEfficiency, y, y_total_prediction_proba)
    #             self.staged_bdt_cut.append(globalCut)
    #             local_efficiencies = ComputeLocalEfficiencies(globalCut, self.knnIndices, y, y_total_prediction_proba)
    # #             globalCut = ComputeBDTCut(self.targetEfficiency, y, y_predict_proba)
    # #             localEfficiencies = ComputeLocalEfficiencies(globalCut, self.knnIndices, y, y_predict_proba)
    #             eprime = numpy.sum(sample_weight * abs(local_efficiencies - self.targetEfficiency))
    #             beta = math.log((1.0-eprime) / eprime)
    #
    #             self.debug_dict['beta'].append(beta)
    #             self.debug_dict['eprime'].append(eprime)
    #             self.debug_dict['sample_weight'].append(sample_weight)
    #             self.debug_dict['weight_sum'].append(numpy.sum(sample_weight))
    #             self.debug_dict['efficiencies'].append(local_efficiencies)
    #
    #             sample_weight /= sum(sample_weight)
    #
    #             # weight is changed only for signal events
    #             sample_weight *= np.exp((self.targetEfficiency - local_efficiencies) * (y > 0.5) * beta)
    #             # pay attention we are replacing everything was before
    #             sample_weight = np.exp((self.targetEfficiency - local_efficiencies) * (y > 0.5) * beta)
    #             sample_weight /= sum(sample_weight)
    #
    #         return sample_weight, 1., estimator_error


    def _boost_discrete(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm."""

        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        # mask is to prevent overfitting
        masked_sample_weight = sample_weight.copy()
        n_samples = len(X)
        if self.bagging == True:
            indices = self.global_random_generator.randint(0, n_samples, n_samples)
            sample_counts = bincount(indices, minlength=n_samples)
            masked_sample_weight *= sample_counts
        elif isinstance(self.bagging, float):
            masked_sample_weight *= (self.global_random_generator.rand(len(X)) > 1 - self.bagging)
        else:
            assert self.bagging == False, "something wrong was passed as bagging"

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

        # I switched this off
        # Stop if the error is at least as bad as random guessing
        # if estimator_error >= 1. - (1. / n_classes):
        #     self.estimators_.pop(-1)
        #     return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # My addition - we are stating new weight beforehead
        self.estimator_weights_[iboost] = estimator_weight

        # Only boost positive weights (TODO why?)
        sample_weight *= np.exp(estimator_weight * incorrect *
                                ((sample_weight > 0) |
                                 (estimator_weight < 0)))
        # uboost changes
        sample_weight /= numpy.sum(sample_weight)

        # cumulative proba
        self.cumulative_proba += self.estimators_[-1].predict_proba(X) * estimator_weight
            # = sum(estimator.predict_proba(X) * w
            # for estimator, w in zip(self.estimators_,
            #                         self.estimator_weights_))


        bdt_prediction_proba = self.cumulative_proba / self.estimator_weights_.sum()
        bdt_prediction_proba = np.exp((1. / (n_classes - 1)) * bdt_prediction_proba)
        normalizer = bdt_prediction_proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        bdt_prediction_proba /= normalizer


        globalCut = computeBDTCut(self.target_efficiency, y, bdt_prediction_proba)
        # print 'global cut = ', globalCut
        # print "y_total_prediction_proba\n", y_total_prediction_proba
        self.staged_bdt_cut.append(globalCut)
        local_efficiencies = computeLocalEfficiencies(globalCut, self.knnIndices, y, bdt_prediction_proba)
        eprime = numpy.sum(sample_weight * abs(local_efficiencies - self.target_efficiency))
        # TODO why do we have nominator here?
        beta = math.log((1.0 - eprime) / eprime)
        if self.boost_only_signal:
            sample_weight *= np.exp((self.target_efficiency - local_efficiencies) * y * beta)
        else:
            sample_weight *= np.exp((self.target_efficiency - local_efficiencies) * beta)

        # not needed, will be done outside this function automatically
        # sample_weight /= sum(sample_weight)

        return sample_weight, estimator_weight, estimator_error



class uBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, uniform_variables=None,
                 knn=50,
                 efficiency_steps=100,
                 random_state=None,
                 n_estimators=40,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 bagging=True,
                 train_variables=None,
                 boost_only_signal=True):
        self.uniform_variables = uniform_variables
        self.knn = knn
        self.efficiency_steps = efficiency_steps
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.bagging = bagging
        self.train_variables = train_variables
        self.boost_only_signal = boost_only_signal

    def fit(self, X, y):
        if self.uniform_variables is None:
            raise ValueError("Please set uniformVariables")
        if len(self.uniform_variables) != 1:
            raise ValueError("Only one uniform variable is supported by now")
        if self.train_variables is not None:
            X_train_vars = X[self.train_variables]
        else:
            X_train_vars = X

        signal_indices = numpy.where(y)[0]
        uniforming_features_of_signal = X.ix[signal_indices, self.uniform_variables]
        neighbours = NearestNeighbors(n_neighbors=self.knn, algorithm='kd_tree').fit(uniforming_features_of_signal)
        _, knn_signal_indices = neighbours.kneighbors(X[self.uniform_variables])
        knn_indices = numpy.take(signal_indices, knn_signal_indices)

        self.target_efficiencies = [(i + 1.0) / (self.efficiency_steps + 1.0) for i in range(self.efficiency_steps)]
        self.classifiers = []
        for efficiency in self.target_efficiencies:
            classifier = uBoostBDT(self.uniform_variables, efficiency, neighbours=knn_indices,
                                   n_estimators=self.n_estimators, base_estimator=self.base_estimator,
                                   random_state=self.random_state, bagging=self.bagging,
                                   boost_only_signal=self.boost_only_signal)
            classifier.fit(X_train_vars, y)
            self.classifiers.append(classifier)
        return self

    def print_bdt_cuts_efficiency(self, X, y):
        X_uniform_vars = X[self.uniform_variables]
        if self.train_variables is not None:
            X = X[self.train_variables]
        for efficiency, classifier in zip(self.target_efficiencies, self.classifiers):
            signal_probas = classifier.predict_proba(X)
            plotScoreVariableCorrelation(y, signal_probas, X_uniform_vars, thresholds=classifier.BDTCut)

    def predict(self, X):
        if self.train_variables is not None:
            X = X[self.train_variables]
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X, smoothing_width=None):
        if self.train_variables is not None:
            X = X[self.train_variables]
        if smoothing_width is None:
            smoothing_width = 0.2 / self.efficiency_steps
        signal_proba = np.zeros(len(X))
        result = numpy.zeros((len(X), 2))
        for efficiency, classifier in zip(self.target_efficiencies, self.classifiers):
            signal_proba += sigmoidFunction(classifier.predict_proba(X)[:, 1] - classifier.BDTCut, smoothing_width)

        signal_proba /= self.efficiency_steps
        result[:, 1] = signal_proba
        result[:, 0] = 1.0 - signal_proba
        return result

    def staged_predict_proba(self, X, smoothing_width=None):
        if self.train_variables is not None:
            X = X[self.train_variables]
        if smoothing_width is None:
            smoothing_width = 0.2 / self.efficiency_steps
        signal_proba_stages = numpy.zeros(( self.n_estimators, len(X) ))
        for classifier in self.classifiers:
            staged_predicitions = list(classifier.staged_predict_proba(X))
            i = 0
            for stage_prediction in staged_predicitions:
                signal_proba_stages[i, :] += sigmoidFunction(stage_prediction[:, 1] - classifier.staged_bdt_cut[i],
                                                             smoothing_width)
                i += 1
            for i in range(len(staged_predicitions), self.n_estimators):
                signal_proba_stages[i, :] = signal_proba_stages[i - 1, :]

        signal_proba_stages /= self.efficiency_steps
        result = []
        for signal_proba in signal_proba_stages:
            staged_prediction = numpy.zeros((len(X), 2))
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
                                   neighbours=20,
                                   n_estimators=20, base_estimator=base_classifier)
        bdt_classifier.fit(trainX, trainY)
        filtered = numpy.sum(bdt_classifier.predict_proba(trainX[trainY > 0.5])[:, 1] > bdt_classifier.BDTCut)
        assert abs(filtered - sum(trainY) * target_efficiency) < 5, "global cut is set wrongly"

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
        assert numpy.all(abs(proba1 - proba2)<0.001), "something wrong with predictions"

    print 'uboost is ok'


test_uboost_classifier()

