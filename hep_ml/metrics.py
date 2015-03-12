# About

# this module contains different metrics of uniformity
# and the metrics of quality as well (which support weights, actually)

from __future__ import division, print_function

import numpy
import pandas
from sklearn.base import BaseEstimator
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.utils.validation import column_or_1d, check_arrays
from sklearn.metrics import roc_curve

from .commonutils import check_sample_weight, computeSignalKnnIndices
from . import metrics_utils as ut
from hep_ml.commonutils import take_features, check_xyw, weighted_percentile


__author__ = 'Alex Rogozhnikov'

__all__ = ['sde', 'cvm_flatness', 'theil_flatness']

"""
README on quality metrics

Some notation used here
IsSignal - is really signal
AsSignal - classified as signal
IsBackgroundAsSignal - background, but classified as signal
... and so on. Cute, right?

There are many ways to denote this things
tpr = s = isSasS / isS
fpr = b = isBasS / isB

signal efficiency = tpr = s
background efficiency = isBasB / isB = 1 - fpr
background rejection = background efficiency (physicists don't agree with the last line)

"""

# region Quality metrics


def roc_curve_splitted(data1, data2, sample_weight1=None, sample_weight2=None):
    """Does exactly the same as sklearn.metrics.roc_curve,
    but for signal/background predictions kept in different arrays.

    Returns: tpr, fpr, thresholds, these are parallel arrays with equal lengths.
    """
    sample_weight1 = check_sample_weight(data1, sample_weight=sample_weight1)
    sample_weight2 = check_sample_weight(data1, sample_weight=sample_weight2)
    data = numpy.concatenate([data1, data2])
    sample_weight = numpy.concatenate([sample_weight1, sample_weight2])
    labels = numpy.concatenate([numpy.zeros(len(data1)), numpy.ones(len(data2))])
    return roc_curve(labels, data, sample_weight=sample_weight)


def compute_sb(y_true, y_pred, sample_weight):
    """Here the passed arguments should be already checked, y_pred is array of 0 and 1"""
    total_s = numpy.sum(sample_weight[y_true > 0.5])
    total_b = numpy.sum(sample_weight[y_true < 0.5])
    s = sample_weight[y_true * y_pred > 0.5].sum()
    b = sample_weight[(1 - y_true) * y_pred > 0.5].sum()
    return s / total_s, b / total_b


def efficiency_score(y_true, y_pred, sample_weight=None):
    """Efficiency = right classified signal / everything that is really signal
    Efficiency == recall, returns -0.1 when ill-defined"""
    sample_weight = check_sample_weight(y_true, sample_weight=sample_weight)
    assert len(y_true) == len(y_pred), "Different size of arrays"
    isSignal = numpy.sum(y_true * sample_weight) - 1e-6
    isSignalAsSignal = numpy.sum(y_true * y_pred * sample_weight) + 1e-7
    return isSignalAsSignal / isSignal
    # the same, but with notifications
    # return recall_score(answer, prediction)


def background_efficiency_score(y_true, y_pred, sample_weight=None):
    """BackgroundEfficiency == isBasB / isB == 1 - fpr"""
    return efficiency_score(1 - y_true, 1 - y_pred, sample_weight=sample_weight)


def as_signal_score(y_true, y_pred, sample_weight=None):
    """Part of is signal = classified as signal / total amount of events"""
    sample_weight = check_sample_weight(y_true, sample_weight)
    assert len(y_true) == len(y_pred), "Different size of arrays"
    return numpy.sum(y_pred * sample_weight) / numpy.sum(sample_weight)


def sensitivity(y_true, y_score, sample_weight=None):
    """ Returns s / sqrt{s+b}
    :param y_true: array-like of shape [n_samples] with labels of samples (0 or 1)
    :param y_score: array-like of shape [n_samples] with predicted labels (0 or 1)"""
    y_true, y_score, sample_weight = \
        ut.check_metrics_arguments(y_true, y_score, sample_weight=sample_weight, two_class=True, binary_pred=True)
    s, b = compute_sb(y_true, y_score, sample_weight=sample_weight)
    return s / numpy.sqrt(s + b + 1e-6)


def optimal_sensitivity(y_true, y_score, sample_weight=None):
    """s,b are normalized to be in [0,1] """
    from sklearn.metrics import roc_curve

    b, s, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    return numpy.max(s / numpy.sqrt(s + b + 1e-6))


# endregion


"""
README on flatness

this metrics are unfortunately more complicated than usual ones
and require more information: not only predictions and classes,
but also mass (or other variables along which we want to hav uniformity)

Here we compute the different metrics of uniformity of predictions:

SDE  - the standard deviation of efficiency
Theil- Theil index of Efficiency (Theil index is used in economics)
KS   - based on Kolmogorov-Smirnov distance between distributions
CVM  - based on Cramer-von Mises similarity between distributions

Mask is needed to show the events of needed class,
for instance, if we want to compute the uniformity on signal predictions,
mask should be True on signal events and False on the others.

y_score in usually predicted probabilities of event being a needed class.

So, if I want to compute efficiency on signal, I put:
  mask = y == 1
  y_pred = clf.predict_proba[:, 1]

If want to do it for bck:
  mask = y == 0
  y_pred = clf.predict_proba[:, 0]

"""


# region Uniform metrics (current version)

class AbstractMetric(BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        """
        If metrics needs some initial heavy computations,
        this can be done here.
        interface is the same as for
        """
        pass

    def __call__(self, y, proba, sample_weight):
        """
        Compute value of metrics
        :param proba: numpy.array of shape [n_samples, n_classes]
            with predicted probabilities (typically returned by predict_proba)
        Events should be passed in the same order, as to method fit
        """
        raise NotImplementedError('To be derived by descendant')


class AbstractBinMetrics(AbstractMetric):
    def __init__(self, n_bins, uniform_features, uniform_label=0):
        """
        Abstract class for bin-based metrics of uniformity.

        :param n_bins: int, number of bins along each axis
        :param uniform_features: list of strings, features along which uniformity is desired ()
        :param uniform_label: int, label of class in which uniformity is desired
            (typically, 0 is bck, 1 is signal)
        """
        self.uniform_label = uniform_label
        self.uniform_features = uniform_features
        self.n_bins = n_bins

    def fit(self, X, y, sample_weight=None):
        """ Prepare different things for fast computation of metrics """
        X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight)
        self._mask = numpy.array(y == self.uniform_label)
        assert sum(self._mask) > 0, 'No event of class, along which uniformity is desired'
        self._masked_weight = sample_weight[self._mask]

        X_part = numpy.array(take_features(X, self.uniform_features))[self._mask, :]
        self._bin_indices = ut.compute_bin_indices(X_part=X_part, n_bins=self.n_bins)
        self._bin_weights = ut.compute_bin_weights(bin_indices=self._bin_indices,
                                                   sample_weight=sample_weight)


class BinBasedSDE(AbstractBinMetrics):
    def __init__(self, n_bins, uniform_features, uniform_label=0, target_rcp=None, power=2.):
        AbstractBinMetrics.__init__(self, n_bins=n_bins,
                                    uniform_features=uniform_features,
                                    uniform_label=uniform_label)
        self.power = power
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]

        result = 0.
        cuts = weighted_percentile(y_pred, self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            bin_efficiencies = ut.compute_bin_efficiencies(y_pred, bin_indices=self._bin_indices,
                                                           cut=cut, sample_weight=self._masked_weight)
            result += ut.weighted_deviation(bin_efficiencies, weights=self._bin_weights, power=self.power)

        return (result / len(cuts)) ** (1. / self.power)


class BinBasedTheil(AbstractBinMetrics):
    def __init__(self, n_bins, uniform_features, uniform_label=0, target_rcp=None, power=2.):
        AbstractBinMetrics.__init__(self, n_bins=n_bins,
                                    uniform_features=uniform_features,
                                    uniform_label=uniform_label)
        self.power = power
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]

        result = 0.
        cuts = weighted_percentile(y_pred, self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            bin_efficiencies = ut.compute_bin_efficiencies(y_pred, bin_indices=self._bin_indices,
                                                           cut=cut, sample_weight=self._masked_weight)
            result += ut.theil(bin_efficiencies, weights=self._bin_weights)
        return result / len(cuts)


class BinBasedCvM(AbstractBinMetrics):
    def __init__(self, n_bins, uniform_features, uniform_label=0, power=2.):
        AbstractBinMetrics.__init__(self, n_bins=n_bins,
                                    uniform_features=uniform_features,
                                    uniform_label=uniform_label)
        self.power = power

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        global_data, global_weight, global_cdf = ut.prepare_distibution(y_pred, weights=self._bin_weights)

        result = 0.
        for bin, bin_weight in enumerate(self._bin_weights):
            if bin_weight <= 0:
                continue
            bin_mask = self._bin_indices == bin
            local_distribution = y_pred[bin_mask]
            local_weights = self._masked_weight[bin_mask]
            result += bin_weight * ut._cvm_2samp_fast(global_data, local_distribution,
                                                      global_weight, local_weights, global_cdf)


class AbstractKnnMetrics(AbstractMetric):
    def __init__(self, uniform_features, n_neighbours=50, uniform_label=0):
        """
        Abstract class for knn-based metrics of uniformity.

        :param n_neighbours: int, number of neighbours
        :param uniform_features: list of strings, features along which uniformity is desired ()
        :param uniform_label: int, label of class in which uniformity is desired
            (typically, 0 is bck, 1 is signal)
        """
        self.uniform_label = uniform_label
        self.uniform_features = uniform_features
        self.n_neighbours = n_neighbours

    def fit(self, X, y, sample_weight=None):
        """ Prepare different things for fast computation of metrics """
        X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight)
        self._mask = numpy.array(y == self.uniform_label)
        assert sum(self._mask) > 0, 'No events of uniform class!'
        self._masked_weight = sample_weight[self._mask]

        X_part = numpy.array(take_features(X, self.uniform_features))[self._mask, :]
        # computing knn indices
        neighbours = NearestNeighbors(n_neighbors=self.n_neighbours, algorithm='kd_tree').fit(X_part)
        _, self._groups_indices = neighbours.kneighbors(X_part)
        self._group_weights = ut.compute_group_weights(self._groups_indices, sample_weight=self._masked_weight)


class KnnBasedSDE(AbstractKnnMetrics):
    def __init__(self, n_neighbours, uniform_features, uniform_label=0, target_rcp=None, power=2.):
        AbstractKnnMetrics.__init__(self, n_neighbours=n_neighbours,
                                    uniform_features=uniform_features,
                                    uniform_label=uniform_label)
        self.power = power
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.target_rcp = numpy.array(self.target_rcp)

        result = 0.
        cuts = weighted_percentile(y_pred, percentiles=1 - self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            groups_efficiencies = ut.compute_group_efficiencies(y_pred, groups_indices=self._groups_indices, cut=cut,
                                                                sample_weight=self._masked_weight)
            result += ut.weighted_deviation(groups_efficiencies, weights=self._group_weights, power=self.power)
        return (result / len(cuts)) ** (1. / self.power)


class KnnBasedTheil(AbstractKnnMetrics):
    def __init__(self, n_neighbours, uniform_features, uniform_label=0, target_rcp=None, power=2.):
        AbstractKnnMetrics.__init__(self, n_neighbours=n_neighbours,
                                    uniform_features=uniform_features,
                                    uniform_label=uniform_label)
        self.power = power
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.target_rcp = numpy.array(self.target_rcp)

        result = 0.
        cuts = weighted_percentile(y_pred, percentiles=1 - self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            groups_efficiencies = ut.compute_group_efficiencies(y_pred, groups_indices=self._groups_indices, cut=cut,
                                                                sample_weight=self._masked_weight)
            result += ut.weighted_deviation(groups_efficiencies, weights=self._group_weights, power=self.power)
        return (result / len(cuts)) ** (1. / self.power)


class KnnBasedCvM(AbstractKnnMetrics):
    def __init__(self, n_neighbours, uniform_features, uniform_label=0, power=2.):
        AbstractKnnMetrics.__init__(self, n_neighbours=n_neighbours,
                                    uniform_features=uniform_features,
                                    uniform_label=uniform_label)
        self.power = power

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]

        result = 0.
        global_data, global_sample_weight, global_cdf = ut.prepare_distibution(y_pred, weights=self._masked_weight)
        for group, group_weight in zip(self._groups_indices, self._group_weights):
            local_distribution = y_pred[group]
            local_sample_weights = self._masked_weight[group]
            result += group_weight * ut._cvm_2samp_fast(global_data, local_distribution,
                                                        global_sample_weight, local_sample_weights, global_cdf)
        return result


# endregion


# region Uniformity metrics (old version)

def sde(y, proba, X, uniform_variables, sample_weight=None, label=1, knn=30):
    """ The most simple way to compute SDE, this is however very slow
    if you need to recompute SDE many times
    :param y: real classes of events, shape = [n_samples]
    :param proba: predicted probabilities, shape = [n_samples, n_classes]
    :param X: pandas.DataFrame with uniform features
    :param uniform_variables: features, along which uniformity is desired, list of strings
    :param sample_weight: weights of events, shape = [n_samples]
    :param label: class, for which uniformity is measured (usually, 0 is bck, 1 is signal)
    :param knn: number of nearest neighbours used in knn

    Example of usage:
    proba = classifier.predict_proba(testX)
    sde(testY, proba=proba, X=testX, uniform_variables=['mass'])
    """
    y, proba = check_arrays(y, proba)
    assert len(y) == len(proba) == len(X), 'Different lengths'

    y = column_or_1d(y)
    sample_weight = check_sample_weight(y, sample_weight=sample_weight)

    X = pandas.DataFrame(X)
    mask = y == label
    groups = computeSignalKnnIndices(uniform_variables=uniform_variables, dataframe=X, is_signal=mask, n_neighbors=knn)
    groups = groups[mask, :]

    return ut.compute_sde_on_groups(proba[:, label], mask=mask, groups_indices=groups,
                                    target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


def theil_flatness(y, proba, X, uniform_variables, sample_weight=None, label=1, knn=30):
    """This is ready-to-use function, and it is quite slow to use many times"""
    sample_weight = check_sample_weight(y, sample_weight=sample_weight)
    mask = y == label
    groups_indices = computeSignalKnnIndices(uniform_variables, X, is_signal=mask, n_neighbors=knn)[mask, :]
    return ut.compute_theil_on_groups(proba[:, label], mask=mask, groups_indices=groups_indices,
                                      target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


def cvm_flatness(y, proba, X, uniform_variables, sample_weight=None, label=1, knn=30):
    """ The most simple way to compute Cramer-von Mises flatness, this is however very slow
    if you need to compute it many times
    :param y: real classes of events, shape = [n_samples]
    :param proba: predicted probabilities, shape = [n_samples, n_classes]
    :param X: pandas.DataFrame with uniform features (i.e. test dataset)
    :param uniform_variables: features, along which uniformity is desired, list of strings
    :param sample_weight: weights of events, shape = [n_samples]
    :param label: class, for which uniformity is measured (usually, 0 is bck, 1 is signal)
    :param knn: number of nearest neighbours used in knn

    Example of usage:
    proba = classifier.predict_proba(testX)
    cvm_flatness(testY, proba=proba, X=testX, uniform_variables=['mass'])
    """
    y, proba = check_arrays(y, proba)
    assert len(y) == len(proba) == len(X), 'Different lengths'
    y = column_or_1d(y)
    sample_weight = check_sample_weight(y, sample_weight=sample_weight)

    X = pandas.DataFrame(X)

    signal_mask = y == label
    groups_indices = computeSignalKnnIndices(uniform_variables=uniform_variables, dataframe=X,
                                             is_signal=signal_mask, n_neighbors=knn)
    groups_indices = groups_indices[signal_mask, :]

    return ut.group_based_cvm(proba[:, label], mask=signal_mask, groups_indices=groups_indices,
                              sample_weight=sample_weight)


# endregion
