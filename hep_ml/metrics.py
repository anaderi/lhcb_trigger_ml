# About

# this module contains different metrics of uniformity
# and the metrics of quality as well (which support weights, actually)

from __future__ import division, print_function

import numpy
import pandas
from sklearn.metrics import auc, roc_curve
from sklearn.utils.validation import column_or_1d, check_arrays

from .commonutils import check_sample_weight, compute_cut_for_efficiency, computeSignalKnnIndices, sigmoid_function


__author__ = 'Alex Rogozhnikov'

__all__ = ['sde', 'cvm_flatness', 'cvm_2samp', 'theil']

# TODO simpler interfaces
# TODO uniformity in usage of masks

# region Utilities


def compute_cdf(ordered_weights):
    """Computes cumulative distribution function (CDF) by ordered weights,
    be sure that sum(ordered_weights) == 1
    """
    return numpy.cumsum(ordered_weights) - 0.5 * ordered_weights

#endregion


""" README on quality metrics

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

#region Quality metrics


def check_metrics_arguments(y_true, y_pred, sample_weight, two_class=True, binary_pred=True):
    sample_weight = check_sample_weight(y_true, sample_weight=sample_weight)
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    assert len(y_true) == len(y_pred), \
        'The lengths of y_true and y_pred are different: %i and %i' % (len(y_true), len(y_pred))
    if two_class:
        assert numpy.in1d(y_true, [0, 1]).all(), 'The y_true array should contain only two labels: 0 and 1, ' \
                                                 'it contains:' + str(numpy.unique(y_true))
    if binary_pred:
        assert numpy.in1d(y_pred, [0, 1]).all(), 'The y_pred array should contain only two labels: 0 and 1, ' \
                                                 'it contains:' + str(numpy.unique(y_pred))
    return y_true, y_pred, sample_weight


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


def roc_auc_score(y_true, y_score, sample_weight=None):
    """The same as sklearn.metrics.roc_auc_score, but supports weights """
    if len(numpy.unique(y_true)) != 2:
        raise ValueError("AUC is defined for binary classification only")
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    return auc(fpr, tpr, reorder=True)


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
        check_metrics_arguments(y_true, y_score, sample_weight=sample_weight, two_class=True, binary_pred=True)
    s, b = compute_sb(y_true, y_score, sample_weight=sample_weight)
    return s / numpy.sqrt(s + b)


def optimal_sensitivity(y_true, y_score, sample_weight=None):
    """s,b are normalized to be in [0,1] """
    b, s, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    # skipping initial zero
    s, b = s[1:], b[1:]
    return numpy.max(s / numpy.sqrt(s + b))


#endregion


#region Helpful functions to work with bins and groups

"""

the difference between bins and groups: each event belongs to one and only one bin,
in the case of groups each event may belong to several groups.
Knn is one particular case of groups, bins can be reduced to groups either

Bin_indices is an array, where for each event it's bin is written:
bin_indices = [0,0,1,2,2,4]

Group_indices is list, each item is indices of events in some group
group_indices = [[0,1], [2], [3,4], [5]]

While bin indices are computed for all the events together, group indices
are typically computed only for events of some particular class.
"""


def compute_bin_indices(X, var_names, bin_limits=None, mask=None, n_bins=20):
    """For arbitrary number of variables computes the indices of data,
    the indices are unique numbers of bin from zero to \prod_j (len(bin_limits[j])+1)
    Example:
        var_names = ["M2AB", "M2AC"]
        bin_limits = [numpy.linspace(0, 1, 21), numpy.linspace(0, 1, 21)]

    If bin_limits is not provided, they are computed using mask and n_bins
    """
    if bin_limits is None:
        bin_limits = []
        for var_name in var_names:
            var_data = X.loc[:, var_name] if mask is None else X.loc[mask, var_name]
            bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])

    assert len(var_names) == len(bin_limits), "Different size of arrays"
    bin_indices = numpy.zeros(len(X), dtype=numpy.int)
    for var_name, bin_limits_axis in zip(var_names, bin_limits):
        bin_indices *= (len(bin_limits_axis) + 1)
        bin_indices += numpy.searchsorted(bin_limits_axis, X[var_name])
    return bin_indices


def bin_to_group_indices(bin_indices, mask):
    """ Transforms bin_indices into group indices, skips empty bins
    :type bin_indices: numpy.array, each element in index of bin this event belongs, shape = [n_samples]
    :type mask: numpy.array, boolean mask of indices to split into bins, shape = [n_samples]
    :rtype: list(numpy.array), each element is indices of elements in some bin
    """
    assert len(bin_indices) == len(mask), "Different length"
    bins_id = numpy.unique(bin_indices)
    result = list()
    for bin_id in bins_id:
        result.append(numpy.where(mask & (bin_indices == bin_id))[0])
    return result


#endregion


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


#region Supplementary uniformity-related functions (to measure flatness of predictions)

def compute_bin_weights(bin_indices, sample_weight):
    assert len(bin_indices) == len(sample_weight), 'Different lengths of array'
    result = numpy.bincount(bin_indices, weights=sample_weight)
    return result / numpy.sum(result)


def compute_divided_weight(group_indices, sample_weight):
    """Divided weight takes into account that different events
    are met different number of times """
    indices = numpy.concatenate(group_indices)
    occurences = numpy.bincount(indices, minlength=len(sample_weight))
    return sample_weight / numpy.maximum(occurences, 1)


def compute_group_weights(group_indices, sample_weight):
    """
    Group weight = sum of divided weights of indices inside that group.
    """
    divided_weight = compute_divided_weight(group_indices, sample_weight=sample_weight)
    result = numpy.zeros(len(group_indices))
    for i, group in enumerate(group_indices):
        result[i] = numpy.sum(divided_weight[group])
    return result / numpy.sum(result)


def compute_bin_efficiencies(y_score, bin_indices, cut, sample_weight=None, minlength=None):
    """Efficiency of bin = total weight of (signal) events that passed the cut
    in the bin / total weight of signal events in the bin.
    Returns small negative number for empty bins"""
    y_score = column_or_1d(y_score)
    assert len(y_score) == len(bin_indices), "different size"
    sample_weight = check_sample_weight(y_score, sample_weight=sample_weight)
    if minlength is None:
        minlength = numpy.max(bin_indices) + 1

    bin_total = numpy.bincount(bin_indices, weights=sample_weight, minlength=minlength)
    passed_cut = y_score > cut
    bin_passed_cut = numpy.bincount(bin_indices[passed_cut],
                                    weights=sample_weight[passed_cut], minlength=minlength)
    return bin_passed_cut / numpy.maximum(bin_total, 1)


def compute_group_efficiencies(y_score, groups_indices, cut, sample_weight=None, smoothing=0.0):
    y_score = column_or_1d(y_score)
    sample_weight = check_sample_weight(y_score, sample_weight=sample_weight)
    # with smoothing=0, this is
    passed_cut = sigmoid_function(y_score - cut, width=smoothing)

    if isinstance(groups_indices, numpy.ndarray) and numpy.ndim(groups_indices) == 2:
        # this speedup is specially for knn
        result = numpy.average(numpy.take(passed_cut, groups_indices),
                               weights=numpy.take(sample_weight, groups_indices),
                               axis=1)
    else:
        result = numpy.zeros(len(groups_indices))
        for i, group in enumerate(groups_indices):
            result[i] = numpy.average(passed_cut[group], weights=sample_weight[group])
    return result


def weighted_deviation(a, weights, power=2.):
    """ sum weight * |x - x_mean|^power """
    mean = numpy.average(a, weights=weights)
    return numpy.average(numpy.abs(mean - a) ** power, weights=weights)


#endregion


#region SDE

def compute_sde_on_bins(y_pred, mask, bin_indices, target_efficiencies, power=2., sample_weight=None):
    """
    See article [1] for details on SDE

    :param y_pred: array-like of shape [n_samples] with floats predictions
    :param mask: array-like of shape [n_samples] with bool.
        Needed to mark events in which uniformity is desired
    :param bin_indices: array-like of shape [n_samples]
    :param target_efficiencies: array-like with efficiencies used in SDE
    :param power: float (default: 2)
    :param sample_weight: array-like of shape [n_samples] with weights of events.
    :return: float, computed SDE value.
    """
    # skip check for a while
    # ignoring events from other classes
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    y_pred = y_pred[mask]
    bin_indices = bin_indices[mask]
    sample_weight = sample_weight[mask]

    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)
    cuts = compute_cut_for_efficiency(target_efficiencies, mask=numpy.ones(len(y_pred), dtype=bool),
                                      y_pred=y_pred, sample_weight=sample_weight)

    result = 0.
    for cut in cuts:
        bin_efficiencies = compute_bin_efficiencies(y_pred, bin_indices=bin_indices,
                                                    cut=cut, sample_weight=sample_weight)
        result += weighted_deviation(bin_efficiencies, weights=bin_weights, power=power)

    return (result / len(cuts)) ** (1. / power)


def compute_sde_on_groups(y_pred, mask, groups_indices, target_efficiencies, sample_weight=None, power=2.):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    group_weights = compute_group_weights(groups_indices, sample_weight=sample_weight)
    cuts = compute_cut_for_efficiency(target_efficiencies, mask=mask, y_pred=y_pred, sample_weight=sample_weight)
    sde = 0.
    for cut in cuts:
        group_efficiencies = compute_group_efficiencies(y_pred, groups_indices=groups_indices,
                                                        cut=cut, sample_weight=sample_weight)
        sde += weighted_deviation(group_efficiencies, weights=group_weights, power=power)
    return (sde / len(cuts)) ** (1. / power)


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

    return compute_sde_on_groups(proba[:, label], mask=mask, groups_indices=groups,
                                 target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


#endregion


#region Theil Index of Efficiency

def theil(x, weights):
    """Theil index of array with regularization"""
    assert numpy.all(x >= 0), "negative numbers can't be used in Theil"
    x_mean = numpy.average(x, weights=weights) + 1e-100
    normed = x / x_mean
    normed[normed < 1e-10] = 1e-10
    return numpy.average(normed * numpy.log(normed), weights=weights)


def compute_theil_on_bins(y_pred, mask, bin_indices, target_efficiencies, sample_weight):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)

    # ignoring events from other classes
    y_pred = y_pred[mask]
    bin_indices = bin_indices[mask]
    sample_weight = sample_weight[mask]

    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)
    cuts = compute_cut_for_efficiency(target_efficiencies, mask=numpy.ones(len(y_pred), dtype=bool),
                                      y_pred=y_pred, sample_weight=sample_weight)
    result = 0.
    for cut in cuts:
        bin_efficiencies = compute_bin_efficiencies(y_pred, bin_indices=bin_indices,
                                                    cut=cut, sample_weight=sample_weight)
        result += theil(bin_efficiencies, weights=bin_weights)
    return result / len(cuts)


def compute_theil_on_groups(y_pred, mask, groups_indices, target_efficiencies, sample_weight):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    groups_weights = compute_group_weights(groups_indices, sample_weight=sample_weight)
    cuts = compute_cut_for_efficiency(target_efficiencies, mask=mask,
                                      y_pred=y_pred, sample_weight=sample_weight)
    result = 0.
    for cut in cuts:
        groups_efficiencies = compute_group_efficiencies(y_pred, groups_indices, cut, sample_weight=sample_weight)
        result += theil(groups_efficiencies, groups_weights)
    return result / len(cuts)


def theil_flatness(y, proba, X, uniform_variables, sample_weight=None, label=1, knn=30):
    """This is ready-to-use function, and it is quite slow to use many times"""
    sample_weight = check_sample_weight(y, sample_weight=sample_weight)
    mask = y == label
    groups_indices = computeSignalKnnIndices(uniform_variables, X, is_signal=mask, n_neighbors=knn)[mask, :]
    return compute_theil_on_groups(proba[:, label], mask=mask, groups_indices=groups_indices,
                                   target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


#endregion


#region Similarity-based measures of flatness: KS, CvM

def _prepare_data(data, weights):
    """Prepares the distribution to be used later in KS and CvM"""
    weights = weights / numpy.sum(weights)
    prepared_data = numpy.unique(data)
    indices = numpy.searchsorted(prepared_data, data)
    prepared_weights = numpy.bincount(indices, weights=weights)
    F = compute_cdf(prepared_weights)
    return prepared_data, prepared_weights, F


def _ks_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, F1):
    """Pay attention - prepared data should not only be sorted,
    but equal items should be merged (by summing weights),
    data2 should not have elements larger then max(prepared_data1) """
    indices = numpy.searchsorted(prepared_data1, data2)
    weights2 /= numpy.sum(weights2)
    prepared_weights2 = numpy.bincount(indices, weights=weights2, minlength=len(prepared_data1))
    F2 = compute_cdf(prepared_weights2)
    return numpy.max(numpy.abs(F1 - F2))


def ks_2samp_weighted(data1, data2, weights1, weights2):
    x = numpy.unique(numpy.concatenate([data1, data2]))
    weights1 /= numpy.sum(weights1)
    weights2 /= numpy.sum(weights2)
    inds1 = numpy.searchsorted(x, data1)
    inds2 = numpy.searchsorted(x, data2)
    w1 = numpy.bincount(inds1, weights=weights1, minlength=len(x))
    w2 = numpy.bincount(inds2, weights=weights2, minlength=len(x))
    F1 = compute_cdf(w1)
    F2 = compute_cdf(w2)
    return numpy.max(numpy.abs(F1 - F2))


def bin_based_ks(y_pred, mask, sample_weight, bin_indices):
    """Kolmogorov-Smirnov flatness on bins"""
    assert len(y_pred) == len(sample_weight) == len(bin_indices) == len(mask)
    y_pred = y_pred[mask]
    sample_weight = sample_weight[mask]
    bin_indices = bin_indices[mask]

    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)
    prepared_data, prepared_weight, prep_F = _prepare_data(y_pred, weights=sample_weight)

    result = 0.
    for bin, bin_weight in enumerate(bin_weights):
        if bin_weight <= 0:
            continue
        local_distribution = y_pred[bin_indices == bin]
        local_weights = sample_weight[bin_indices == bin]
        result += bin_weight * \
                  _ks_2samp_fast(prepared_data, local_distribution, prepared_weight, local_weights, prep_F)
    return result


def groups_based_ks(y_pred, mask, sample_weight, groups_indices):
    """Kolmogorov-Smirnov flatness on groups """
    assert len(y_pred) == len(sample_weight) == len(mask)
    group_weights = compute_group_weights(groups_indices, sample_weight=sample_weight)
    prepared_data, prepared_weight, prep_F = _prepare_data(y_pred[mask], weights=sample_weight[mask])

    result = 0.
    for group_weight, group_indices in zip(group_weights, groups_indices):
        local_distribution = y_pred[group_indices]
        local_weights = sample_weight[group_indices]
        result += group_weight * \
                  _ks_2samp_fast(prepared_data, local_distribution, prepared_weight, local_weights, prep_F)
    return result


def cvm_2samp(data1, data2, weights1=None, weights2=None, power=2.):
    """A handmade function for Cramer-von Mises similarity,
    \int |F_2 - F_1|^p dF_1
    This implementation sorts the arrays each time"""
    weights1 = check_sample_weight(data1, sample_weight=weights1)
    weights2 = check_sample_weight(data2, sample_weight=weights2)
    weights1 /= numpy.sum(weights1)
    weights2 /= numpy.sum(weights2)
    data = numpy.unique(numpy.concatenate([data1, data2]))
    bins = numpy.append(data, data[-1] + 1)
    weights1_new = numpy.histogram(data1, bins=bins, weights=weights1)[0]
    weights2_new = numpy.histogram(data2, bins=bins, weights=weights2)[0]
    F1 = compute_cdf(weights1_new)
    F2 = compute_cdf(weights2_new)
    return numpy.average(numpy.abs(F1 - F2) ** power, weights=weights1_new)


def _cvm_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, F1, power=2.):
    """Pay attention - prepared data should not only be sorted,
    but equal items should be merged (by summing weights) """
    indices = numpy.searchsorted(prepared_data1, data2)
    weights2 /= numpy.sum(weights2)
    prepared_weights2 = numpy.bincount(indices, weights=weights2, minlength=len(prepared_data1))
    F2 = compute_cdf(prepared_weights2)
    return numpy.average(numpy.abs(F1 - F2) ** power, weights=prepared_weights1)


def bin_based_cvm(y_pred, sample_weight, bin_indices):
    """Cramer-von Mises similarity, quite slow meanwhile"""
    # TODO get rid of recomputing each time bin_mask
    assert len(y_pred) == len(sample_weight) == len(bin_indices)
    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)

    result = 0.
    global_data, global_weight, global_F = _prepare_data(y_pred, weights=sample_weight)

    for bin, bin_weight in enumerate(bin_weights):
        if bin_weight <= 0:
            continue
        bin_mask = bin_indices == bin
        local_distribution = y_pred[bin_mask]
        local_weights = sample_weight[bin_mask]
        # result += bin_weight * cvm_2samp(y_pred, local_distribution, sample_weight, local_weights)
        result += bin_weight * _cvm_2samp_fast(global_data, local_distribution,
                                               global_weight, local_weights, global_F)

    return result


def group_based_cvm(y_pred, mask, sample_weight, groups_indices):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    group_weights = compute_group_weights(groups_indices, sample_weight=sample_weight)

    result = 0.
    global_data, global_weight, global_F = _prepare_data(y_pred[mask], weights=sample_weight[mask])
    for group, group_weight in zip(groups_indices, group_weights):
        local_distribution = y_pred[group]
        local_weights = sample_weight[group]
        # result += group_weight * cvm_2samp(y_pred[mask], local_distribution, sample_weight[mask], local_weights)
        result += group_weight * _cvm_2samp_fast(global_data, local_distribution,
                                                 global_weight, local_weights, global_F)
    return result


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

    return group_based_cvm(proba[:, label], mask=signal_mask, groups_indices=groups_indices,
                           sample_weight=sample_weight)

#endregion
