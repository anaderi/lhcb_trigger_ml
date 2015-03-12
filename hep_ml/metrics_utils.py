from __future__ import division, print_function, absolute_import

import numpy
from .commonutils import check_sample_weight, sigmoid_function, compute_cut_for_efficiency
from sklearn.utils.validation import column_or_1d

__author__ = 'Alex Rogozhnikov'


def check_metrics_arguments(y_true, y_pred, sample_weight, two_class=True, binary_pred=True):
    """
    Checks the arguments passed to metrics
    :param y_true:
    :param y_pred:
    :param sample_weight:
    :param two_class:
    :param binary_pred:
    :return:
    """
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


def prepare_distibution(data, weights):
    """Prepares the distribution to be used later in KS and CvM,
    merges equal data, computes (summed) weights and cumulative distribution.
    All output arrays are of same length and correspond to each other."""
    weights = weights / numpy.sum(weights)
    prepared_data, indices = numpy.unique(data, return_inverse=True)
    prepared_weights = numpy.bincount(indices, weights=weights)
    prepared_cdf = compute_cdf(prepared_weights)
    return prepared_data, prepared_weights, prepared_cdf


# region Helpful functions to work with bins and groups

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


def compute_bin_indices(X_part, bin_limits=None, n_bins=20):
    """For arbitrary number of variables computes the indices of data,
    the indices are unique numbers of bin from zero to \prod_j (len(bin_limits[j])+1)
    Example:
        var_names = ["M2AB", "M2AC"]
        bin_limits = [numpy.linspace(0, 1, 21), numpy.linspace(0, 1, 21)]

    If bin_limits is not provided, they are computed using mask and n_bins
    """
    if bin_limits is None:
        bin_limits = []
        for variable_data in range(X_part.shape[1]):
            bin_limits.append(numpy.linspace(numpy.min(variable_data), numpy.max(variable_data), n_bins + 1)[1: -1])

    bin_indices = numpy.zeros(len(X_part), dtype=numpy.int)
    for axis, bin_limits_axis in enumerate(bin_limits):
        bin_indices *= (len(bin_limits_axis) + 1)
        bin_indices += numpy.searchsorted(bin_limits_axis, X_part[:, axis])

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


# endregion


# region Supplementary uniformity-related functions (to measure flatness of predictions)

def compute_cdf(ordered_weights):
    """Computes cumulative distribution function (CDF) by ordered weights,
    be sure that sum(ordered_weights) == 1
    """
    return numpy.cumsum(ordered_weights) - 0.5 * ordered_weights


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


def compute_bin_efficiencies(y_score, bin_indices, cut, sample_weight, minlength=None):
    """Efficiency of bin = total weight of (signal) events that passed the cut
    in the bin / total weight of signal events in the bin.
    Returns small negative number for empty bins"""
    y_score = column_or_1d(y_score)
    assert len(y_score) == len(sample_weight) == len(bin_indices), "different size"
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


# endregion


# region Special methods for uniformity metrics

def compute_sde_on_bins(y_pred, mask, bin_indices, target_efficiencies, power=2., sample_weight=None):
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
    prepared_data, prepared_weight, prep_F = prepare_distibution(y_pred, weights=sample_weight)

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
    prepared_data, prepared_weight, prep_F = prepare_distibution(y_pred[mask], weights=sample_weight[mask])

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
    This implementation sorts the arrays each time, so it's slow"""
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
    assert len(y_pred) == len(sample_weight) == len(bin_indices)
    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)

    result = 0.
    global_data, global_weight, global_F = prepare_distibution(y_pred, weights=sample_weight)

    for bin, bin_weight in enumerate(bin_weights):
        if bin_weight <= 0:
            continue
        bin_mask = bin_indices == bin
        local_distribution = y_pred[bin_mask]
        local_weights = sample_weight[bin_mask]
        result += bin_weight * _cvm_2samp_fast(global_data, local_distribution,
                                               global_weight, local_weights, global_F)

    return result


def group_based_cvm(y_pred, mask, sample_weight, groups_indices):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    group_weights = compute_group_weights(groups_indices, sample_weight=sample_weight)

    result = 0.
    global_data, global_weight, global_F = prepare_distibution(y_pred[mask], weights=sample_weight[mask])
    for group, group_weight in zip(groups_indices, group_weights):
        local_distribution = y_pred[group]
        local_weights = sample_weight[group]
        result += group_weight * _cvm_2samp_fast(global_data, local_distribution,
                                                 global_weight, local_weights, global_F)
    return result







    # endregion