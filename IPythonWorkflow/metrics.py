# About

# this module contains different metrics of uniformity
# and the metrics of quality as well (which support weights, actually)

from __future__ import division
from __future__ import print_function

import warnings

import numpy
import pandas
from sklearn.metrics import auc
from sklearn.utils.validation import column_or_1d, check_arrays
from numpy.random.mtrand import RandomState
from scipy.stats import ks_2samp

from commonutils import check_sample_weight, compute_cut_for_efficiency, computeSignalKnnIndices


__author__ = 'Alex Rogozhnikov'

__all__ = ['sde', 'cvm_flatness', 'cvm_2samp', 'theil']

#TODO simpler interfaces
#TODO uniformity in usage of masks

#region Utilities


def compute_cdf(ordered_weights):
    """Computes cumulative distribution function (CDF) by ordered weights,
    be sure that sum(ordered_weights) == 1
    """
    return numpy.cumsum(ordered_weights) - 0.5 * ordered_weights

def generate_test_dataset(n_samples, n_bins):
    random = RandomState()
    y = random.uniform(size=n_samples) > 0.5
    pred = random.uniform(size=(n_samples, 2))
    weights = random.exponential(size=(n_samples,))
    bins = random.randint(0, n_bins, n_samples)
    groups = bin_to_group_indices(bin_indices=bins, mask=(y == 1))
    return y, pred, weights, bins, groups

#endregion


""" README on quality metrics

Some notation used here
IsSignal - is really signal
AsSignal - classified as signal
IsBackgroundAsSignal - background, but classified as signal
... and so on. Cute, right?

tpr = s = isSasS / isS
fpr = b = isBasS / isB

signal efficiency = tpr = s
background efficiency = isBasB / isB = 1 - fpr
background rejection = background efficiency

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


def roc_curve(y_true, y_score, sample_weight=None):
    """ The same as sklearn.metrics.roc_curve, but this one supports weights,
    function will be removed when move to sklearn 0.15"""
    y_true, y_score, sample_weight = \
        check_metrics_arguments(y_true, y_score, sample_weight=sample_weight,
                                two_class=True, binary_pred=False)
    order = numpy.argsort(-y_score)
    thresholds = y_score[order]
    y_true = y_true[order]
    sample_weight = sample_weight[order]
    tpr = numpy.insert(numpy.cumsum(sample_weight * y_true), 0, 0.)
    tpr /= tpr[-1]
    fpr = numpy.insert(numpy.cumsum(sample_weight * (1 - y_true)), 0, 0.)
    fpr /= fpr[-1]
    thresholds = numpy.insert(thresholds, 0, thresholds[0] + 1.)
    # For physicists: fpr = b, tpr = s
    return fpr, tpr, thresholds


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
    """ Returns s / sqrt{s+b} """
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


def test_roc_curve(size=100):
    import sklearn.metrics
    y = (numpy.random.random(size) > 0.5) * 1
    pred = numpy.random.random(size) * 10
    fpr1, tpr1, thr1 = sklearn.metrics.roc_curve(y, pred)
    fpr2, tpr2, thr2 = roc_curve(y, pred)
    # this is insufficient test really
    assert auc(fpr1, tpr1) == auc(fpr2, tpr2)

test_roc_curve(100)

#endregion


#region Helpful functions to work with bins and groups

"""

the difference between bins and groups: each event belongs to one and only one bin,
in the case of groups each event may belong to several groups.
Knn is one particular case of groups, bins can be reduced to groups either

Bin_indices is an array, where for each event it's bin is written:
bin_indices = [0,0,1,2,2,4]

Group_indices is an list, each item is indices of events in some group
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


def test_bin_to_group_indices(size=100, bins=10):
    bin_indices = RandomState().randint(0, bins, size=size)
    mask = RandomState().randint(0, 2, size=size) > 0.5
    group_indices = bin_to_group_indices(bin_indices, mask=mask)
    assert numpy.sum([len(group) for group in group_indices]) == numpy.sum(mask)
    a = numpy.sort(numpy.concatenate(group_indices))
    b = numpy.where(mask > 0.5)[0]
    assert numpy.all(a == b), 'group indices are computed wrongly'

test_bin_to_group_indices()


def test_bins(size=500, n_bins=10):
    columns = ['var1', 'var2']
    df = pandas.DataFrame(numpy.random.random((size, 2)), columns=columns)
    x_limits = numpy.linspace(0, 1, n_bins + 1)[1:-1]
    bins = compute_bin_indices(df, columns, [x_limits, x_limits])
    assert numpy.all(0 <= bins) and numpy.all(bins < n_bins * n_bins), "the bins with wrong indices appeared"

test_bins()

#endregion


"""
README on flatness

this metrics are unfortunately more complicated than usual ones
and require more information: not only predictions and classes,
but also mass (or other variables along which we want to split data)

Here we compute the different metrics of uniformity of predictions:

SDE (formerly MSEE) - the standard deviation of efficiency
Theil - Theil index of Efficiency (Theil index is used in economics)
KS  - based on Kolmogorov-Smirnov distance between distributions
CVM - based on Cramer-von Mises similarity between distributions

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


def compute_group_efficiencies(y_score, groups_indices, cut, sample_weight=None):
    y_score = column_or_1d(y_score)
    sample_weight = check_sample_weight(y_score, sample_weight=sample_weight)
    passed_cut = y_score > cut

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

    return 10 * (result / len(cuts)) ** (1./power)


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
    return 10 * (sde / len(cuts)) ** (1./power)


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


def test_compare_sde_computations(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_test_dataset(n_samples=n_samples, n_bins=n_bins)
    target_efficiencies = RandomState().uniform(size=3)
    a = compute_sde_on_bins(pred[:, 1], mask=(y == 1), bin_indices=bins,
                            target_efficiencies=target_efficiencies, sample_weight=weights)
    b = compute_sde_on_groups(pred[:, 1], mask=(y == 1), groups_indices=groups,
                              target_efficiencies=target_efficiencies, sample_weight=weights)
    assert numpy.allclose(a, b)

test_compare_sde_computations()


#endregion


#region Theil Index of Efficiency

def theil(x, weights):
    """Theil index of array with regularization"""
    assert numpy.all(x >= 0)
    x_mean = numpy.average(x, weights=weights)
    normed = x / x_mean
    normed[normed < 1e-10] = 1e-10
    return numpy.average(normed * numpy.log(normed), weights=weights)


def compute_theil_on_bins(y_pred, mask, bin_indices, target_efficiencies, sample_weight):
    warnings.warn('Theil on bins is in experimental version', UserWarning)
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


def test_theil(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_test_dataset(n_samples=n_samples, n_bins=n_bins)
    a = compute_theil_on_bins(pred[:, 1], y == 1, bins, [0.5, 0.78], sample_weight=weights)
    b = compute_theil_on_groups(pred[:, 1], y == 1, groups, [0.5, 0.78], sample_weight=weights)
    assert numpy.allclose(a, b)

test_theil()

#endregion


#region Similarity-based measures of flatness, KS, CvM

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


def test_ks2samp_fast(size=1000):
    y1 = RandomState().uniform(size=size)
    y2 = y1[RandomState().uniform(size=size) > 0.5]
    a = ks_2samp(y1, y2)[0]
    prep_data, prep_weights, prep_F = _prepare_data(y1, numpy.ones(len(y1)))
    b = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), F1=prep_F)
    c = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), F1=prep_F)
    assert numpy.allclose(a, b, rtol=1e-2, atol=1e-3)
    assert numpy.allclose(b, c)

test_ks2samp_fast()


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


def test_ks(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_test_dataset(n_samples=n_samples, n_bins=n_bins)
    mask = y == 1
    a = bin_based_ks(pred[:, 1], mask=mask, sample_weight=weights, bin_indices=bins)
    b = groups_based_ks(pred[:, 1], mask=mask, sample_weight=weights, groups_indices=groups)
    assert numpy.allclose(a, b)

test_ks()


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
    assert numpy.all(F1 >= 0.) and numpy.all(F1 <= 1.001)
    assert numpy.all(F2 >= 0.) and numpy.all(F2 <= 1.001)
    return numpy.average(numpy.abs(F1 - F2) ** power, weights=weights1_new)


def _cvm_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, F1, power=2.):
    """Pay attention - prepared data should not only be sorted,
    but equal items should be merged (by summing weights) """
    indices = numpy.searchsorted(prepared_data1, data2)
    weights2 /= numpy.sum(weights2)
    prepared_weights2 = numpy.bincount(indices, weights=weights2, minlength=len(prepared_data1))
    F2 = compute_cdf(prepared_weights2)
    return numpy.average(numpy.abs(F1 - F2) ** power, weights=prepared_weights1)


def test_fast_cvm(n_samples=1000):
    random = RandomState()
    data1 = random.uniform(size=n_samples)
    weights1 = random.uniform(size=n_samples)
    mask = random.uniform(size=n_samples) > 0.5
    data2 = data1[mask]
    weights2 = weights1[mask]
    a = cvm_2samp(data1, data2, weights1, weights2)
    prepared_data1, prepared_weights1, F1 = _prepare_data(data1, weights1)
    b = _cvm_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, F1=F1)
    assert numpy.allclose(a, b)


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

    return group_based_cvm(proba[:, label], mask=signal_mask, groups_indices=groups_indices, sample_weight=sample_weight)


def check_cvm(size=1000):
    y_pred = numpy.random.random(size)
    y = numpy.random.random(size) > 0.5
    sample_weight = numpy.random.exponential(size=size)
    bin_indices = numpy.random.randint(0, 10, size=size)
    mask = y == 1
    groups_indices = bin_to_group_indices(bin_indices=bin_indices, mask=mask)
    cvm1 = bin_based_cvm(y_pred[mask], sample_weight=sample_weight[mask], bin_indices=bin_indices[mask])
    cvm2 = group_based_cvm(y_pred, mask=mask, sample_weight=sample_weight, groups_indices=groups_indices)
    assert numpy.allclose(cvm1, cvm2)

check_cvm()


def check_limit(size=2000):
    """ Checks that in the limit CvM coincides with MSE """
    effs = numpy.linspace(0, 1, 2000)
    y_pred = numpy.random.random(size)
    y = numpy.random.random(size) > 0.5
    sample_weight = numpy.random.exponential(size=size)
    bin_indices = numpy.random.randint(0, 10, size=size)
    y_pred += bin_indices * numpy.random.random()
    mask = y == 1

    val1 = bin_based_cvm(y_pred[mask], sample_weight=sample_weight[mask], bin_indices=bin_indices[mask])
    val2 = compute_sde_on_bins(y_pred, mask=mask, bin_indices=bin_indices, target_efficiencies=effs,
                               sample_weight=sample_weight)

    assert numpy.allclose(val1, (val2 / 10) ** 2, atol=1e-3, rtol=1e-2)


check_limit()

#endregion
