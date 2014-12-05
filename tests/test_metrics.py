from __future__ import division, print_function, absolute_import

import numpy
import pandas
from numpy.random.mtrand import RandomState
from scipy.stats import ks_2samp
import sklearn.metrics
from sklearn.metrics import auc

from hep_ml.metrics import bin_to_group_indices, compute_bin_indices, compute_sde_on_bins, \
    compute_sde_on_groups, compute_theil_on_bins, compute_theil_on_groups, \
    _prepare_data, _ks_2samp_fast, ks_2samp_weighted, bin_based_ks, \
    groups_based_ks, cvm_2samp, _cvm_2samp_fast, bin_based_cvm, group_based_cvm

__author__ = 'Alex Rogozhnikov'

random = RandomState()


def generate_binned_dataset(n_samples, n_bins):
    """useful function, generates dataset with bins, groups, random weights.
    This is used to test correlation functions. """
    random = RandomState()
    y = random.uniform(size=n_samples) > 0.5
    pred = random.uniform(size=(n_samples, 2))
    weights = random.exponential(size=(n_samples,))
    bins = random.randint(0, n_bins, n_samples)
    groups = bin_to_group_indices(bin_indices=bins, mask=(y == 1))
    return y, pred, weights, bins, groups


def test_bin_to_group_indices(size=100, bins=10):
    bin_indices = RandomState().randint(0, bins, size=size)
    mask = RandomState().randint(0, 2, size=size) > 0.5
    group_indices = bin_to_group_indices(bin_indices, mask=mask)
    assert numpy.sum([len(group) for group in group_indices]) == numpy.sum(mask)
    a = numpy.sort(numpy.concatenate(group_indices))
    b = numpy.where(mask > 0.5)[0]
    assert numpy.all(a == b), 'group indices are computed wrongly'


def test_bins(size=500, n_bins=10):
    columns = ['var1', 'var2']
    df = pandas.DataFrame(random.uniform(size=(size, 2)), columns=columns)
    x_limits = numpy.linspace(0, 1, n_bins + 1)[1:-1]
    bins = compute_bin_indices(df, columns, [x_limits, x_limits])
    assert numpy.all(0 <= bins) and numpy.all(bins < n_bins * n_bins), "the bins with wrong indices appeared"


def test_compare_sde_computations(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_binned_dataset(n_samples=n_samples, n_bins=n_bins)
    target_efficiencies = RandomState().uniform(size=3)
    a = compute_sde_on_bins(pred[:, 1], mask=(y == 1), bin_indices=bins,
                            target_efficiencies=target_efficiencies, sample_weight=weights)
    b = compute_sde_on_groups(pred[:, 1], mask=(y == 1), groups_indices=groups,
                              target_efficiencies=target_efficiencies, sample_weight=weights)
    assert numpy.allclose(a, b)


def test_theil(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_binned_dataset(n_samples=n_samples, n_bins=n_bins)
    a = compute_theil_on_bins(pred[:, 1], y == 1, bins, [0.5, 0.78], sample_weight=weights)
    b = compute_theil_on_groups(pred[:, 1], y == 1, groups, [0.5, 0.78], sample_weight=weights)
    assert numpy.allclose(a, b)


def test_ks2samp_fast(size=1000):
    y1 = RandomState().uniform(size=size)
    y2 = y1[RandomState().uniform(size=size) > 0.5]
    a = ks_2samp(y1, y2)[0]
    prep_data, prep_weights, prep_F = _prepare_data(y1, numpy.ones(len(y1)))
    b = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), F1=prep_F)
    c = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), F1=prep_F)
    d = ks_2samp_weighted(y1, y2, numpy.ones(len(y1)) / 3, numpy.ones(len(y2)) / 4)
    assert numpy.allclose(a, b, rtol=1e-2, atol=1e-3)
    assert numpy.allclose(b, c)
    assert numpy.allclose(b, d)
    print('ks2samp is ok')


def test_ks(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_binned_dataset(n_samples=n_samples, n_bins=n_bins)
    mask = y == 1
    a = bin_based_ks(pred[:, 1], mask=mask, sample_weight=weights, bin_indices=bins)
    b = groups_based_ks(pred[:, 1], mask=mask, sample_weight=weights, groups_indices=groups)
    assert numpy.allclose(a, b)


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


def test_cvm(size=1000):
    y_pred = random.uniform(size=size)
    y = random.uniform(size=size) > 0.5
    sample_weight = random.exponential(size=size)
    bin_indices = random.randint(0, 10, size=size)
    mask = y == 1
    groups_indices = bin_to_group_indices(bin_indices=bin_indices, mask=mask)
    cvm1 = bin_based_cvm(y_pred[mask], sample_weight=sample_weight[mask], bin_indices=bin_indices[mask])
    cvm2 = group_based_cvm(y_pred, mask=mask, sample_weight=sample_weight, groups_indices=groups_indices)
    assert numpy.allclose(cvm1, cvm2)


def test_cvm_sde_limit(size=2000):
    """ Checks that in the limit CvM coincides with MSE """
    effs = numpy.linspace(0, 1, 2000)
    y_pred = random.uniform(size=size)
    y = random.uniform(size=size) > 0.5
    sample_weight = random.exponential(size=size)
    bin_indices = random.randint(0, 10, size=size)
    y_pred += bin_indices * random.uniform()
    mask = y == 1

    val1 = bin_based_cvm(y_pred[mask], sample_weight=sample_weight[mask], bin_indices=bin_indices[mask])
    val2 = compute_sde_on_bins(y_pred, mask=mask, bin_indices=bin_indices, target_efficiencies=effs,
                               sample_weight=sample_weight)

    assert numpy.allclose(val1, val2 ** 2, atol=1e-3, rtol=1e-2)


