# About

# This file contains some helpful functions and classes
# which are often used (by other modules)

from __future__ import print_function
from __future__ import division
from sklearn.utils.validation import check_arrays

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import math
import io
import numpy
import pandas
from numpy.random.mtrand import RandomState
from scipy.special import expit
import sklearn.cross_validation
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.unsupervised import NearestNeighbors

__author__ = "Alex Rogozhnikov"


def execute_notebook(filename):
    """Allows one to execute cell-by-cell some IPython notebook provided its name"""
    from IPython.core.getipython import get_ipython
    from IPython.nbformat import current

    with io.open(filename) as f:
        notebook = current.read(f, 'json')
    ip = get_ipython()
    for cell in notebook.worksheets[0].cells:
        if cell.cell_type == 'code':
            ip.run_cell(cell.input)


def map_on_cluster(ipc_profile, *args, **kw_args):
    """The same as map, but the first argument is ipc_profile
    :param str|None ipc_profile: the IPython cluster profile to use.
    :return: the result of mapping
    """
    if ipc_profile is None:
        return map(*args, **kw_args)
    else:
        from IPython.parallel import Client

        return Client(profile=ipc_profile).load_balanced_view().map_sync(*args, **kw_args)


def sigmoid_function(x, width):
    """ Sigmoid function is smoothing oh Heaviside function,
    the less width, the closer we are to Heaviside function
    :type x: array-like with floats, arbitrary shape
    :type width: float, if width == 0, this is simply Heaviside function
    """
    assert width >= 0, 'the width should be non-negative'
    if abs(width) > 0.0001:
        return expit(numpy.clip(x / width, -500, 500))
    else:
        return (x > 0) * 1.0


def generate_sample(n_samples, n_features, distance=2.0):
    """Generates some test distribution,
    signal and background distributions are gaussian with same dispersion and different centers,
    all variables are independent (gaussian correlation matrix is identity)"""
    from sklearn.datasets import make_blobs

    centers = numpy.zeros((2, n_features))
    centers[0, :] = - distance / 2
    centers[1, :] = distance / 2

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
    columns = ["column" + str(x) for x in range(n_features)]
    X = pandas.DataFrame(X, columns=columns)
    return X, y


def check_sample_weight(y_true, sample_weight):
    """Checks the weights, returns normalized version """
    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The length of weights is different: not {0}, but {1}".format(len(y_true), len(sample_weight))
        return sample_weight


def reorder_by_first(*arrays):
    """ Applies the same permutation to all passed arrays,
    permutation sorts the first passed array """
    arrays = check_arrays(*arrays)
    order = numpy.argsort(arrays[0])
    return [arr[order] for arr in arrays]


def reorder_by_first_inverse(*arrays):
    """The same as reorder, but the first array is ordered by descending"""
    arrays = check_arrays(*arrays)
    order = numpy.argsort(-arrays[0])
    return [arr[order] for arr in arrays]


def train_test_split(*arrays, **kw_args):
    """Does the same thing as train_test_split, but preserves columns in DataFrames.
    Uses the same parameters: test_size, train_size, random_state
    :type list[numpy.array|pandas.DataFrame] arrays:
    """
    assert len(arrays) > 0, "at least one array should be passed"
    length = len(arrays[0])
    for array in arrays:
        assert len(array) == length, "different size"
    train_indices, test_indices = sklearn.cross_validation.train_test_split(range(length), **kw_args)
    result = []
    for array in arrays:
        if isinstance(array, pandas.DataFrame):
            result.append(array.iloc[train_indices, :])
            result.append(array.iloc[test_indices, :])
        else:
            result.append(array[train_indices])
            result.append(array[test_indices])
    return result


def test_splitting():
    signal_df = pandas.DataFrame(numpy.ones([10, 10]))
    bg_df = pandas.DataFrame(numpy.zeros([10, 10]))

    common_X = pandas.concat([signal_df, bg_df], ignore_index=True)
    common_y = numpy.concatenate([numpy.ones(len(signal_df)), numpy.zeros(len(bg_df))])

    trainX, testX, trainY, testY = train_test_split(common_X, common_y)

    for (index, row), label in zip(trainX.iterrows(), trainY):
        assert numpy.all(label == row), 'wrong data partition'
    for (index, row), label in zip(testX.iterrows(), testY):
        assert numpy.all(label == row), 'wrong data partition'

test_splitting()


def weighted_percentile(array, percentiles, sample_weight=None, array_sorted=False, old_style=False):
    array = numpy.array(array)
    percentiles = numpy.array(percentiles)
    sample_weight = check_sample_weight(array, sample_weight)
    assert numpy.all(percentiles >= 0) and numpy.all(percentiles <= 1), 'Percentiles should be in [0, 1]'

    if not array_sorted:
        array, sample_weight = reorder_by_first(array, sample_weight)

    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(percentiles, weighted_quantiles, array)


def test_percentile(size=100, q_size=20):
    random = RandomState()
    array = random.permutation(size)
    quantiles = random.uniform(size=q_size)
    q_permutation = random.permutation(q_size)
    result1 = weighted_percentile(array, quantiles)[q_permutation]
    result2 = weighted_percentile(array, quantiles[q_permutation])
    result3 = weighted_percentile(array[random.permutation(size)], quantiles[q_permutation])
    assert numpy.all(result1 == result2) and numpy.all(result1 == result3), 'breaks on permutations'

    # checks that order is kept
    quantiles = numpy.linspace(0, 1, size * 3)
    x = weighted_percentile(array, quantiles, sample_weight=random.exponential(size=size))
    assert numpy.all(x == numpy.sort(x)), "doesn't preserve order"

    array = numpy.array([0, 1, 2, 5])
    # comparing with simple percentiles
    for x in random.uniform(size=10):
        assert numpy.abs(numpy.percentile(array, x * 100) - weighted_percentile(array, x, old_style=True)) < 1e-7, \
            "doesn't coincide with numpy.percentile"


test_percentile(100, 20)
test_percentile(20, 100)


def build_normalizer(signal, sample_weight=None):
    """Prepares normalization function for some set of values
    transforms it to uniform distribution from [0, 1]. Example of usage:
        normalizer = build_normalizer(signal)
        hist(normalizer(background))
        # this one should be uniform in [0,1]
        hist(normalizer(signal))
    Parameters:
    :param numpy.array signal: shape = [n_samples] with floats
    :param numpy.array sample_weight: shape = [n_samples], non-negative weights associated to events.
    """
    sample_weight = check_sample_weight(signal, sample_weight)
    assert numpy.all(sample_weight >= 0.), 'sample weight must be non-negative'
    signal, sample_weight = reorder_by_first(signal, sample_weight)
    predictions = numpy.cumsum(sample_weight) / numpy.sum(sample_weight)

    def normalizing_function(data):
        return numpy.interp(data, signal, predictions)
    return normalizing_function


def test_build_normalizer(checks=10):
    predictions = numpy.array(RandomState().normal(size=2000))
    result = build_normalizer(predictions)(predictions)
    assert numpy.all(result[numpy.argsort(predictions)] == sorted(result))
    assert numpy.all(result >= 0) and numpy.all(result <= 1)
    percentiles = [100 * (i + 1.) / (checks + 1.) for i in range(checks)]
    assert numpy.all(abs(numpy.percentile(result, percentiles) - numpy.array(percentiles) / 100.) < 0.01)

    # testing with weights
    predictions = numpy.exp(predictions / 2)
    weighted_normalizer = build_normalizer(predictions, sample_weight=predictions)
    result = weighted_normalizer(predictions)
    assert numpy.all(result[numpy.argsort(predictions)] == sorted(result))
    assert numpy.all(result >= 0) and numpy.all(result <= 1)
    predictions = numpy.sort(predictions)
    result = weighted_normalizer(predictions)
    result2 = numpy.cumsum(predictions) / numpy.sum(predictions)
    assert numpy.all(numpy.abs(result - result2) < 0.005)
    print("normalizer is ok")

test_build_normalizer()


def compute_cut_for_efficiency(efficiency, mask, y_pred, sample_weight=None):
    """ Computes such cut(s), that provide given signal efficiency.
    :type efficiency: float or numpy.array with target efficiencies, shape = [n_effs]
    :type mask: array-like, shape = [n_samples], True for needed classes
    :type y_pred: array-like, shape = [n_samples], predictions or scores (float)
    :type sample_weight: None | array-like, shape = [n_samples]
    :return: float or numpy.array, shape = [n_effs]
    """
    sample_weight = check_sample_weight(mask, sample_weight)
    assert len(mask) == len(y_pred), 'lengths are different'
    efficiency = numpy.array(efficiency)
    is_signal = mask > 0.5
    y_pred, sample_weight = y_pred[is_signal], sample_weight[is_signal]
    return weighted_percentile(y_pred, 1. - efficiency, sample_weight=sample_weight)


def test_compute_cut():
    random = RandomState()
    predictions = random.permutation(100)
    labels = numpy.ones(100)
    for eff in [0.1, 0.5, 0.75, 0.99]:
        cut = compute_cut_for_efficiency(eff, labels, predictions)
        assert numpy.sum(predictions > cut) / len(predictions) == eff, 'the cut was set wrongly'

    weights = numpy.array(random.exponential(size=100))
    for eff in random.uniform(size=100):
        cut = compute_cut_for_efficiency(eff, labels, predictions, sample_weight=weights)
        lower = numpy.sum(weights[predictions > cut + 1]) / numpy.sum(weights)
        upper = numpy.sum(weights[predictions > cut - 1]) / numpy.sum(weights)
        assert lower < eff < upper

test_compute_cut()


def compute_bdt_cut(target_efficiency, y_true, y_pred, sample_weight=None):
    """Computes cut which gives fixed efficiency.
    :type target_efficiency: float from 0 to 1 or numpy.array with floats in [0,1]
    :type y_true: numpy.array, of zeros and ones, shape = [n_samples]
    :type y_pred: numpy.array, prediction probabilities returned by classifier, shape = [n_samples, 2]
    """
    assert len(y_true) == len(y_pred), "different size"
    signal_proba = y_pred[y_true > 0.5]
    percentiles = 1. - target_efficiency
    sig_weights = None if sample_weight is None else sample_weight[y_true > 0.5]
    return weighted_percentile(signal_proba, percentiles, sample_weight=sig_weights)


#TODO replace this function with one from metrics
def compute_groups_efficiencies(global_cut, knn_indices, answers, y_pred,
                                sample_weight=None, smoothing_width=0.0):
    """Fast implementation in numpy"""
    assert len(answers) == len(y_pred), 'different size'
    sample_weight = check_sample_weight(answers, sample_weight)
    predictions = sigmoid_function(y_pred - global_cut, smoothing_width)
    groups_predictions = numpy.take(predictions, knn_indices)
    groups_weights = numpy.take(sample_weight, knn_indices)
    return numpy.average(groups_predictions, weights=groups_weights, axis=1)

#region Knn-related things

def computeSignalKnnIndices(uniform_variables, dataframe, is_signal, n_neighbors=50):
    """For each event returns the knn closest signal(!) events. No matter of what class the event is.
    :type uniform_variables: list of names of variables, using which we want to compute the distance
    :type dataframe: pandas.DataFrame, should contain these variables
    :type is_signal: numpy.array, shape = [n_samples] with booleans
    :rtype numpy.array, shape [len(dataframe), knn], each row contains indices of closest signal events
    """
    assert len(dataframe) == len(is_signal), "Different lengths"
    signal_indices = numpy.where(is_signal)[0]
    for variable in uniform_variables:
        assert variable in dataframe.columns, "Dataframe is missing %s column" % variable
    uniforming_features_of_signal = numpy.array(dataframe.ix[is_signal, uniform_variables])
    neighbours = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(uniforming_features_of_signal)
    _, knn_signal_indices = neighbours.kneighbors(dataframe[uniform_variables])
    return numpy.take(signal_indices, knn_signal_indices)


def computeKnnIndicesOfSameClass(uniform_variables, X, y, n_neighbours=50):
    """Works as previous function, but returns the neighbours of the same class as element
    :param list[str] uniform_variables: the names of columns"""
    assert len(X) == len(y), "different size"
    result = numpy.zeros([len(X), n_neighbours], dtype=numpy.int)
    for label in set(y):
        is_signal = y == label
        label_knn = computeSignalKnnIndices(uniform_variables, X, is_signal, n_neighbours)
        result[is_signal, :] = label_knn[is_signal, :]
    return result


def test_compute_knn_indices(n_events=100):
    X, y = generate_sample(n_events, 10, distance=.5)
    is_signal = y > 0.5
    signal_indices = numpy.where(is_signal)[0]
    uniform_columns = X.columns[:1]
    knn_indices = computeSignalKnnIndices(uniform_columns, X, is_signal, 10)
    distances = pairwise_distances(X[uniform_columns])
    for i, neighbours in enumerate(knn_indices):
        assert numpy.all(is_signal[neighbours]), "returned indices are not signal"
        not_neighbours = [x for x in signal_indices if not x in neighbours]
        min_dist = numpy.min(distances[i, not_neighbours])
        max_dist = numpy.max(distances[i, neighbours])
        assert min_dist >= max_dist, "distances are set wrongly!"

    knn_all_indices = computeKnnIndicesOfSameClass(uniform_columns, X, is_signal, 10)
    for i, neighbours in enumerate(knn_all_indices):
        assert numpy.all(is_signal[neighbours] == is_signal[i]), "returned indices are not signal/bg"

    print("computeSignalKnnIndices is ok")

test_compute_knn_indices()

#endregion


def smear_dataset(testX, smeared_variables=None, smearing_factor=0.1):
    """For the selected features 'smears' them in dataset,
    pay attention, that only float feature can be smeared by now.
    If smeared variables is None, all the features are smeared"""
    assert isinstance(testX, pandas.DataFrame), "the passed object is not of type pandas.DataFrame"
    testX = pandas.DataFrame.copy(testX)
    if smeared_variables is None:
        smeared_variables = testX.columns
    for var in smeared_variables:
        assert var in testX.columns, "The variable %s was not found in dataframe"
    result = pandas.DataFrame.copy(testX)
    for var in smeared_variables:
        sigma = math.sqrt(numpy.var(result[var]))
        result[var] += RandomState().normal(0, smearing_factor * sigma, size=len(result))
    return result


def memory_usage():
    """Memory usage of the current process in bytes. Created for notebooks.
    This will only work on systems with a /proc file system (like Linux)."""
    result = {'peak': 0, 'rss': 0}
    with open('/proc/self/status') as status:
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = "{:,} kB".format(int(parts[1]))
    return result


def subdict(start_dict, keys=None):
    """Returns the ordered dictionary from initial one
     which contains only listed keys
    """
    if keys is None:
        return OrderedDict(start_dict)
    result = OrderedDict()
    for key in keys:
        result[key] = start_dict[key]
    return result


def indices_of_values(array):
    indices = numpy.argsort(array)
    sorted_array = array[indices]
    diff = numpy.nonzero(numpy.ediff1d(sorted_array))[0]
    limits = [0] + list(diff + 1) + [len(array)]
    for i in range(len(limits) - 1):
        yield sorted_array[limits[i]], indices[limits[i]: limits[i+1]]

