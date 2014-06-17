# About

# This file contains some helpful functions and classes
# which are often used (by other modules)


from __future__ import print_function
from __future__ import division

import math
import pandas
import numpy
from numpy.random.mtrand import RandomState
import io
import sklearn.cross_validation
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.metrics import auc

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


def check_sample_weight(y_true, sample_weight):
    """Checks the weights, returns normalized version """
    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The lengths are different: {0} and {1}".format(len(y_true), len(sample_weight))
        return sample_weight


def my_train_test_split(*arrays, **kw_args):
    """Does the same thing as train_test_split, but preserves columns in DataFrames.
    Uses the same parameters: test_size, train_size, random_state
    """
    assert len(arrays) > 0, "at least one array should be given"
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


# TODO delete
# def split_on_test_and_train(signal_df, bg_df, **kw_args):
#     """Useful way to split data when is given as two DataFrames """
#     assert set(signal_df.columns) == set(bg_df.columns), 'Different set  of columns'
#     common_df = pandas.concat([signal_df, bg_df], ignore_index=True)
#     answers = numpy.concatenate([numpy.ones(len(signal_df)), numpy.zeros(len(bg_df))])
#     assert len(answers) == len(common_df), 'Something gone wrong during splitting'
#     return my_train_test_split(common_df, answers, **kw_args)

def weighted_percentile(array, percentiles, sample_weight=None, array_sorted=False, old_style=False):
    sample_weight = check_sample_weight(array, sample_weight)
    if not array_sorted:
        order = numpy.argsort(array)
        array, sample_weight = array[order], sample_weight[order]
    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    percentiles = numpy.array(percentiles)
    assert numpy.all(percentiles >= 0) and numpy.all(percentiles <= 1), 'Percentiles should be in [0, 1]'
    upper = numpy.clip(numpy.searchsorted(weighted_quantiles, percentiles), 1, len(array) - 1)
    lower = upper - 1
    lower_input = numpy.take(weighted_quantiles, lower)
    upper_input = numpy.take(weighted_quantiles, upper)
    lower_output = numpy.take(array, lower)
    upper_output = numpy.take(array, upper)
    t = (percentiles - lower_input) / (upper_input - lower_input + 1e-20)
    return upper_output * t + lower_output * (1 - t)


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


def test_splitting():
    signal_df = pandas.DataFrame(numpy.ones([10, 10]))
    bg_df = pandas.DataFrame(numpy.zeros([10, 10]))

    common_X = pandas.concat([signal_df, bg_df], ignore_index=True)
    common_y = numpy.concatenate([numpy.ones(len(signal_df)), numpy.zeros(len(bg_df))])

    trainX, testX, trainY, testY = my_train_test_split(common_X, common_y)

    for (index, row), label in zip(trainX.iterrows(), trainY):
        assert numpy.all(label == row), 'wrong data partition'
    for (index, row), label in zip(testX.iterrows(), testY):
        assert numpy.all(label == row), 'wrong data partition'

test_splitting()


class Binner:
    def __init__(self, values, n_bins, sample_weight=None):
        """Binner is a class that helps to split the values into several bins.
        Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
        and thus we are computing limits (boundaries of bins)."""
        percentiles = [i / n_bins for i in range(1, n_bins)]
        self.limits = weighted_percentile(values, percentiles, sample_weight=sample_weight)

    def get_bins(self, values):
        return numpy.searchsorted(self.limits, values)

    def set_limits(self, limits):
        self.limits = limits

    @property
    def n_bins(self):
        return len(self.limits) + 1

    def split_into_bins(self, *arrays):
        """Splits the data of parallel arrays into bins, the first array is binning variable"""
        values = arrays[0]
        numpy_arrays = []
        for i, array in enumerate(arrays):
            assert len(array) == len(values), "passed arrays have different lengths"
            numpy_arrays.append(numpy.array(array))
        bins = self.get_bins(values)
        result = []
        for bin in range(len(self.limits) + 1):
            indices = bins == bin
            result.append([array[indices] for array in numpy_arrays])
        return result


def test_binner():
    """This function tests binner class"""
    random = RandomState()
    binner = Binner(random.permutation(30), 3)
    assert numpy.all(binner.limits > [9, 19]), 'failed on the limits'
    assert numpy.all(binner.limits < [10, 20]), 'failed on the limits'
    bins = binner.get_bins([-1000, 1000, 0, 10, 20, 9.0, 10.1, 19.0, 20.1])
    assert numpy.all(bins == [0, 2, 0, 1, 2, 0, 1, 1, 2]), 'wrong binning'

    binner = Binner(random.permutation(20), 5)
    p = random.permutation(40)
    # checking whether binner preserves correspondence
    list1 = list(binner.split_into_bins(numpy.array(range(-10, 30))[p], numpy.array(range(0, 40))[p]))
    for a, b in list1:
        for x, y in zip(a, b):
            assert x + 10 == y, 'transpositions are wrong after binning'

    binner = Binner(random.permutation(30), 3)
    result2 = list(binner.split_into_bins(range(10, 20)))
    answer2 = [[], range(10, 20), []]

    for a, b in zip(result2, answer2):
        for x, y in zip(a[0], b):
            assert x == y, 'binning is wrong'

    result3 = list(binner.split_into_bins(random.permutation(45)))
    answer3 = list(binner.split_into_bins(range(45)))
    for x, y in zip(result3, answer3):
        assert set(x[0]) == set(y[0]), "binner doesn't work well with permutations"

    print('binner is ok')

test_binner()


def build_normalizer(signal, sample_weight=None):
    """Prepares normalization function for some set of values
    transforms it to uniform distribution from [0, 1]. Example of usage:
        normalizer = build_normalizer(signal)
        hist(normalizer(background))
        # this one should be uniform in [0,1]
        hist(normalizer(signal))
    Parameters:
        signal: array-like, shape = [n_samples]
        sample_weight: array-like, shape = [n_samples], the weights associated to events.
    """
    sample_weight = check_sample_weight(signal, sample_weight)
    assert numpy.all(sample_weight >= 0.), 'sample weight must be non-negative'
    order = numpy.argsort(signal)
    signal, sample_weight = signal[order], sample_weight[order]
    predictions = numpy.cumsum(sample_weight) / numpy.sum(sample_weight)

    def normalizing_function(data):
        data = numpy.clip(data, signal[0], signal[-1])
        upper = numpy.searchsorted(signal, data)
        upper = numpy.clip(upper, 1, len(signal))
        lower = upper - 1
        lower_output = numpy.take(predictions, lower)
        upper_output = numpy.take(predictions, upper)
        lower_input = numpy.take(signal, lower)
        upper_input = numpy.take(signal, upper)
        t = (data - lower_input) / (upper_input - lower_input + 1e-10)
        return t * upper_output + (1.-t) * lower_output
    return normalizing_function


def test_build_normalizer(checks=10):
    predictions = RandomState().normal(size=2000)
    result = build_normalizer(predictions)(predictions)
    assert numpy.all(result[numpy.argsort(predictions)] == sorted(result))
    assert numpy.all(result >= 0)
    assert numpy.all(result <= 1)
    percentiles = [100 * (i + 1.) / (checks + 1.) for i in range(checks)]
    assert numpy.all(abs(numpy.percentile(result, percentiles) - numpy.array(percentiles) / 100.) < 0.01)

    # testing with weights
    predictions = numpy.exp(predictions)
    weighted_normalizer = build_normalizer(predictions, sample_weight=predictions)
    result = weighted_normalizer(predictions)
    assert numpy.all(result[numpy.argsort(predictions)] == sorted(result))
    assert numpy.all(result >= 0)
    assert numpy.all(result <= 1)
    predictions = numpy.sort(predictions)
    result = weighted_normalizer(predictions)
    result2 = numpy.cumsum(predictions) / numpy.sum(predictions)
    assert numpy.all(numpy.abs(result-result2) < 0.005)
    print("normalizer is ok")


test_build_normalizer()

# Functions primarily for uBoost

def compute_cut_for_efficiency(efficiency, y_true, y_pred, sample_weight=None):
    """ Computes such cut(s), that provide given signal efficiency.
    :param efficiency: float or numpy.array with target efficiencies, shape = [n_effs]
    :param y_true: array-like, shape = [n_samples], labels (zeros ans ones)
    :param y_pred: array-like, shape = [n_samples], predictions or scores (float)
    :param sample_weight: None or array-like, shape = [n_samples]
    :return: float or numpy.array, shape = [n_effs]
    """
    sample_weight = check_sample_weight(y_true, sample_weight)
    assert len(y_true) == len(y_pred), 'lengths are different'

    is_signal = y_true > 0.5
    y_pred, sample_weight = y_pred[is_signal], sample_weight[is_signal]
    return weighted_percentile(y_pred, 1. - efficiency, sample_weight=sample_weight)


def test_compute_cut():
    random = RandomState()
    predictions = random.permutation(100)
    labels = numpy.ones(100)
    for eff in [0.1, 0.5, 0.75, 0.99]:
        cut = compute_cut_for_efficiency(eff, labels, predictions)
        assert numpy.sum(predictions > cut) / len(predictions) == eff, 'the cut was set wrongly'

    weights = random.exponential(size=100)
    for eff in random.uniform(size=100):
        cut = compute_cut_for_efficiency(eff, labels, predictions, sample_weight=weights)
        lower = numpy.sum(weights[predictions > cut + 1]) / numpy.sum(weights)
        upper = numpy.sum(weights[predictions > cut - 1]) / numpy.sum(weights)
        assert lower < eff < upper

test_compute_cut()


def compute_bdt_cut(target_efficiency, y_true, y_pred, sample_weight=None):
    """Computes cut which gives fixed efficiency.
    Parameters:
        target_efficiency: float from 0 to 1 or numpy.array with floats in [0,1]
        y_true: an array of zeros and ones, shape = [n_samples]
        y_pred: prediction probabilities returned by classifier, shape = [n_samples, 2]
    """
    assert len(y_true) == len(y_pred), "different size"
    signal_probas = y_pred[y_true > 0.5, 1]
    percentiles = 100 - target_efficiency * 100
    if isinstance(percentiles, numpy.ndarray):
        percentiles = list(percentiles)
    return numpy.percentile(signal_probas, percentiles)


def compute_groups_efficiencies(global_cut, knn_indices, answers, prediction_proba,
                                sample_weight=None, smoothing_width=0.0):
    """Fast implementation in numpy"""
    assert len(answers) == len(prediction_proba), 'different size'
    sample_weight = check_sample_weight(answers, sample_weight)
    predictions = sigmoid_function(prediction_proba[:, 1] - global_cut, smoothing_width)
    groups_predictions = numpy.take(predictions, knn_indices)
    groups_weights = numpy.take(sample_weight, knn_indices)
    # TODO test this new implementation
    return numpy.average(groups_predictions, weights=groups_weights, axis=1)
    # neigh_predictions.mean(axis=1)


def sigmoid_function(x, width):
    """ Sigmoid function is smoothing oh Heaviside function, the lesser width,
        the closer we are to Heaviside function
    Parameters:
        x: array-like with floats, arbitrary shape
        width: float, if width == 0, this is simply Heaviside function
    """
    assert width >= 0, 'the width should be non-negative'
    if abs(width) > 0.0001:
        return 1.0 / (1.0 + numpy.exp(-x / width))
    else:
        return (x > 0) * 1.0


def generateSample(n_samples, n_features, distance=2.0):
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


def computeSignalKnnIndices(uniform_variables, dataframe, is_signal, n_neighbors=50):
    """For each event returns the knn closest signal(!) events. No matter of what class the event is.
    Parameters:
        uniform_variables: list of names of variables, using which we want to compute the distance
        dataframe: should contain these variables
        is_signal: is boolean numpy.array, shape = [n_samples]
    Returns:
        ndarray of shape [len(dataframe), knn],
        each row contains indices of closest signal events
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
    """Works as previous function, but returns the neighbours of the same class as element"""
    assert len(X) == len(y), "different size"
    result = numpy.zeros([len(X), n_neighbours], dtype=numpy.int)
    for label in set(y):
        is_signal = y == label
        label_knn = computeSignalKnnIndices(uniform_variables, X, is_signal, n_neighbours)
        result[is_signal, :] = label_knn[is_signal, :]
    return result


def test_compute_knn_indices(n_events=100):
    X, y = generateSample(n_events, 10, distance=.5)
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


def smear_dataset(testX, smeared_variables=None, smearing_factor=0.1):
    """For the selected features 'smears' them in dataset,
    pay attention, that only float feature can be smeared by now.
    If smeared variables is None, all the features are smeared"""
    assert isinstance(testX, pandas.DataFrame), "the passed object is not of type pandas.DataFrame"
    if smeared_variables is None:
        smeared_variables = testX.columns
    for var in smeared_variables:
        assert var in testX.columns, "The variable %s was not found in dataframe"
    result = pandas.DataFrame.copy(testX)
    for var in smeared_variables:
        sigma = math.sqrt(numpy.var(result[var]))
        result[var] += RandomState().normal(0, smearing_factor * sigma, len(result))
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


def roc_curve(y_true, y_score, sample_weight=None):
    """ The same as sklearn.metrics.roc_curve, but this one supports weights    """
    if sample_weight is None:
        sample_weight = numpy.ones(len(y_score))
    assert len(y_true) == len(y_score) == len(sample_weight), 'the lengths are different'
    assert set(y_true) == {0, 1}, "the labels should be 0 and 1, labels are " + str(set(y_true))
    order = numpy.argsort(y_score)[::-1]
    thresholds = y_score[order]
    y_true = y_true[order]
    sample_weight = sample_weight[order]
    tpr = numpy.insert(numpy.cumsum(sample_weight * y_true), 0, 0.)
    tpr /= tpr[-1]
    fpr = numpy.insert(numpy.cumsum(sample_weight * (1 - y_true)), 0, 0.)
    fpr /= fpr[-1]
    thresholds = numpy.insert(thresholds, 0, thresholds[0] + 1.)
    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score, sample_weight=None):
    """The same as sklearn.metrics.roc_auc_score, but supports weights """
    if len(numpy.unique(y_true)) != 2:
        raise ValueError("AUC is defined for binary classification only")
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    return auc(fpr, tpr, reorder=True)


def test_roc_curve(size=100):
    import sklearn.metrics
    y = (numpy.random.random(size) > 0.5) * 1
    pred = numpy.random.random(size)
    fpr1, tpr1, thr1 = sklearn.metrics.roc_curve(y, pred)
    fpr2, tpr2, thr2 = roc_curve(y, pred)
    assert numpy.all(fpr1 == fpr2)
    assert numpy.all(tpr1 == tpr2)
    assert numpy.all(thr1 == thr2)

# test_roc_curve(10)


def export_root_to_csv(filename, branches=None):
    """From selected file exports all the trees in separate files, exports all the branches,
    requires rootpy and root_numpy modules"""
    import root_numpy
    import os
    trees = root_numpy.list_trees(filename)
    print("The following branches are found:\n %s" % trees)
    result = []
    for tree_name in trees:
        x = root_numpy.root2array(filename, treename=tree_name, branches=branches)
        new_file_name = os.path.splitext(filename)[0] + '_' + tree_name + '.csv'
        pandas.DataFrame(x).to_csv(new_file_name)
        result.append(new_file_name)
    print("Successfully converted")
    return result


