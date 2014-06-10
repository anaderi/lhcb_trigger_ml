# About
# This file contains some helpful functions and classes which are often used.

import math
import pandas
import numpy
import io
from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.unsupervised import NearestNeighbors

__author__ = "Alex Rogozhnikov"

def execute_notebook(filename):
    """Allows one to execute cell-by-cell some notebook provided its name"""
    from IPython.core.getipython import get_ipython
    from IPython.nbformat import current

    with io.open(filename) as f:
        nb = current.read(f, 'json')
    ip = get_ipython()
    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.input)


def my_train_test_split(*arrays, **kw_args):
    """Does the same thing as train_test_split, but preserves columns in DataFrames
    Uses the same parameters: test_size, train_size, random_state
    """
    assert len(arrays) > 0, "at least one array should be given"
    length = len(arrays[0])
    for array in arrays:
        assert len(array) == length, "different size"
    train_indices, test_indices = train_test_split(range(length), **kw_args)
    result = []
    for array in arrays:
        if isinstance(array, pandas.DataFrame):
            result.append(array.irow(train_indices))
            result.append(array.irow(test_indices))
        else:
            result.append(array[train_indices])
            result.append(array[test_indices])
    return result


def split_on_test_and_train(signal_df, bg_df, **kw_args):
    assert set(signal_df.columns) == set(bg_df.columns), 'Different set  of columns'
    common_df = pandas.concat([signal_df, bg_df], ignore_index=True)
    answers = numpy.concatenate([numpy.ones(len(signal_df)), numpy.zeros(len(bg_df))])
    assert len(answers) == len(common_df), 'Something gone wrong during splitting'
    return my_train_test_split(common_df, answers, **kw_args)


def test_splitting():
    signal_df = pandas.DataFrame(numpy.ones([10, 10]))
    bg_df = pandas.DataFrame(numpy.zeros([10, 10]))

    trainX, testX, trainY, testY = split_on_test_and_train(signal_df, bg_df, train_size=0.5)
    for (index, row), pred in zip(trainX.iterrows(), trainY):
        assert numpy.all(pred == row), 'wrongly data partition'
    for (index, row), pred in zip(testX.iterrows(), testY):
        assert numpy.all(pred == row), 'wrongly data partition'

test_splitting()


class Binner:
    def __init__(self, values, bins_number):
        """Binner is a class that helps to split the values into several bins.
        Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
        and thus we are computing limits (boundaries of bins)."""
        percentiles = [i * 100.0 / bins_number for i in range(1, bins_number)]
        self.limits = numpy.percentile(values, percentiles)

    def get_bins(self, values):
        return numpy.searchsorted(self.limits, values)

    def set_limits(self, limits):
        self.limits = limits

    def bins_number(self):
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
    binner = Binner(numpy.random.permutation(30), 3)
    assert numpy.all(binner.limits > [9, 19]), 'failed on the limits'
    assert numpy.all(binner.limits < [10, 20]), 'failed on the limits'
    bins = binner.get_bins([-1000, 1000, 0, 10, 20, 9.0, 10.1, 19.0, 20.1])
    assert numpy.all(bins == [0, 2, 0, 1, 2, 0, 1, 1, 2]), 'wrong binning'

    binner = Binner(numpy.random.permutation(20), 5)
    p = numpy.random.permutation(40)
    # checking whether binner preserves correspondence
    list1 = list(binner.split_into_bins(numpy.array(range(-10, 30))[p], numpy.array(range(0, 40))[p]))
    for a, b in list1:
        for x, y in zip(a, b):
            assert x + 10 == y, 'transpositions are wrong after binning'
    binner = Binner(numpy.random.permutation(30), 3)
    res2 = list(binner.split_into_bins(range(10, 20)))
    ans2 = [[], range(10, 20), []]

    for a, b in zip(res2, ans2):
        for x, y in zip(a[0], b):
            assert x == y, 'binning is wrong'

    res3 = list(binner.split_into_bins(numpy.random.permutation(45)))
    ans3 = list(binner.split_into_bins(range(45)))
    for x, y in zip(res3, ans3):
        assert set(x[0]) == set(y[0]), "binner doesn't work well with permutations"

    print('binner is ok')

test_binner()


def build_normalizer(signal, steps=None):
    """Prepares normalization function for some set of values
    transforms it to uniform distribution from [0, 1]. Example of usage:
        normalizer = build_normalizer(signal)
        hist(normalizer(background))
        hist(normalizer(signal))
    Parameters:
        signal: array-like, shape = [n_samples]
        steps: number of steps in interpolating function
    """
    if steps is None:
        steps = len(signal)
    effs = numpy.array([i / float(steps) for i in range(steps+1)])
    effs_by_100 = [100 * x for x in effs]
    percs = numpy.percentile(signal, effs_by_100)

    def normalizing_function(data):
        data = numpy.clip(data, percs[0], percs[-1])
        upper = numpy.searchsorted(percs, data)
        upper = numpy.clip(upper, 1, steps)
        lower = upper - 1
        lower_output = numpy.take(effs, lower)
        upper_output = numpy.take(effs, upper)
        lower_input = numpy.take(percs, lower)
        upper_input = numpy.take(percs, upper)
        t = (data - lower_input) / (upper_input - lower_input + 1e-10)
        return t * upper_output + (1-t) * lower_output
    return normalizing_function


def test_build_normalizer(checks=10):
    predictions = numpy.random.normal(size=2000)
    result = build_normalizer(predictions, steps=200)(predictions)
    assert numpy.all(result[numpy.argsort(predictions)] == sorted(result))
    assert numpy.all(result >= 0)
    assert numpy.all(result <= 1)
    percentiles = [100 * (i + 1.) / (checks + 1.) for i in range(checks)]
    assert numpy.all(abs(numpy.percentile(result, percentiles) - numpy.array(percentiles) / 100.) < 0.01)
    print "normalizer is ok"


test_build_normalizer()



# Functions primarily for uBoost


def computeBDTCut(target_efficiency, answers, prediction_probas):
    """Computes cut which gives targetEfficiency
    * targetEfficiency from 0 to 1
    * answers is an array of zeros and ones
    * prediction_probas is prediction probabilities returned by classifier, shape = [n_samples, 2]
    """
    assert len(answers) == len(prediction_probas), "different size"

    signal_probas = prediction_probas[answers > 0.5, 1]
    percentiles = 100 - target_efficiency * 100
    if isinstance(percentiles, numpy.ndarray):
        percentiles = list(percentiles)
    return numpy.percentile(signal_probas, percentiles)


def computeLocalEfficiencies(globalCut, knnIndices, answers, prediction_proba, smoothing_width=0.0):
    """Fast implementation in numpy"""
    assert len(answers) == len(prediction_proba), 'different size'
    predictions = sigmoidFunction(prediction_proba[:, 1] - globalCut, smoothing_width)
    neigh_predictions = numpy.take(predictions, knnIndices)
    return neigh_predictions.mean(axis=1)


def sigmoidFunction(x, width):
    """Sigmoid function is smoothing oh Heaviside function, the lesser width,
       the closer we are to Heaviside function
    Parameters:
        * x - array-like with floats, arbitrary shape
        * width is float, if width == 0, this is simply Heaviside function
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
        * uniform_variables is list of names of variables, using which we want to compute the distance
        * dataframe should contain these variables
        * is_signal is boolean numpy.array, shape = [n_samples]
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


def testComputeSignalKnnIndices(n_events=100):
    X, y = generateSample(n_events, 10, distance=.5)
    is_signal = y > 0.5
    signal_indices = numpy.where(is_signal)[0]
    unif_columns = X.columns[:1]
    knn_indices = computeSignalKnnIndices(unif_columns, X, is_signal, 10)
    distances = pairwise_distances(X[unif_columns])
    for i, neighbours in enumerate(knn_indices):
        assert numpy.all(is_signal[neighbours]), "returned indices are not signal"
        not_neighbours = [x for x in signal_indices if not x in neighbours]
        min_dist = numpy.min(distances[i, not_neighbours])
        max_dist = numpy.max(distances[i, neighbours])
        assert min_dist >= max_dist, "distances are set wrongly!"

    knn_all_indices = computeKnnIndicesOfSameClass(unif_columns, X, is_signal, 10)
    for i, neighbours in enumerate(knn_all_indices):
        assert numpy.all(is_signal[neighbours] == is_signal[i]), "returned indices are not signal/bg"

    print("computeSignalKnnIndices is ok")

testComputeSignalKnnIndices()


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
        result[var] += numpy.random.normal(0, smearing_factor * sigma, len(result))
    return result


def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result


def export_root_to_csv(filename):
    """From selected file exports all the trees in separate files, exports all the branches,
    requires rootpy and root_numpy modules"""
    import root_numpy
    import os
    trees = root_numpy.list_trees(filename)
    print("The following branches are found:\n %s" % trees)
    result = []
    for tree_name in trees:
        x = root_numpy.root2array(filename, treename=tree_name)
        new_file_name = os.path.splitext(filename)[0] + '_' + tree_name + '.csv'
        pandas.DataFrame(x).to_csv(new_file_name)
        result.append(new_file_name)
    print("Successfully converted")
    return result

