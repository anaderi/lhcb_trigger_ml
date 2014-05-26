# #About
# 
# This file contains some helpful functions and classes which are often used.
# This file is ROOT-independent

import math
from IPython.nbformat import current
from numpy.random.mtrand import normal

import pandas
import numpy
import io
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.unsupervised import NearestNeighbors


Precision = precision_score
Recall = recall_score
F1Score = f1_score


def execute_notebook(fileName):
    from IPython.core.getipython import get_ipython
    with io.open(fileName) as f:
        nb = current.read(f, 'json')
    ip = get_ipython()
    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code': continue
        ip.run_cell(cell.input)


def addIsSignalColumn(dataFrame, is_signal):
    """Is signal can be either 1 or 0 or array """
    dataFrame["IsSignal"] = is_signal


def getProbabilitiesOfSignal(classifier, test_data):
    """predictProba returns the 2d array, 
        [:,0] - probabilities of 0 class (bg)
        [:,1] - probabilities of 1 class (signal)
    """
    return classifier.predict_proba(test_data)[:, 1]


def shuffleDataSet(dataFrame, answers):
    """Shuffles the rows in the dataFrame and answersColumn simultaneously
    Pay attention that dataFrame is changed in the procedure,
    this may cause some side-effects, so if you need original dataFrame, use clone() before
    """
    # TODO sklearn shuffle
    length = len(dataFrame)
    if len(answers) != length:
        raise ValueError("Different lengths")
    permutation = numpy.random.permutation(length)
    # don't use inplace and copy without real need
    # these operations just economy the time
    dataFrame.set_index([range(length)], inplace=True)
    dataFrame = dataFrame.reindex(permutation, copy=False)
    # restoring index
    dataFrame.set_index([range(length)], inplace=True)
    return dataFrame, answers[permutation]


def splitOnTestAndTrain(signalDataFrame, bgDataFrame,
                        signalTrainPart=0.5, bgTrainPart=0.5):
    signalTrainInd, signalTestInd = train_test_split(range(len(signalDataFrame)), train_size=signalTrainPart)
    bgTrainInd, bgTestInd = train_test_split(range(len(bgDataFrame)), train_size=bgTrainPart)

    signalTrain = signalDataFrame.irow(signalTrainInd)
    signalAnsTrain = numpy.ones_like(signalTrainInd)
    signalTest = signalDataFrame.irow(signalTestInd)
    signalAnsTest = numpy.ones_like(signalTestInd)

    bgTrain = bgDataFrame.irow(bgTrainInd)
    bgAnsTrain = numpy.zeros_like(bgTrainInd)
    bgTest = bgDataFrame.irow(bgTestInd)
    bgAnsTest = numpy.zeros_like(bgTestInd)

    # Concatenating in single dataframe
    train = pandas.concat([signalTrain, bgTrain], join='inner', ignore_index=True)
    test = pandas.concat([signalTest, bgTest], join='inner', ignore_index=True)
    trainAns = numpy.concatenate((signalAnsTrain, bgAnsTrain))
    testAns = numpy.concatenate((signalAnsTest, bgAnsTest))

    # Shuffling. It isn't mandatory, just in case classifier would somehow take order into account
    # it is better to shuffle data
    train, trainAns = shuffleDataSet(train, trainAns)
    test, testAns = shuffleDataSet(test, testAns)

    return train, trainAns, test, testAns


def my_train_test_split(*arrays, **kw_args):
    """
    Does the same thin as train_test_split, but preserves columns in DataFrames
    Uses the same parameters: test_size. train_size, random_state
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


df = pandas.DataFrame(numpy.random.rand(100, 10))
a, b = my_train_test_split(df)




class Binner:
    def __init__(self, values, bins_number):
        """Binner is a class that helps to split the values into several bins.
        Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
        and thus we are computing limits (boundaries of bins)."""
        percentiles = [i * 100.0 / bins_number for i in range(1, bins_number)]
        self.limits = numpy.percentile(values, percentiles)

    def get_bins(self, values):
        return numpy.searchsorted(self.limits, values)

    def get_bins_dumb(self, values):
        """This is the sane as previous function, but a bit slower and naive"""
        result = numpy.zeros(len(values))
        for limit in self.limits:
            result += values > limit
        return result

    def set_limits(self, limits):
        self.limits = limits

    def bins_number(self):
        return len(self.limits) + 1

    def split_into_bins(self, *arrays):
        """
        Splits the data of parallel arrays into bins, the first array is binning variable
        """
        values = arrays[0]
        for array in arrays:
            assert len(array) == len(values), "passed arrays have different length"
        bins = self.get_bins(values)
        result = []
        for bin in range(len(self.limits) + 1):
            indices = bins == bin
            result.append([numpy.array(array)[indices] for array in arrays])
        return result


def testBinner():
    """
    This function tests binner class
    """
    binner = Binner(numpy.random.permutation(30), 3)
    assert numpy.all(binner.limits > [9, 19]), 'failed on the limits'
    assert numpy.all(binner.limits < [10, 20]), 'failed on the limits'
    bins = binner.get_bins([-1000, 1000, 0, 10, 20, 9.0, 10.1, 19.0, 20.1])
    assert numpy.all(bins == [0, 2, 0, 1, 2, 0, 1, 1, 2]), 'wrong binning'

    binner = Binner(numpy.random.permutation(100), 7)
    p = numpy.random.permutation(100)
    assert numpy.all(binner.get_bins(p) == binner.get_bins_dumb(p)), "getBins() function is wrong"

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


testBinner()


def slidingEfficiencyArray(answers, prediction_proba):
    """Returns two arrays,
    if threshold == second array[i]
    then efficiency == first array[i] (approximately)
    """
    assert len(answers) == len(prediction_proba), "different size of arrays"
    signal_probabilities = prediction_proba[:, 1]
    indices = numpy.argsort(signal_probabilities)
    ans = answers[indices]
    probs = signal_probabilities[indices]

    IsSig = numpy.sum(ans) + 1e-6
    IsSigAsSig = numpy.sum(ans) - numpy.cumsum(ans)

    return IsSigAsSig / IsSig, probs


def computeEfficiencyAtCuts(answers, predictions_proba, cuts):
    """A bit unprecise, for each cut this function computes the efficiency"""
    efficiencies, thresholds = slidingEfficiencyArray(answers, predictions_proba)
    indices = numpy.searchsorted(thresholds, cuts)
    indices = numpy.clip(indices, 0, len(thresholds) - 1)
    return efficiencies[indices]


def interpolate(y_array, x):
    """Assuming we have a function, that has at point i value y_array[i]
    Then it returns piecewise-linear interpolation of it at point x"""
    if x >= len(y_array) - 1.0001:
        return y_array[-1]
    if x <= 0:
        return y_array[0]
    n = int(math.floor(x))
    t = x - n
    return y_array[n] * (1.0 - t) + y_array[n + 1] * t


def massive_interpolate(y_array, x):
    """The same as interpolate, but x is array now
    returns array of the same length as x
    """
    y_array = numpy.array(y_array)
    x = numpy.clip(x, 0.0001, len(y_array) - 1.0001)
    n = numpy.floor(x).astype(numpy.int)
    t = x - n
    return y_array.take(n) * (1.0 - t) + y_array.take(n + 1) * t


def correctionFunction(answers, predict_proba, steps=10):
    cuts = [i * 1.0 / steps for i in range(0, steps + 1)]
    values = numpy.array([recall_score(answers, predict_proba[:, 1] > cut) for cut in cuts])
    return lambda x: interpolate(values, x * (len(values) - 1))


def massiveCorrectionFunction(answers, predict_proba, steps=10):
    cuts = [i * 1.0 / steps for i in range(0, steps + 1)]
    values = computeEfficiencyAtCuts(answers, predict_proba, cuts)
    return lambda x: massive_interpolate(values, x * (len(values) - 1))


def testCorrectionFunctionIteration():
    import pylab
    l = 100
    answers1 = numpy.zeros(l)
    answers2 = numpy.zeros(l) + 1
    answers = numpy.concatenate((answers1, answers2))
    probs1 = numpy.random.rand(l) * numpy.random.rand(l)
    probs2 = - numpy.random.rand(l) * numpy.random.rand(l) + 1.0
    probs = numpy.zeros((len(probs1) + len(probs2), 2))
    probs[:, 1] = numpy.concatenate((probs1, probs2))
    probs[:, 0] = 1 - probs[:, 1]

    precisions, cuts = slidingEfficiencyArray(answers, probs)

    lmb = massiveCorrectionFunction(answers, probs, 20)
    newCuts = lmb(cuts)

    mse = mean_squared_error(precisions, newCuts)
    max_mse = 0.001
    x_range = numpy.arange(0, 1, 0.01)
    if mse >= max_mse:
        # the second graph should look like approximation of the first one
        pylab.plot(cuts, precisions)
        pylab.plot(x_range, lmb(x_range))
        pylab.show()
        # these two plots should coincide
        pylab.plot(newCuts, precisions)
        pylab.plot(x_range, x_range)
        pylab.show()
        pylab.plot(precisions)
        pylab.plot(newCuts)
        pylab.show()

    assert mse < max_mse, "unexpectedly big deviation of mse " + str(mse)


def testCorrectionFunction():
    for i in range(5):
        testCorrectionFunctionIteration()
    print('correction function is ok')


testCorrectionFunction()


def computeBDTCut(target_efficiency, answers, prediction_probas):
    """Computes cut which gives targetEfficiency
    * targetEfficiency from 0 to 1
    * answers is an array of zeros and ones
    * predictionProbas is prediction probabilites returned by BDT at some step
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
    * x - array of values
    * width is float, if width == 0, this is simply heaviside function
    """
    if abs(width) > 0.0001:
        return 1.0 / (1.0 + numpy.exp(-x / width))
    else:
        return (x > 0) * 1.0


def generateSample(size, featuresNumber, distance=2.0):
    """
    Generates some test distribution,
    signal and background distributions are gaussian with same dispersion and different centers,
    all variables are independent (gaussian correlation matrix is identity)
    """
    X = numpy.zeros((size, featuresNumber))
    y = numpy.zeros(size)
    signal_indices, bg_indices = train_test_split(range(size), test_size=0.5)
    X[signal_indices, :] = numpy.random.normal(distance / 2., 1, (len(signal_indices), featuresNumber))
    X[bg_indices, :] = numpy.random.normal(-distance / 2., 1, (len(bg_indices), featuresNumber))

    y[signal_indices] = 1
    y[bg_indices] = 0

    columns = ["column" + str(x) for x in range(featuresNumber)]
    X = pandas.DataFrame(X, columns=columns)
    return X, y


def computeSignalKnnIndices(uniform_variables, dataframe, is_signal, n_neighbors=50):
    """For each event returns the knn closest signal events.
    Parameters:
        *uniform_variables* is list of names of variables,
        using which we want to compute the distance

        *dataframe* should contain these variables

        *is_signal* is boolean numpy.array
    Returns:
        ndarray of shape (len(dataframe), knn),
        each row contains indices of closest signal events
    """
    assert len(dataframe) == len(is_signal), "Different lengths"
    signal_indices = numpy.where(is_signal)[0]
    for variable in uniform_variables:
        if variable not in dataframe.columns:
            raise ValueError("Dataframe is missing %s column" % variable)
    uniforming_features_of_signal = numpy.array(dataframe.ix[is_signal, uniform_variables])
    neighbours = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(uniforming_features_of_signal)
    _, knn_signal_indices = neighbours.kneighbors(dataframe[uniform_variables])
    return numpy.take(signal_indices, knn_signal_indices)


def computeKnnIndicesOfSameClass(uniform_variables, trainX, trainY, n_neighbours=50):
    """Works as previous function, but returns the neighbours of the same class as element"""
    assert len(trainX) == len(trainY), "different size"
    is_signal = trainY > 0.5
    signal_knn = computeSignalKnnIndices(uniform_variables, trainX, is_signal, n_neighbours)
    bg_knn = computeSignalKnnIndices(uniform_variables, trainX, ~is_signal, n_neighbours)
    bg_knn[is_signal, :] = signal_knn[is_signal, :]
    return bg_knn


def testComputeSignalKnnIndices(n_events=100):
    df = pandas.DataFrame(numpy.random.rand(n_events, 10))
    is_signal = numpy.random.rand(n_events) > 0.5
    signal_indices = numpy.where(is_signal)[0]
    unif_columns = df.columns[:1]
    knn_indices = computeSignalKnnIndices(unif_columns, df, is_signal, 10)
    distances = pairwise_distances(df[unif_columns])
    # distances = MinkowskiDistance(p=2).pairwise(df[unif_columns])
    for i, neighbours in enumerate(knn_indices):
        assert numpy.all(is_signal[neighbours]), "returned indices are not signal"
        not_neighbours = [x for x in signal_indices if not x in neighbours]
        minr = numpy.min(distances[i, not_neighbours])
        maxr = numpy.max(distances[i, neighbours])
        assert minr >= maxr, "distances are set wrongly!"

    knn_all_indices = computeKnnIndicesOfSameClass(unif_columns, df, is_signal, 10)
    for i, neighbours in enumerate(knn_all_indices):
        assert numpy.all(is_signal[neighbours] == is_signal[i]), "returned indices are not signal/bg"

    print("computeSignalKnnIndices is ok")

testComputeSignalKnnIndices()


def smear_dataset(testX, smeared_variables=None, smearing_factor=0.1):
    """For the selected features 'smears' them in dataset,
    pay attention, that only float feature can be smeared by now. """
    assert isinstance(testX, pandas.DataFrame), "the passed object is not of type pandas.DataFrame"
    if smeared_variables is None:
        smeared_variables = testX.columns
    for var in smeared_variables:
        assert var in testX.columns, "The variable %s was not found in dataset"
    result = pandas.DataFrame.copy(testX)
    for var in smeared_variables:
        sigma = math.sqrt(numpy.var(result[var]))
        result[var] += normal(0, smearing_factor * sigma, len(result))
    return result





