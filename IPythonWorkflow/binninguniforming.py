from __future__ import division
from __future__ import print_function

import numpy
from commonutils import check_sample_weight, weighted_percentile, \
    build_normalizer, generate_sample
from numpy.random.mtrand import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import train_test_split


__author__ = 'Alex Rogozhnikov'
# About

# How does it work: it splits the mass space into bins,
# for each bin a separate basic classifier is learnt,
# when getting prediction_proba, the predictions of basic classifier
# are altered by some correcting monotonic function

# WARNING
# this module is concerned deprecated

# TODO try doing the same without binning


class Binner:
    def __init__(self, values, n_bins, sample_weight=None):
        """Binner is a class that helps to split the values into several bins.
        Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
        and thus we are computing limits (boundaries of bins)."""
        percentiles = [i / n_bins for i in range(1, n_bins)]
        sample_weight = check_sample_weight(values, sample_weight=sample_weight)
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



def get_train_test_data_for_bin(bins_data, bin_index, mode='outer'):
    """
    Mode = outer or innerNoSplit or innerSplit
    Returns binFitX, binCorrX, binFitY, binCorrY
    """
    binX = bins_data[bin_index][1]
    binY = bins_data[bin_index][2]

    if mode == 'outer':
        other_bins = bins_data[:bin_index] + bins_data[bin_index + 1:]
        binFitX = numpy.concatenate([bin_data[1] for bin_data in other_bins])
        binFitY = numpy.concatenate([bin_data[2] for bin_data in other_bins])
        binCorrX = binX
        binCorrY = binY
    elif mode == 'innerNoSplit':
        binFitX = binCorrX = binX
        binFitY = binCorrY = binY
    elif mode == 'innerSplit':
        binFitX, binCorrX, binFitY, binCorrY = \
            train_test_split(binX, binY, test_size=0.5)
    else:
        raise ValueError("something wrong was passed as 'mode' argument")
    return binFitX, binCorrX, binFitY, binCorrY


class BinningUniformClassifier(BaseEstimator, ClassifierMixin):
    """The meta-classifier, which tries to make efficiency uniform on the mass or some other variable"""
    def __init__(self, base_estimator, uniform_variables=None, n_bins=10, mode='outer',
                 correction_steps=30, train_variables=None):
        """
        * BaseClassifier is some sklearn classifier, which we build new classifier upon
        * uniformVariables - a list of variables (currently only one variable is supported)
             in which we want to get uniform efficiency, example: ['mass']
        * binsNumber - the number of bins on which we split the space of uniform variables
        """
        self.base_estimator = base_estimator
        self.n_bins = n_bins
        self.uniform_variables = uniform_variables
        self.mode = mode
        self.correction_steps = correction_steps
        self.train_variables = train_variables

    def fit(self, X, y):
        assert self.uniform_variables is not None, "set uniform variables in ctor!"
        assert len(self.uniform_variables) == 1, "only one uniforming variable is supported by now"
        if self.n_bins == 1 and self.mode == 'outer':
            raise Exception('Cannot use outer mode with only one bin')
        assert len(X) == len(y), "Different size of arrays"

        masses = X[self.uniform_variables[0]]
        # binning
        self.binner = Binner(masses, self.n_bins)
        self.classifiers = []
        self.normalizers = []
        bins_data = list(self.binner.split_into_bins(masses, self.get_train_variables(X), y))
        for binIndex in range(len(bins_data)):
            binFitX, binCorrX, binFitY, binCorrY = get_train_test_data_for_bin(bins_data, binIndex, self.mode)
            # binFitX, binFitY are used to fit base classifier
            # binCorrX, binCorrY are used to obtain uniformer function

            bin_classifier = clone(self.base_estimator)
            self.classifiers.append(bin_classifier.fit(binFitX, binFitY))
            predictions = bin_classifier.predict_proba(binCorrX)
            self.normalizers.append(build_normalizer(predictions[binCorrY > 0.5, 1]))

        return self

    def predict(self, X):
        # TODO set some threshold, simply max is absolutely irrelevant
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        # use binner for correcting predictions
        # it is simpler to fit efficiency
        masses = X.loc[:, self.uniform_variables[0]]
        X = self.get_train_variables(X)
        predicts = numpy.ndarray([len(masses), 2])
        bins_indices = self.binner.get_bins(masses)
        for bin_index in range(self.n_bins):
            indices = bins_indices == bin_index
            raw_proba = self.classifiers[bin_index].predict_proba(X[indices])
            predicts[indices, 1] = self.normalizers[bin_index](raw_proba[:, 1])
        predicts[:, 0] = 1. - predicts[:, 1]

        return predicts

    def get_train_variables(self, X):
        if self.train_variables is None:
            return X
        else:
            return X.loc[:, self.train_variables]


def test_binning_uniforming():
    from sklearn.lda import LDA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_auc_score

    trainX, trainY = generate_sample(1000, 10, 2)
    testX, testY = generate_sample(1000, 10, 2)

    uniform_variables = [trainX.columns[0]]

    unifier = BinningUniformClassifier(LDA(), uniform_variables=uniform_variables)
    _ = clone(unifier)

    unifier = BinningUniformClassifier(GaussianNB(), uniform_variables=uniform_variables, n_bins=1,
                                       mode='innerSplit').fit(trainX, trainY)
    _ = unifier.predict(testX)
    proba = unifier.predict_proba(testX)

    score = roc_auc_score(testY, proba[:, 1])
    if score < 0.8:
        print('Predictions are awful, roc = %.2f' % score)


test_binning_uniforming()