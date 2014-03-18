__author__ = 'Alex Rogozhnikov'

import numpy
import commonutils
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import train_test_split
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score


# How does it work: it splits the mass space into bins,
# for each bin a separate basic classifier is learnt,
# when getting prediction_probas, the predictions of basic classifier
# are altered by some correcting monotonic function



def getTrainTestDataForBin(bins_data, bin_index, mode='outer'):
    """
    Mode = outer or innerNoSplit or innerSplit
    Returns binFitX, binCorrX, binFitY, binCorrY
    """
    binX = bins_data[bin_index][1]
    binY = bins_data[bin_index][2]

    if mode == 'outer':
        otherBins = bins_data[:bin_index] + bins_data[bin_index + 1:]
        binFitX = numpy.concatenate([binData[1] for binData in otherBins])
        binFitY = numpy.concatenate([binData[2] for binData in otherBins])
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
        self.binner = commonutils.Binner(masses, self.n_bins)
        self.classifiers = []
        self.lambda_uniformers = []
        self.massive_lambda_uniformers = []
        splittedBinsData = list(self.binner.split_into_bins(masses, self.get_train_variables(X), y))
        for binIndex in range(len(splittedBinsData)):
            binFitX, binCorrX, binFitY, binCorrY = \
                getTrainTestDataForBin(splittedBinsData, binIndex, self.mode)
            # binFitX, binFitY are used to fit base classifier
            # binCorrX, binCorrY are used to obtain uniformer function

            binClassifier = clone(self.base_estimator)
            self.classifiers.append(binClassifier.fit(binFitX, binFitY))
            predictions = binClassifier.predict_proba(binCorrX)
            lmb = commonutils.correctionFunction(binCorrY, predictions, self.correction_steps)
            self.lambda_uniformers.append(lmb)
            self.massive_lambda_uniformers.append(
                commonutils.massiveCorrectionFunction(binCorrY, predictions, self.correction_steps))
        return self

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)
        # masses = X[self.uniform_variables[0]]
        # X_train_vars = self.get_train_variables(X)
        #
        # result = numpy.zeros(len(X))
        # bins_indices = self.binner.get_bins(masses)
        #
        # for bin_index in range(self.n_bins):
        #     indices = bins_indices == bin_index
        #     result[indices] = self.classifiers[bin_index].predict(X_train_vars[indices])
        # return result

    def predict_proba(self, X):
        # use binner for correcting predictions
        masses = X[self.uniform_variables[0]]
        X_train_vars = self.get_train_variables(X)
        predicts = numpy.ndarray([len(masses), 2])
        bins_indices = self.binner.get_bins(masses)
        for bin_index in range(self.n_bins):
            indices = bins_indices == bin_index
            raw_probas = self.classifiers[bin_index].predict_proba(X_train_vars[indices])
            predicts[indices, 1] = 1. - self.massive_lambda_uniformers[bin_index](raw_probas[:, 1])
        predicts[:, 0] = 1. - predicts[:, 1]

        # slow place
        # predicts2 = numpy.zeros([len(masses), 2])
        # for i in range(len(masses)):
        #     binNumber = bins_indices[i]
        #     p = self.classifiers[binNumber].predict_proba(X.irow(i))[0, 1]
        #     lmb = self.lambda_uniformers[binNumber]
        #     pNew = 1 - lmb(p)
        #     predicts2[i, 0] = 1 - pNew
        #     predicts2[i, 1] = pNew
        # assert numpy.max(abs(predicts - predicts2)) < 0.001, "interpolation works in the wrong way"
        return predicts

    def get_train_variables(self, X):
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]




def testBinningUniformerClassifier():
    trainX, trainY = commonutils.generateSample(1000, 10, 2)
    testX, testY = commonutils.generateSample(1000, 10, 2)

    uniformVariables = [trainX.columns[0]]

    unifier = BinningUniformClassifier(LDA(), uniform_variables=uniformVariables)
    _ = clone(unifier)

    unifier = BinningUniformClassifier(GaussianNB(), uniform_variables=uniformVariables, n_bins=1,
                                       mode='innerNoSplit').fit(trainX, trainY)
    predictions = unifier.predict(testX)
    _ = unifier.predict_proba(testX)

    if f1_score(predictions, testY) < 0.75:
        print 'predictions are awful, f1 = ' + str(f1_score(predictions, testY))
    print 'uniformer classifier is ok'


testBinningUniformerClassifier()


