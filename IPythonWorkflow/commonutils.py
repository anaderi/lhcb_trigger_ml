# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #About
# 
# This file contains some helpful functions and classes which are often used.
# This file is ROOT-independent
import time

from matplotlib.pyplot import figure, plot

import pandas
import numpy
import pylab
import math

from numpy import rec
from math import floor

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error

Precision = precision_score
Recall = recall_score
F1Score = f1_score



def constantArray(length, value, dtype = 'int16'):
    return numpy.zeros(length, dtype=dtype) + value

def addIsSignalColumn(dataFrame, is_signal):
    """Is signal can be either 1 or 0 or array """
    dataFrame["IsSignal"] = is_signal

def getProbabilitiesOfSignal(classifier, test_data):
    """predictProba returns the 2d array, 
        [:,0] - probabilities of 0 class (bg)
        [:,1] - probabilities of 1 class (signal)
    """
    return classifier.predict_proba(test_data)[:,1]


def shuffleDataSet(dataFrame, answers):
    """Shuffles the rows in the dataFrame and answersColumn simultaneously
    Pay attention that dataFrame is changed in the procedure,
    this may cause some side-effects, so if you need original dataFrame, use clone() before
    """
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
    signalTrainInd, signalTestInd = train_test_split(range(len(signalDataFrame)), 
        train_size = signalTrainPart)
    
    bgTrainInd, bgTestInd = train_test_split(range(len(bgDataFrame)), 
        train_size = bgTrainPart)
    
    signalTrain = signalDataFrame.irow(signalTrainInd)
    signalAnsTrain = constantArray(len(signalTrainInd), 1)
    signalTest  = signalDataFrame.irow(signalTestInd)
    signalAnsTest = constantArray(len(signalTestInd), 1)
    
    bgTrain = bgDataFrame.irow(bgTrainInd)
    bgAnsTrain = constantArray(len(bgTrainInd), 0)
    bgTest  = bgDataFrame.irow(bgTestInd)
    bgAnsTest = constantArray(len(bgTestInd), 0)
    
    # Concatenating in single dataframe
    train = pandas.concat([signalTrain, bgTrain], join='inner', ignore_index=True)
    test  = pandas.concat([signalTest, bgTest],   join='inner', ignore_index=True)
    trainAns = numpy.concatenate((signalAnsTrain, bgAnsTrain))
    testAns = numpy.concatenate((signalAnsTest, bgAnsTest))
    
    # Shuffling. It isn't mandatory, in case classifier would somehow take order into account
    # it is better to shuffle data
    train, trainAns = shuffleDataSet(train, trainAns)
    test, testAns = shuffleDataSet(test, testAns)
    
    return train, trainAns, test, testAns



def buildRocCurves(testAnswer, testPredictionProbas, isBigPlot = True):
    """
    testAnswer in numpy.array with zeros and ones
    testPredictions is dictionary:
    - key is string (classifier name usually)
    - value is numpy.array with probabilities of class 1
    """
    if isBigPlot:
        figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    pylab.clf()
    for classifierName, predictions in testPredictionProbas.iteritems():
        fpr, tpr, thresholds = roc_curve(testAnswer, predictions[:,1])
        # tpr = recall = isSasS / isS = signalEfficiecncy
        # fpr = isBasS / isB = 1 - specifity ?=?  1 - backgroundRejection
        bgRej =  1 - numpy.array(fpr)
        roc_auc = auc(fpr, tpr)
        pylab.plot(tpr, bgRej, label='%s (area = %0.3f)' % (classifierName, roc_auc))
    
    pylab.plot([0, 1], [1, 0], 'k--')
    pylab.xlim([-0.003, 1.00])
    pylab.ylim([0.0, 1.003])
    pylab.xlabel('Signal Efficiency')
    pylab.ylabel('Background Rejection')
    pylab.title('Receiver operating characteristic (ROC)')
    pylab.legend(loc="lower left")

    pylab.show()



class Binner:
    """
    Binner is a class that helps to split the values into several bins.
    Initially an array of values is given, which is then splitted into 'bins_number' equal parts,
    and thus we are computing limits (boundaries of bins).
    """
    def __init__(self, values, bins_number):
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
        for bin in range(len(self.limits)+1):
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
    # assert numpy.all(binner.getBins(p) == binner.getBinsDumb2(p)), "getBins() function is wrong"

    binner = Binner(numpy.random.permutation(20), 5)
    p = numpy.random.permutation(40)
    # checking whether binner preserves correspondence
    list1 = list(binner.split_into_bins(numpy.array(range(-10, 30))[p], numpy.array(range(0, 40))[p]))
    for a, b in list1:
        for x, y in zip(a,b):
            assert x + 10 == y, 'transpositions are wrong after binning'
    binner = Binner(numpy.random.permutation(30), 3)
    res2 = list(binner.split_into_bins(range(10, 20)))
    ans2 = [[], range(10, 20), []]

    for a, b in zip(res2, ans2):
        for x, y in zip(a[0],b):
            assert x == y, 'binning is wrong'

    res3 = list(binner.split_into_bins(numpy.random.permutation(45)))
    ans3 = list(binner.split_into_bins(range(45)))
    for x, y in zip(res3, ans3):
        assert set(x[0]) == set(y[0]), "doesn't work well with permutations"

    p1 = numpy.random.permutation(100)
    p2 = numpy.random.permutation(100)

    # splitted_1 = list(binner.splitIntoBins(p1, p2))
    # splitted_2 = list(binner.splitIntoBinsDumb(p1, p2))
    #
    # for vals1, vals2 in zip(splitted_1, splitted_2):
    #     for arr1, arr2 in zip(vals1, vals2):
    #         assert set(arr1) == set(arr2), "something is wrong with splitIntoBins"

    print 'binner is ok'

testBinner()

# Score functions
# Some notation used here
# IsSignal - is really signal
# AsSignal - classified as signal
# IsBackgroundAsSignal - background, but classified as signal
# ... and so on. Cute, right?

def Efficiency(answer, prediction):
    """Efficiency = right classified signal / everything that is really signal
    Efficiency == recall
    """
    isSignal = 0.01 + numpy.sum(answer)
    isSignalAsSignal = numpy.sum(answer * prediction)
    return isSignalAsSignal * 1.0 / isSignal

def partOfIsSignal(answer, prediction):
    """Part of is signal = signal events / total amount of events"""
    if len(answer) != len(prediction):
        raise ValueError("Different size of arrays")
    return numpy.sum(answer) * 1.0 / len(answer)

def partOfAsSignal(answer, prediction):
    """Part of is signal = Is signal / total amount of events"""
    if len(answer) != len(prediction):
        raise ValueError("Different size of arrays")
    return numpy.sum(prediction) * 1.0 / len(answer)

def plotScoreVariableCorrelation(answers, prediction_proba, correlation_values,
        classifier_name="", var_name="", score_function=Efficiency,
        bins_number=20, thresholds=None, y_limits=None, draw_separately=True,
        is_big_plot=True, show_legend=False):
    """
    Different score functions available: Efficiency, Precision, Recall, F1Score, 
    and other things from sklearn.metrics
    var_name - for example, 'mass', just a name for plotting.
    """

    if thresholds is None:
        thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
    if is_big_plot:
        figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    # TODO smoothing and many-binning calculations
    binner = Binner(correlation_values, bins_number=bins_number)
    bins_data = binner.split_into_bins(correlation_values, answers, prediction_proba)
    for threshold in thresholds:
        x_values = []
        y_values = []
        for bin_data in bins_data:
            masses = bin_data[0]
            answers = bin_data[1]
            probabilities = bin_data[2]
            y_values.append(score_function(answers, probabilities[:,1] > threshold))
            x_values.append(numpy.mean(masses))
        plot(x_values, y_values, label="threshold = %0.2f" % threshold)

    pylab.title("Correlation with results of " + classifier_name)
    pylab.xlabel(var_name)
    pylab.ylabel(score_function.__name__)
    if y_limits is not None:
        pylab.ylim(y_limits)
    if show_legend:
        pylab.legend(loc="lower right")
    if draw_separately:
        pylab.show()


def plotMassEfficiencyCorrelation(answers, predictionsProbabilities,
                                  masses, classifierName):
    """
    Just a particular case of previous roccfunction
    Splits all the events by mass into 20 bins of equal size, 
    computes efficiency for each bin and draws a plot
    - answers - array of 0 and 1
    - predictionProbabilities - array of probabilities given by classifier
    - masses - array of masses
    """
    plotScoreVariableCorrelation(answers, predictionsProbabilities, masses,
                             classifierName, var_name= 'mass', score_function= Efficiency )


def SplitIntoArrayBins(binsNumber, mainVariable, *arrays):
    """Returns the generator for bin data
    Example:
    list( SplitIntoArrayBins(4, range(40), range(5,45)) )
    """
    jointArray = rec.fromarrays([mainVariable] + list(arrays))
    jointArray.sort()
    for binIndex in range(binsNumber):
        first = len(jointArray) * binIndex / binsNumber 
        afterLast = len(jointArray) * (binIndex + 1) / binsNumber 
        jointPart = jointArray[first:afterLast]
        yield list([jointPart["f" + str(i)] for i in range(len(arrays) + 1)])


def SplitIntoRecBins(binsNumber, mainVariable, *arrays):
    """Returns the list of bin data
    Each entry is recarray, which has fields f0, f1, f2
    corresponding to input a-rray (f0 - main variable)
    """
    jointArray = rec.fromarrays([mainVariable] + list(arrays))
    jointArray.sort()
    for binIndex in range(binsNumber):
        first = len(jointArray) * binIndex / binsNumber 
        afterLast = len(jointArray) * (binIndex + 1) / binsNumber 
        yield jointArray[first:afterLast]


def slidingEfficiencyArray(answers, prediction_probas):
    """Returns two arrays,
    if threshold == second array[i]
    then efficiency == first array[i] (approximately)
    """
    assert len(answers) == len(prediction_probas), "different size of arrays"
    predictionProbabilities = prediction_probas[:,1]
    indices = numpy.argsort(predictionProbabilities)
    ans = answers[indices]
    probs = predictionProbabilities[indices]
    AsSig = len(ans)
    IsSig = numpy.sum(ans)
    IsSigAsSig = numpy.sum(ans)
    precision = numpy.zeros(len(ans))
    for i in range(len(ans)):
        AsSig -= 1
        if(ans[i]):
            IsSigAsSig -= 1
        precision[i] = IsSigAsSig * 1.0 / (IsSig + 0.0001)
    return precision, probs


def getQuantilesOnTargets(sortedArray, targets):
    return numpy.minimum(numpy.searchsorted(sortedArray, targets), len(sortedArray) - 1)

def getQuantiles(sortedArray, n):
    """Get the positions, at which array values 
    becomes greater then quantiles (well, this is what really quantiles is,
    though numpy quantiles are a bit different thing)
    """
    targets = [0.0 + i * 1.0 / n for i in range(0, n+1)]
    return getQuantilesOnTargets(sortedArray, targets)


def getQuantilesOfTargetsPrecise(sortedArray, targets):
    #print 'targets', targets, len(targets)
    if len(targets) == 0:
        return
    currentTarget = 0
    for i in range(len(sortedArray)):
        while(targets[currentTarget] <= sortedArray[i]):
            yield i
            currentTarget += 1
            if(currentTarget >= len(targets)):
                return
    while currentTarget < len(targets):
        yield len(sortedArray) 
        currentTarget += 1      


def getProbabilityQuantiles(answers, prediction_probas, n, isFunctionGrowing = True):
    precision, cuts = slidingEfficiencyArray(answers, prediction_probas)

    assert numpy.all(cuts == numpy.sort(cuts)), 'Something wrong with cuts - not monotonic '
    indices = list(getQuantiles(cuts, n))

    return precision[indices]

def interpolate(y_array, x):
    """Assuming we have a function, that has at point i value y_array[i]
    Then it returns piecewise-linear interpolation of it at point x"""
    if x >= len(y_array) - 1.001:
        return y_array[-1]
    if x <= 0:
        return y_array[0]
    n = int(floor(x))
    t = x - n
    return y_array[n] * (1 - t) + y_array[n+1] * t

# TODO use this function
def massive_interpolate(y_array, x):
    """The same as interpolate, but x is array now
    returns array of the same length as x
    """
    y_array = numpy.array(y_array)
    x = numpy.minimum(x, len(y_array) - 1.0001)
    x = numpy.maximum(x, 0.0001)
    n = numpy.array(numpy.int(numpy.floor(x)), dtype=numpy.int)
    t = x - n
    return y_array.take(n) * (1.0 - t) + y_array.take(n+1) * t


def correctionFunction(answers, predictedProbabilities, steps = 10):
    quantiles = getProbabilityQuantiles(answers, predictedProbabilities, steps, False)
    #print 'steps, quantiles = ', steps, quantiles
    return lambda x: interpolate(quantiles, x * (len(quantiles) - 1) )


def plotFunction(lmb, segment=None):
    if segment is None:
        segment = [0,1]
    xpoints = numpy.arange(segment[0], segment[1], (segment[1] - segment[0] * 0.01))
    plot(xpoints, [lmb(x) for x in xpoints])

def testQuantiles():
    y = numpy.array(range(100)) * 0.0101
    targets = [0.005 + 0.099 * i for i in range(10)]
    y = y*y*y
    quantiles = list(getQuantilesOnTargets(y, targets))
    for i, target in zip(quantiles, targets):
        assert y[i-1] <= target <= y[i], 'quantiles are wrong'
    print 'quantiles are ok'
    

def testCorrectionFunctionIteration():
    l = 100
    answers1 = numpy.zeros(l)
    answers2 = numpy.zeros(l) + 1
    answers = numpy.concatenate((answers1, answers2))
    probs1  = numpy.random.rand(l) * numpy.random.rand(l)
    probs2  = - numpy.random.rand(l) * numpy.random.rand(l) + 1.0
    probs = numpy.zeros( (len(probs1) + len(probs2), 2))
    probs[:,1] = numpy.concatenate((probs1, probs2))
    probs[:,0] = 1 - probs[:,1]
    
    #plot(sort(probs1), sort(probs2))
    
    precisions, cuts = slidingEfficiencyArray(answers, probs)
    
    lmb = correctionFunction(answers, probs, 20)
    newCuts = list([lmb(x) for x in cuts])

    mse = mean_squared_error(precisions, newCuts)
    
    maxMse = 0.001
    if mse >= maxMse:
        # the second graph should look like approximation of the first one
        plot(cuts, precisions)
        plotFunction(lmb)
        pylab.show()
        # this two graphs should coincide
        plot(newCuts, precisions)
        plotFunction(lambda x: x)
        pylab.show()
        plot(precisions)
        plot(newCuts)
        pylab.show()
        
    assert mse < maxMse, "unexpectedly big deviation of mse " + str(mse)

def testCorrectionFunction():
    for i in range(10):
        testCorrectionFunctionIteration()
    print 'Correction function is ok'
    
    
testQuantiles()
testCorrectionFunction()


def efficiencyPlotData(answers, prediction_probas):
    precisions, cuts = slidingEfficiencyArray(answers, prediction_probas)
    return cuts, precisions

def efficiencyPlotData2(answers, prediction_probas, cuts = None, scoreFunc = Recall):
    """All the same like precisionPlotData, but 10 times slower.
    Can compute not only recall, but other score functions as well"""
    if cuts is None:
        cuts = numpy.array(range(100)) * 0.01
    precisions = []
    for cut in cuts:
        precisions.append(scoreFunc(answers, prediction_probas[:,1] > cut) )
    return cuts, precisions
    
def TestEfficiencyPlotFunctions():
    for i in range(5):
        length =  (i + 1) * 100
        getRand = lambda : numpy.random.rand(length)
        predict_probas = numpy.zeros((length, 2))
        predict_probas[:,1] = getRand() * getRand() 
        predict_probas[:,0] = 1 - predict_probas[:,1]
        
        res = getRand() * 0.4 + 0.2
        answers = predict_probas[:,1] > res
        cuts, precisions = efficiencyPlotData(answers, predict_probas)
        _, precisions2 = efficiencyPlotData2(answers, predict_probas, cuts = cuts)
        mse = mean_squared_error(precisions, precisions2)
        maxMse = 1e-8
        if mse >= maxMse:
            plot(cuts, precisions)
            plot(cuts, precisions2)
            pylab.show()
        assert mse < maxMse, "Something wrong with mse of efficiency functions, mse = " + str(mse)
    print "EfficiencyPlotData functions are ok"
    
TestEfficiencyPlotFunctions()

# <codecell>

# execution time comparison
#%timeit efficiencyPlotData(numpy.random.rand(1000) > 0.5, numpy.random.rand(1000))
#%timeit efficiencyPlotData2(numpy.random.rand(1000) > 0.5, numpy.random.rand(1000))

# <codecell>



def computeEfficiency(cut, answers, predictionProbas):
    return recall_score(answers, predictionProbas[:,1] > cut)


def computeBDTCut(target_efficiency, answers, prediction_probas):
    """Computes cut which gives targetEfficiency
    * targetEfficiency from 0 to 1
    * answers is an array of zeros and ones
    * predictionProbas is prediction probabilites returned by BDT at some step
    """
    assert len(answers) == len(prediction_probas), "different size"

    indices = (answers == 1)
    signal_probas = prediction_probas[indices, 1]
    return numpy.percentile(signal_probas, 100 - target_efficiency * 100)


def computeLocalEfficiencies(globalCut, knnIndices, answers, prediction_proba):
    """Fast implementation in numpy"""
    assert len(answers) == len(prediction_proba), 'different size'
    predictions = prediction_proba[:, 1] > globalCut
    neigh_predictions = numpy.take(predictions, knnIndices)
    return neigh_predictions.mean(axis=1)


def ComputeLocalEfficienciesDumb(globalCut, knnIndices, answers, prediction_proba):
    """Slow, but obvious realization"""
    assert len(answers) == len(prediction_proba), 'different size'
    result = numpy.zeros(len(answers))
    predictions = prediction_proba[:, 1] > globalCut

    for i in range(len(knnIndices)):
        neighbours = knnIndices[i]
        result[i] = numpy.sum(predictions[neighbours]) * 1.0 / len(neighbours)
    return result


def plotVsMass(mass, values, is_signal):
    assert len(mass) == len(values), 'different size'
    assert len(mass) == len(is_signal), 'different size'
    
    plot(mass[is_signal], values[is_signal], ',', label='signal')
    isBG = (is_signal == False)
    plot(mass[isBG], values[isBG], ',', label='bg')


def sigmoidFunction(x, width):
    """
    Sigmoid function is smoothing oh Heaviside function, the lesser width,
     the closer we are to Heaviside function
    Parameters:
    * x - array of values
    * width is float
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
    X[signal_indices,:] = numpy.random.normal(distance / 2, 1, (len(signal_indices), featuresNumber))
    X[bg_indices,:]  = numpy.random.normal(-distance / 2, 1, (len(bg_indices), featuresNumber))

    y[signal_indices] = 1
    y[bg_indices] = 0

    columns = ["column" + str(x) for x in range(featuresNumber)]
    X = pandas.DataFrame(X, columns=columns)
    return X, y

def computeMseVariation(answer, prediction_proba, mass, binner):
    cuts = [computeBDTCut(target_eff, answer, prediction_proba) for target_eff in [(i+1.0)/11 for i in range(10)]]
    bins_data = binner.split_into_bins(mass, answer, prediction_proba)
    result = 0
    for cut in cuts:
        efficiencies = []
        for bin_masses, bin_answer, bin_proba in bins_data:
            efficiencies.append(computeEfficiency(cut, bin_answer, bin_proba))
        result += numpy.std(efficiencies) ** 2
    return math.sqrt(result * 1.0 / binner.bins_number())



def trainClassifiers(classifiers_dict, trainX, trainY):
    for name, classifier in classifiers_dict.iteritems():
        start_time = time.time()
        classifier.fit(trainX, trainY)
        print "Classifier %10s is learnt in %0.2f seconds" % (name, time.time() - start_time)


def getClassifiersPredictionProba(classifiers_dict, testX):
    return {name: classifier.predict_proba(testX) for name, classifier in classifiers_dict.iteritems()}


def getClassifiersStagedPredictionProba(classifiers_dict, testX):
    result = {}
    for name, classifier in classifiers_dict.iteritems():
        try:
            result[name] = list(classifier.staged_predict_proba(testX))
        except AttributeError:
            print "Classifier %s doesn't provide staged_predict_proba" % name
    return result

