# This module contains functions to build reports:
# training, getting predictions, building various plots

try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict

import numpy
import pandas
import pylab
from sklearn.metrics import roc_auc_score, recall_score, roc_curve, auc
from sklearn.utils.validation import check_arrays
from commonutils import computeBDTCut, Binner
import time
import math
from matplotlib import cm

__author__ = 'Alex Rogozhnikov'


# Score functions
# Some notation used here
# IsSignal - is really signal
# AsSignal - classified as signal
# IsBackgroundAsSignal - background, but classified as signal
# ... and so on. Cute, right?

def Efficiency(answer, prediction):
    """Efficiency = right classified signal / everything that is really signal
    Efficiency == recall"""
    # assert len(answer) == len(prediction), "Different size of arrays"
    # isSignal = 0.01 + numpy.sum(answer)
    # isSignalAsSignal = numpy.sum(answer * prediction)
    # return isSignalAsSignal * 1.0 / isSignal
    return recall_score(answer, prediction)


def BackgroundEfficiency(answer, prediction):
    """BackgroundEfficiency = right classified bg / everything that is really bg"""
    return Efficiency(1 - answer, 1 - prediction)


def partOfIsSignal(answer, prediction):
    """Part of is signal = signal events / total amount of events"""
    assert len(answer) == len(prediction), "Different size of arrays"
    return numpy.sum(answer) * 1.0 / len(answer)


def partOfAsSignal(answer, prediction):
    """Part of is signal = Is signal / total amount of events"""
    assert len(answer) == len(prediction), "Different size of arrays"
    return numpy.sum(prediction) * 1.0 / len(answer)


# NB:
# everything is kept as dictionaries,
# classifiers_dict = {classifier_name: classifier object}
# predict_proba_dict = {classifier_name: it's predict_proba}
# staged_predict_proba_dict = {classifier_name: it's staged_predict_proba}


def trainClassifiers(classifiers_dict, trainX, trainY, ipc_profile=None):
    """Trains all classifiers on the same train data"""
    if ipc_profile is None:
        for name, classifier in classifiers_dict.iteritems():
            start_time = time.time()
            classifier.fit(trainX, trainY)
            print "Classifier %12s is learnt in %0.2f seconds" % (name, time.time() - start_time)
    else:
        from IPython.parallel import Client
        client = Client(profile=ipc_profile)
        start_time = time.time()
        cview = client.load_balanced_view()
        def trainClassifier(name_classifier, X, y):
            name_classifier[1].fit(X, y)
            return name_classifier
        result = cview.map_sync(trainClassifier, classifiers_dict.iteritems(),
                                [trainX] * len(classifiers_dict),  [trainY] * len(classifiers_dict))
        print "We spent %.3f seconds on parallel training" % (time.time() - start_time)
        for name, classifier in result:
            classifiers_dict[name] = classifier


def getClassifiersPredictionProba(classifiers_dict, testX):
    return OrderedDict([(name, classifier.predict_proba(testX))
                        for name, classifier in classifiers_dict.iteritems()])


def getClassifiersStagedPredictionProba(classifiers_dict, testX):
    """Returns dictionary:
        {classifier_name: staged_predict_proba of classifier} """
    result = OrderedDict()
    for name, classifier in classifiers_dict.iteritems():
        try:
            result[name] = list(classifier.staged_predict_proba(testX))
        except AttributeError:
            print "Classifier %12s doesn't provide staged_predict_proba" % name
    return result


def getStageOfStagedProbaDict(stage, staged_predict_proba_dict):
    """Returns the predict_proba_dict, corresponding to 'stage' iteration
    of every classifier"""
    return OrderedDict([(name, predictions[stage])
                        for name, predictions in staged_predict_proba_dict.iteritems()])


def plotScoreVariableCorrelation(answers, prediction_proba, correlation_values,
                                 classifier_name="", var_name="", score_function=Efficiency,
                                 bins_number=20, thresholds=None, ylim=None, show_legend=False):
    """
    Different score functions available: Efficiency, Precision, Recall, F1Score,
    and other things from sklearn.metrics
    var_name - for example, 'mass', just a name for plotting.
    """
    answers, prediction_proba, correlation_values = \
        check_arrays(answers, prediction_proba, correlation_values)

    if thresholds is None:
        thresholds = [computeBDTCut(eff, answers, prediction_proba) for eff in [0.2, 0.4, 0.5, 0.6, 0.8]]

    binner = Binner(correlation_values, bins_number=bins_number)
    bins_data = binner.split_into_bins(correlation_values, answers, prediction_proba)
    for threshold in thresholds:
        x_values = []
        y_values = []
        for bin_data in bins_data:
            masses = bin_data[0]
            answers = bin_data[1]
            probabilities = bin_data[2]
            y_values.append(score_function(answers, probabilities[:, 1] > threshold))
            x_values.append(numpy.mean(masses))
        pylab.plot(x_values, y_values, label="threshold = %0.2f" % threshold)

    pylab.title("Correlation with results of " + classifier_name)
    pylab.xlabel(var_name)
    pylab.ylabel(score_function.__name__)
    if ylim is not None:
        pylab.ylim(ylim)
    if show_legend:
        pylab.legend(loc="lower right")


def plotMassEfficiencyCorrelation(answers, prediction_proba, masses, classifier_name):
    """
    Just a particular case of previous function
    Splits all the events by mass into 20 bins of equal size,
    computes efficiency for each bin and draws a plot
    - answers - array of 0 and 1
    - predictionProbabilities - array of probabilities given by classifier
    - masses - array of masses
    """
    plotScoreVariableCorrelation(answers, prediction_proba, masses, classifier_name,
                                 var_name='mass', score_function=Efficiency, ylim=(0, 1))


def plotLearningCurves(answers, staged_proba_dict, step=1, metrics=roc_auc_score):
    """Plots learning curves of several classifiers,
    'metrics' is evaluated after each 'step' iterations"""
    for classifier_name, staged_probas in staged_proba_dict.iteritems():
        roc = [metrics(answers, prediction_proba[:, 1]) for prediction_proba in staged_probas[::step]]
        pylab.plot(range(len(staged_probas))[::step], roc, label=classifier_name)
    pylab.legend(loc="lower right")
    pylab.xlabel("stage")
    pylab.ylabel("ROC AUC")
    pylab.show()


def plotRocCurves(answers, predict_proba_dict, is_big_plot=True):
    """TestAnswer in numpy.array with zeros and ones
    testPredictions is dictionary:
    - key is string (classifier name usually)
    - value is numpy.array with probabilities of class 1
    """
    if is_big_plot:
        pylab.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    for classifierName, predictions in predict_proba_dict.iteritems():
        assert len(answers) == len(predictions), "different length"
        fpr, tpr, thresholds = roc_curve(answers, predictions[:, 1])
        # tpr = recall = isSasS / isS = signalEfficiecncy
        # fpr = isBasS / isB = 1 - specifity ?=?  1 - backgroundRejection
        bgRej = 1 - numpy.array(fpr)
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


def compute1DMseVariation(answers, prediction_proba, mass, binner, target_efficiencies=None):
    """Computes 'non-flatness' of predictions, the lesser output, the better
    :param answers: numpy.array
    :param prediction_proba: numpy.ndarray
    :param mass: numpy.array
    :param binner: Binner
    """
    if target_efficiencies is None:
        target_efficiencies = [(i + 1.0) / 11 for i in range(10)]

    bin_indices = binner.get_bins(mass)
    return computeMseVariationOnBins(answers > 0.5, prediction_proba,
                                     bin_indices=bin_indices, target_efficiencies=target_efficiencies)


def computeNdimentionalBinIndices(X, var_names, bin_limits):
    """For arbitrary number of variables computes the indices of data,
    the indices are unique numbers of bin from zero to \prod_j (len(bin_limits[j])+1)
    Example:
        var_names = ["M2AB", "M2AC"]
        bin_limits = [numpy.linspace(0, 1, 20), numpy.linspace(0, 1, 20)]
    """
    assert len(var_names) == len(bin_limits), "Different size of arrays"
    bin_indices = numpy.zeros(len(X), dtype=numpy.int)
    for var_name, bin_limits_axis in zip(var_names, bin_limits):
        bin_indices *= (len(bin_limits_axis) + 1)
        bin_indices += numpy.searchsorted(bin_limits_axis, X[var_name])
    return bin_indices


def computeLocalEfficienciesOfBins(answers, prediction_proba, bin_indices, n_total_bins, cut):
    assert len(answers) == len(prediction_proba) == len(bin_indices), "different size"
    is_signal = answers > 0.5
    # n_bins = numpy.max(bin_indices) + 1
    bin_total = numpy.bincount(bin_indices[is_signal], minlength=n_total_bins) + 1e-10

    passed_cut = prediction_proba[:, 1] > cut
    bin_passed_cut = numpy.bincount(bin_indices[is_signal & passed_cut], minlength=n_total_bins) - 1e-6
    return bin_passed_cut / bin_total


def computeMseVariationOnBins(is_signal, prediction_proba, bin_indices, target_efficiencies, power=2.):
    """ An efficient function to compute MSE """
    assert len(prediction_proba) == len(bin_indices) == len(is_signal), "different size"
    n_bins = numpy.max(bin_indices) + 1
    bin_total = numpy.bincount(bin_indices[is_signal], minlength=n_bins) + 1e-6
    signal_prediction_proba = prediction_proba[is_signal, :]
    signal_answers = numpy.ones(len(signal_prediction_proba), dtype=numpy.int)
    result = 0.
    cuts = computeBDTCut(numpy.array(target_efficiencies), signal_answers, signal_prediction_proba)
    for cut, efficiency in zip(cuts, target_efficiencies):
        passed_cut = signal_prediction_proba[:, 1] > cut
        mean_efficiency = numpy.sum(passed_cut) * 1. / len(passed_cut)
        bin_passed_cut = numpy.bincount(bin_indices[is_signal][passed_cut], minlength=n_bins)
        bin_efficiency = bin_passed_cut / bin_total
        result += numpy.sum(bin_total * numpy.abs(bin_efficiency - mean_efficiency) ** power)
    # Minkowski distance trick
    return 10 * (result / len(target_efficiencies) / len(signal_prediction_proba)) ** (1. / power)


def computeNdimensionalMseVariation(answers, predict_proba, X, var_names, efficiencies, n_bins=30, bin_limits=None):
    if bin_limits is None:
        bin_limits = []
        for var_name in var_names:
            var_data = X[var_name]
            bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])
    bin_indices = computeNdimentionalBinIndices(X, var_names, bin_limits)
    return computeMseVariationOnBins(answers > .5, predict_proba, bin_indices, efficiencies)


def computeStagedMseVariation(answers, testX, var_names, staged_predict_proba, stages, target_efficiencies,
                              n_bins=30, power=2.):
    is_signal = answers > 0.5

    bin_limits = []
    for var_name in var_names:
        var_data = testX[var_name]
        bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])

    bin_indices = computeNdimentionalBinIndices(testX, var_names, bin_limits)

    results = []
    for stage in stages:
        predict_probas = getStageOfStagedProbaDict(stage, staged_predict_proba)
        stage_variations = []
        for name, predict_proba in predict_probas.iteritems():
            stage_variations.append(computeMseVariationOnBins(is_signal, predict_proba, bin_indices=bin_indices,
                                          target_efficiencies=target_efficiencies, power=power))
        results.append(stage_variations)
    return pandas.DataFrame(results, columns=staged_predict_proba.keys(), index=stages)



def testComputeMseAndBins(size=500):
    columns = ['var1', 'var2']
    X = pandas.DataFrame(numpy.random.random((size, 2)), columns=columns)
    y = numpy.random.random(size) > 0.5
    proba = numpy.random.random((size, 2))
    computeNdimensionalMseVariation(y, proba, X, columns, [0.3, 0.5, 0.7])
    n_bins = 5
    x_limits = numpy.linspace(0, 1, n_bins + 1)[1:-1]
    bins = computeNdimentionalBinIndices(X, columns, [x_limits, x_limits])
    assert numpy.all(0 <= bins) and numpy.all(bins < n_bins * n_bins), "whooops"

    effs = computeLocalEfficienciesOfBins(y, proba, bins, n_bins * n_bins, .3)
    # print numpy.mean(effs)
    # todo continue testing


testComputeMseAndBins()


def plotScoreVariableCorrelationSide2SideByPredictProba(predict_proba_dict, testX, testY, var_name,
                                                        score_function=Efficiency, **kwargs):
    assert len(testX) == len(testY), "Different size of arrays"
    pylab.figure(figsize=(18, 7))
    for i, (name, predict_proba) in enumerate(predict_proba_dict.iteritems()):
        pylab.subplot(1, len(predict_proba_dict), i)
        plotScoreVariableCorrelation(testY, predict_proba, testX[var_name], classifier_name=name, var_name=var_name,
                                     score_function=score_function, **kwargs)
        pylab.title(name)
    pylab.show()


def plotScoreVariableCorrelationSide2Side(classifiers_dict, testX, testY, var_name,
                                          score_function=Efficiency, **kwargs):
    predict_proba_dict = getClassifiersPredictionProba(classifiers_dict, testX)
    plotScoreVariableCorrelationSide2SideByPredictProba(predict_proba_dict, testX, testY, var_name,
                                          score_function=score_function, **kwargs)



def plotEfficiency2D(var_name1, var_name2, testX, testY, probas_dict, target_efficiency, order=None, n_bins=30,
                     xlim=None, ylim=None, draw_difference=True):
    """This function plots the efficiency on 2D plot
    - var_name1 is name of first variable
    - var_name2 is name of second variable
    - target_efficiency: float between zero and one,
        the global cut is chosen to give this efficiency
    - order is list of strings, names of classifiers to compare
    - xlim - tuple (x_min, x_max) or None, just as for plot
    - ylim - tuple (y_min, y_max) or None
    """
    if order is None:
        order = probas_dict.keys()
    if xlim is None:
        xlim = numpy.min(testX[var_name1]), numpy.max(testX[var_name1])
    if ylim is None:
        ylim = numpy.min(testX[var_name2]), numpy.max(testX[var_name2])

    x_limits = numpy.linspace(xlim[0], xlim[1], (n_bins + 1))
    y_limits = numpy.linspace(ylim[0], ylim[1], (n_bins + 1))

    fig = pylab.figure(figsize=(5 + 5 * len(order), 7))
    for i, name in enumerate(order):
        predict_proba = probas_dict[name]
        cut = computeBDTCut(target_efficiency, testY, predict_proba)
        bin_indices = computeNdimentionalBinIndices(testX, [var_name1, var_name2], [x_limits[1:-1], y_limits[1:-1]])

        local_efficiencies = computeLocalEfficienciesOfBins(testY, predict_proba, cut=cut,
                bin_indices=bin_indices, n_total_bins=n_bins ** 2).reshape((n_bins, n_bins))
        if draw_difference:
            local_efficiencies[local_efficiencies < 0] = target_efficiency
            local_efficiencies -= target_efficiency
        ax = fig.add_subplot(1, len(order), i + 1)
        if draw_difference:
            p = ax.pcolor(x_limits, y_limits, local_efficiencies, cmap=cm.get_cmap("RdBu"), vmin=-0.2, vmax=+0.2)
        else:
            p = ax.pcolor(x_limits, y_limits, local_efficiencies, cmap=cm.get_cmap("RdBu"), vmin=0.0, vmax=1.0)
        ax.set_xlabel(var_name1)
        ax.set_ylabel(var_name2)
        ax.set_title(name)
        fig.colorbar(p, ax=ax)

    pylab.show()

