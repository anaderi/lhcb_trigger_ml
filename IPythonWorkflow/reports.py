# This module contains functions to build reports:
# training, getting predictions, building various plots


import numpy
import pylab
from sklearn.metrics.metrics import roc_auc_score, recall_score, roc_curve, auc
from sklearn.utils.validation import check_arrays
from commonutils import computeBDTCut, Binner
import time
import math

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


def trainClassifiers(classifiers_dict, trainX, trainY):
    """Trains all classifiers on the same train data"""
    for name, classifier in classifiers_dict.iteritems():
        start_time = time.time()
        classifier.fit(trainX, trainY)
        print "Classifier %12s is learnt in %0.2f seconds" % (name, time.time() - start_time)


def getClassifiersPredictionProba(classifiers_dict, testX):
    return {name: classifier.predict_proba(testX) for name, classifier in classifiers_dict.iteritems()}


def getClassifiersStagedPredictionProba(classifiers_dict, testX):
    """Returns dictionary:
        {classifier_name: staged_predict_proba of classifier} """
    result = {}
    for name, classifier in classifiers_dict.iteritems():
        try:
            result[name] = list(classifier.staged_predict_proba(testX))
        except AttributeError:
            print "Classifier %12s doesn't provide staged_predict_proba" % name
    return result


def getStageOfStagedProbaDict(stage, staged_predict_proba_dict):
    """Returns the predict_proba_dict, corresponding to 'stage' iteration
    of every classifier"""
    return {name: predictions[stage] for name, predictions in staged_predict_proba_dict.iteritems()}


def plotScoreVariableCorrelation(answers, prediction_proba, correlation_values,
                                 classifier_name="", var_name="", score_function=Efficiency,
                                 bins_number=20, thresholds=None, y_limits=None,
                                 show_legend=False):
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
    if y_limits is not None:
        pylab.ylim(y_limits)
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
                                 var_name='mass', score_function=Efficiency, y_limits=(0, 1))


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


def computeMseVariation(answer, prediction_proba, mass, binner):
    """Computes 'non-flatness' of predictions, the lesser output, the better
    :param answer: numpy.array
    :param prediction_proba: numpy.ndarray
    :param mass: numpy.array
    :param binner: Binner
    """
    cuts = [computeBDTCut(target_eff, answer, prediction_proba) for target_eff in [(i + 1.0) / 11 for i in range(10)]]
    bins_data = binner.split_into_bins(mass, answer, prediction_proba)
    result = 0
    for cut in cuts:
        efficiencies = []
        for bin_masses, bin_answer, bin_proba in bins_data:
            efficiency = recall_score(bin_answer, bin_proba[:, 1] > cut)
            efficiencies.append(efficiency)
        result += numpy.std(efficiencies) ** 2
    return math.sqrt(result * 1.0 / binner.bins_number())


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
    assert len(testX) == len(testY), "Different size of arrays"
    predict_proba_dict = getClassifiersPredictionProba(classifiers_dict, testX)
    plotScoreVariableCorrelationSide2SideByPredictProba(predict_proba_dict, testX, testY, var_name,
                                          score_function=score_function, **kwargs)

