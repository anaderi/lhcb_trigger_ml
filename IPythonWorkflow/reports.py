# This module contains functions to build reports:
# training, getting predictions, building various plots
from numpy.testing.decorators import deprecated

try:
    from collections import OrderedDict, defaultdict
except ImportError:
    from ordereddict import OrderedDict

from warnings import warn
import time

import numpy
import pandas
import pylab
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils.validation import check_arrays
from matplotlib import cm

from commonutils import computeBDTCut, Binner


__author__ = 'Alex Rogozhnikov'


# Score functions
# Some notation used here
# IsSignal - is really signal
# AsSignal - classified as signal
# IsBackgroundAsSignal - background, but classified as signal
# ... and so on. Cute, right?

def Efficiency(answer, prediction):
    """Efficiency = right classified signal / everything that is really signal
    Efficiency == recall, returns -1 when ill-defined"""
    assert len(answer) == len(prediction), "Different size of arrays"
    isSignal =  numpy.sum(answer) - 1e-6
    isSignalAsSignal = numpy.sum(answer * prediction) + 1e-6
    return isSignalAsSignal / isSignal
    # the same, but with notifications
    # return recall_score(answer, prediction)


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

# Use OrderedDict instead of dict - the first keeps the order of elements


class ClassifiersDict(OrderedDict):
    def fit(self, X, y, ipc_profile=None):
        """Trains all classifiers on the same train data,
        if ipc_profile in not None, it is used as a name of ipython cluster to use for parallel computations"""
        if ipc_profile is None:
            for name, classifier in self.iteritems():
                start_time = time.time()
                classifier.fit(X, y)
                print("Classifier %12s is learnt in %.2f seconds" % (name, time.time() - start_time))
        else:
            from IPython.parallel import Client
            client = Client(profile=ipc_profile)
            start_time = time.time()
            lb_view = client.load_balanced_view()

            def trainClassifier(name_classifier, X, y):
                """ Trains one classifier on a separate node"""
                name_classifier[1].fit(X, y)
                return name_classifier

            result = lb_view.map_sync(trainClassifier, self.iteritems(), [X] * len(self),  [y] * len(self))
            print("We spent %.2f seconds on parallel training" % (time.time() - start_time))
            for name, classifier in result:
                self[name] = classifier

        return self

    def test_on(self, X, y, low_memory=False):
        return Predictions(self, X, y, low_memory)



class Predictions(object):
    def __init__(self, classifiers_dict, X, y, low_memory=False):
        """The main object for different reports and plots"""
        assert isinstance(classifiers_dict, OrderedDict)
        self.X = X
        self.y = numpy.array(y, dtype=int)
        self.is_signal = y > 0.5
        if low_memory:
            self.predictions = OrderedDict([(name, classifier.predict_proba(X))
                                           for name, classifier in classifiers_dict.iteritems()])
            self.staged_predictions = None
            self.classifiers = classifiers_dict
        else:
            self.predictions = OrderedDict()
            self.staged_predictions = OrderedDict()
            for name, classifier in classifiers_dict.iteritems():
                try:
                    self.staged_predictions[name] = list(classifier.staged_predict_proba(X))
                    self.predictions[name] = self.staged_predictions[name][-1]
                except AttributeError:
                    self.predictions[name] = classifier.predict_proba(X)

    @staticmethod
    def _check_efficiencies(efficiencies):
        if efficiencies is None:
            return [0.6, 0.7, 0.8, 0.9]
        else:
            return efficiencies

    def _check_mask(self, mask):
        """Checkes whether the mask is appropriate and normalizes it"""
        if mask is None:
            return numpy.ones(len(self.y), dtype=numpy.bool)
        assert len(mask) == len(self.y), 'wrogn size of mask'
        assert numpy.result_type(mask) == numpy.bool, 'the mask should be boolean'
        return mask

    def _get_staged_proba(self):
        if self.staged_predictions is not None:
            return self.staged_predictions
        else:
            result = OrderedDict()
            for name, classifier in self.classifiers.iteritems():
                try:
                    result[name] = classifier.staged_predict_proba(self.X)
                except AttributeError:
                    pass # maybe raise warning?
            return result

    def _get_stages(self, stages):
        result = OrderedDict()
        if stages is None:
            for name, preds in self.predictions.iteritems():
                result[name] = pandas.Series(data=[preds], index=['result'])
        else:
            stages = set(stages)
            for name, stage_preds in self._get_staged_proba().iteritems():
                result[name] = pandas.Series()
                for stage, pred in enumerate(stage_preds):
                    if stage not in stages:
                        continue
                    result[name].loc[stage] = numpy.copy(pred)
        return result

    def _map_on_staged_proba(self, function, step=1):
        """Applies a function to every step-th stage of each classifier
        returns: {name: Series[stage_name, result]}"""
        result = OrderedDict()
        for name, staged_proba in self._get_staged_proba().iteritems():
            result[name] = pandas.Series()
            for stage, preds in enumerate(staged_proba):
                if (stage + 1) % step != 0:
                    continue
                result[name].loc[stage] = function(preds)
        return result

    def _map_on_stages(self, function, stages=None):
        """returns a DataFrame """
        selected_stages = self._get_stages(stages)
        result = OrderedDict()
        for name, staged_proba in selected_stages.iteritems():
            result[name] = staged_proba.apply(function)
        return result

    def roc(self, stages=None):
        proba_on_stages = pandas.DataFrame(self._get_stages(stages))
        n_stages = len(proba_on_stages)
        self._strip_figure(n_stages)
        for i, (stage_name, proba_on_stage) in enumerate(proba_on_stages.iterrows()):
            pylab.subplot(1, n_stages, i + 1), pylab.title("stage " + str(stage_name))
            plotRocCurves(predict_proba_dict=proba_on_stage, answers=self.y, is_big_plot=False)
            pylab.title('ROC at stage ' + str(stage_name))
        return self

    def learning_curves(self, metrics=roc_auc_score, step=1):
        function = lambda predictions: metrics(self.y, predictions[:, 1])
        result = self._map_on_staged_proba(function=function, step=step)
        for classifier_name, staged_roc in result.iteritems():
            pylab.plot(staged_roc.keys(), staged_roc, label=classifier_name)
        pylab.legend(loc="lower right")
        pylab.xlabel("stage"), pylab.ylabel("ROC AUC")
        return self

    def _compute_bin_indices(self, var_names, n_bins=20, mask=None):
        """Mask is used to show events that will be binned after"""
        for var in var_names:
            assert var in self.X.columns, "the variable %i is not in dataset" % var
        mask = self._check_mask(mask)
        bin_limits = []
        for var_name in var_names:
            var_data = self.X[var_name][mask]
            bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])
        return computeBinIndices(self.X, var_names, bin_limits)

    def _compute_bin_centers(self, var_names, n_bins=20, mask=None):
        bin_centers = []
        mask = self._check_mask(mask)
        for var_name in var_names:
            var_data = self.X[var_name][mask]
            bin_centers.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), 2 * n_bins + 1)[1::2])
            assert len(bin_centers[-1]) == n_bins
        return bin_centers

    def _compute_staged_mse(self, var_names, target_efficiencies=None, step=3, n_bins=20, power=2., on_signal=True):
        target_efficiencies = self._check_efficiencies(target_efficiencies)
        bin_indices = self._compute_bin_indices(var_names, n_bins=n_bins)
        if on_signal:
            compute_mse = lambda pred: computeMseVariationOnBins(pred, self.is_signal, bin_indices,
                                    target_efficiencies=target_efficiencies, power=power)
        else:   # computing mse on background
            compute_mse = lambda pred: computeMseVariationOnBins(1. - pred, ~self.is_signal, bin_indices,
                                    target_efficiencies=target_efficiencies, power=power)
        return self._map_on_staged_proba(compute_mse, step)

    def _compute_mse(self, var_names, target_efficiencies=None, stages=None, n_bins=20, power=2., on_signal=True):
        target_efficiencies = self._check_efficiencies(target_efficiencies)
        bin_indices = self._compute_bin_indices(var_names, n_bins=n_bins)
        if on_signal:
            compute_mse = lambda pred: computeMseVariationOnBins(pred, self.is_signal, bin_indices,
                                    target_efficiencies=target_efficiencies, power=power)
        else:   # computing mse on background
            compute_mse = lambda pred: computeMseVariationOnBins(1. - pred, ~self.is_signal, bin_indices,
                                    target_efficiencies=target_efficiencies, power=power)
        return self._map_on_stages(compute_mse, stages=stages)

    def print_mse(self, uniform_variables, efficiencies=None, stages=None, in_html=True, on_signal=True):
        result = pandas.DataFrame(self._compute_mse(uniform_variables, efficiencies, stages=stages, on_signal=on_signal))
        if in_html:
            from IPython.display import display_html
            display_html("<b>Staged MSE variation</b>", raw=True)
            display_html(result)
        else:
            print("Staged MSE variation")
            print(result)
        return self

    def mse_curves(self, uniform_variables, target_efficiencies=None, n_bins=20, step=3, power=2., on_signal=True):
        result = self._compute_staged_mse(uniform_variables, target_efficiencies, step=step,
                                          n_bins=n_bins, power=power, on_signal=on_signal)
        for name, mse_stages in result.iteritems():
            pylab.plot(mse_stages.keys(), mse_stages, label=name)
            pylab.xlabel("stage"), pylab.ylabel("MSE")
        pylab.ylim(ymin=0)
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        return self

    def mse_curves2(self, uniform_variables, target_efficiencies=None, power=2., n_bins=20):
        warn('deprecated', DeprecationWarning)
        plotStagedMseCurves(self.staged_predictions, testX=self.X, testY=self.y, uniform_variables=uniform_variables,
                            target_efficiencies=target_efficiencies, power=power, n_bins=n_bins)
        return self

    def efficiency(self, uniform_variables, stages=None, target_efficiencies=None, n_bins=20, on_signal=True):
        target_efficiencies = self._check_efficiencies(target_efficiencies)
        if len(uniform_variables) not in {1, 2}:
            raise ValueError("More than two variables are not implemented, you got a 3d-monitor?")

        mask = self.is_signal if on_signal else ~self.is_signal
        bin_indices = self._compute_bin_indices(uniform_variables, n_bins, mask=mask)
        def computeBinEfficiencies(prediction_proba, target_eff):
            cut = computeBDTCut(target_eff, self.y, prediction_proba)
            return computeLocalEfficienciesOfBins(prediction_proba, self.y, bin_indices,
                                                  n_total_bins=n_bins ** len(uniform_variables), cut=cut)

        if len(uniform_variables) == 1:
            effs = self._map_on_stages(stages=stages,
                    function=lambda pred: [computeBinEfficiencies(pred, eff) for eff in target_efficiencies])
            x_limits, = self._compute_bin_centers(uniform_variables, n_bins=n_bins, mask=mask)

            effs = pandas.DataFrame(effs)
            for stage_name, stage in effs.iterrows():
                self._strip_figure(len(stage))
                for i, (name, eff_stage_data) in enumerate(stage.iteritems()):
                    if isinstance(eff_stage_data, float) and pandas.isnull(eff_stage_data):
                        continue
                    pylab.subplot(1, len(stage), i + 1)
                    for eff, local_effs in zip(target_efficiencies, eff_stage_data):
                        pylab.plot(x_limits, local_effs, label='eff=%.2f' % eff)
                        pylab.title(name)
                        pylab.xlabel(uniform_variables[0]), pylab.ylabel('efficiency')
        else:
            x_limits, y_limits = self._compute_bin_centers(uniform_variables, n_bins=n_bins, mask=mask)
            for target_efficiency in target_efficiencies:
                staged_results = self._map_on_stages(lambda x: computeBinEfficiencies(x, target_efficiency),
                                                     stages=stages)
                staged_results = pandas.DataFrame(staged_results)
                for stage_name, stage_data in staged_results.iterrows():
                    print("Stage %s, efficiency=%.2f" % (str(stage_name), target_efficiency))
                    self._strip_figure(len(stage_data))
                    for i, (name, local_efficiencies) in enumerate(stage_data.iteritems()):
                        if isinstance(local_efficiencies, float) and pandas.isnull(local_efficiencies):
                            continue
                        local_efficiencies = local_efficiencies.reshape((n_bins, n_bins))
                        # drawing difference
                        local_efficiencies[local_efficiencies < 0] = target_efficiency
                        local_efficiencies -= target_efficiency
                        ax = pylab.subplot(1, len(stage_data), i + 1)
                        p = ax.pcolor(x_limits, y_limits, local_efficiencies, cmap=cm.get_cmap("RdBu"),
                                      vmin=-0.2, vmax=+0.2)
                        ax.set_xlabel(uniform_variables[0]), ax.set_ylabel(uniform_variables[1])
                        ax.set_title(name)
                        pylab.colorbar(p, ax=ax)
                    pylab.show()
        return self

    def correlation(self, var_name, stages=None, metrics=Efficiency, n_bins=20, thresholds=None, **kwargs):
        for stage, preds in pandas.DataFrame(self._get_stages(stages=stages)).iterrows():
            self._strip_figure(len(preds))
            print('stage ' + str(stage))
            for i, (name, predictions) in enumerate(preds.iteritems()):
                pylab.subplot(1, len(preds), i + 1)
                plotScoreVariableCorrelation(predictions, self.y, numpy.ravel(self.X[var_name]), classifier_name=name,
                                             var_name=var_name, score_function=metrics, bins_number=n_bins,
                                             thresholds=thresholds, ** kwargs)
        return self

    def compute_metrics(self, stages=None, metrics=roc_auc_score, in_html=True):
        print("Computing " + metrics.__name__)
        result = pandas.DataFrame(self._map_on_stages(lambda preds: metrics(self.y, preds[:, 1]), stages=stages))
        if in_html:
            from IPython.display import display_html
            display_html(result)
        else:
            print(result)
        return self

    def hist(self, var_names):
        plotClassesDistribution(self.X, self.y, var_names)
        return self

    @staticmethod
    def _strip_figure(n):
        pylab.figure(figsize=(12 + 3 * n, 10 - n))

    def show(self):
        pylab.show()
        return self


# the same in old good functions (will be removed soon)
def trainClassifiers(classifiers_dict, trainX, trainY, ipc_profile=None):
    """Trains all classifiers on the same train data,
    if ipc_profile in not None, it is used as a name of ipython cluster to use for parallel computations,
    if block=False, nonblocking mode is used and then() method can be used"""
    if ipc_profile is None:
        for name, classifier in classifiers_dict.iteritems():
            start_time = time.time()
            classifier.fit(trainX, trainY)
            print "Classifier %12s is learnt in %0.2f seconds" % (name, time.time() - start_time)
    else:
        from IPython.parallel import Client
        client = Client(profile=ipc_profile)
        start_time = time.time()
        lb_view = client.load_balanced_view()
        def trainClassifier(name_classifier, X, y):
            name_classifier[1].fit(X, y)
            return name_classifier

        result = lb_view.map_sync(trainClassifier, classifiers_dict.iteritems(),
                                [trainX] * len(classifiers_dict),  [trainY] * len(classifiers_dict))
        print "We spent %.3f seconds on parallel training" % (time.time() - start_time)
        for name, classifier in result:
            classifiers_dict[name] = classifier


def getClassifiersPredictionProba(classifiers_dict, testX):
    return OrderedDict([(name, classifier.predict_proba(testX))
                        for name, classifier in classifiers_dict.iteritems()])

def getClassifiersStagedPredictionProba(classifiers_dict, testX):
    """Returns dictionary: {classifier_name: staged_predict_proba of classifier} """
    result = OrderedDict()
    for name, classifier in classifiers_dict.iteritems():
        try:
            result[name] = list(classifier.staged_predict_proba(testX))
        except AttributeError:
            print "Classifier %12s doesn't provide staged_predict_proba" % name
    return result


def getStageOfStagedProbaDict(staged_predict_proba_dict, stage):
    """Returns the predict_proba_dict, corresponding to 'stage' iteration of every classifier"""
    return OrderedDict([(name, predictions[stage]) for name, predictions in staged_predict_proba_dict.iteritems()])


def plotScoreVariableCorrelation(prediction_proba, answers, correlation_values, classifier_name="", var_name="",
                                 score_function=Efficiency, bins_number=20, thresholds=None, show_legend=False):
    """Different score functions available: Efficiency, Precision, Recall, F1Score,
    and other things from sklearn.metrics
    var_name - for example, 'mass', just a name for plotting. """
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

    if show_legend:
        pylab.legend(loc="lower right")


def plotMassEfficiencyCorrelation(prediction_proba, answers, masses, classifier_name):
    """
    Just a particular case of previous function
    Splits all the events by mass into 20 bins of equal size,
    computes efficiency for each bin and draws a plot
    - answers - array of 0 and 1
    - predictionProbabilities - array of probabilities given by classifier
    - masses - array of masses
    """
    plotScoreVariableCorrelation(prediction_proba, answers, masses, classifier_name, var_name='mass',
                                 score_function=Efficiency)


def plotLearningCurves(staged_proba_dict, answers, step=1, metrics=roc_auc_score):
    """Plots learning curves of several classifiers,
    'metrics' is evaluated after each 'step' iterations"""
    for classifier_name, staged_probas in staged_proba_dict.iteritems():
        rocs, stages = [], []
        for stage, prediction_proba in enumerate(staged_probas):
            if (stage + 1) % step != 0:
                continue
            rocs.append(metrics(answers, prediction_proba[:, 1]))
            stages.append(stage)
            # roc = [metrics(answers, prediction_proba[:, 1]) for prediction_proba in staged_probas[::step]]
        pylab.plot(stages, rocs, label=classifier_name)
    pylab.legend(loc="lower right")
    pylab.xlabel("stage")
    pylab.ylabel("ROC AUC")


def plotRocCurves(predict_proba_dict, answers, is_big_plot=True):
    """TestAnswer in numpy.array with zeros and ones
    testPredictions is dictionary:
    - key is string (classifier name usually)
    - value is numpy.array with probabilities of class 1
    """
    if is_big_plot:
        pylab.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
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


def compute1DMseVariation(prediction_proba, answers, mass, binner, target_efficiencies=None):
    """Computes 'non-flatness' of predictions, the lesser output, the better
    :param answers: numpy.array
    :param prediction_proba: numpy.ndarray
    :param mass: numpy.array
    :param binner: Binner
    """
    if target_efficiencies is None:
        target_efficiencies = [0.5, 0.6, 0.7, 0.8, 0.9]

    bin_indices = binner.get_bins(mass)
    return computeMseVariationOnBins(prediction_proba, answers > 0.5, bin_indices=bin_indices,
                                     target_efficiencies=target_efficiencies)


def computeBinIndices(X, var_names, bin_limits):
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

def computeLocalEfficienciesOfBins(prediction_proba, answers, bin_indices, n_total_bins, cut):
    assert len(answers) == len(prediction_proba) == len(bin_indices), "different size"
    is_signal = answers > 0.5
    bin_total = numpy.bincount(bin_indices[is_signal], minlength=n_total_bins) + 1e-6

    passed_cut = prediction_proba[:, 1] > cut
    bin_passed_cut = numpy.bincount(bin_indices[is_signal & passed_cut], minlength=n_total_bins) - 1e-10
    return bin_passed_cut / bin_total

def computeMseVariationOnBins(prediction_proba, is_signal, bin_indices, target_efficiencies, power=2.):
    """ An efficient function to compute MSE, the splitting into bins should be given in bin_indices """
    assert len(prediction_proba) == len(bin_indices) == len(is_signal), "different size"
    n_bins = numpy.max(bin_indices) + 1
    bin_total = numpy.bincount(bin_indices[is_signal], minlength=n_bins) + 1e-6
    signal_prediction_proba = prediction_proba[is_signal, :]
    signal_answers = numpy.ones(len(signal_prediction_proba), dtype=numpy.int)
    result = 0.
    cuts = computeBDTCut(numpy.array(target_efficiencies), signal_answers, signal_prediction_proba)
    for cut, efficiency in zip(cuts, target_efficiencies):
        passed_cut = signal_prediction_proba[:, 1] > cut
        mean_efficiency = numpy.mean(passed_cut)
        bin_passed_cut = numpy.bincount(bin_indices[is_signal][passed_cut], minlength=n_bins)
        bin_efficiency = bin_passed_cut / bin_total
        result += numpy.sum(bin_total * numpy.abs(bin_efficiency - mean_efficiency) ** power)
    # Minkowski distance trick
    return 10 * (result / len(target_efficiencies) / numpy.sum(is_signal)) ** (1. / power)


def computeMseVariationOnGroups(prediction_proba, is_signal, groups, target_efficiencies, power=2.):
    """ An efficient function to compute MSE, the splitting into groups should be given
     in the format of list, each item is a list of indices inside bin"""
    assert len(prediction_proba) == len(is_signal), "different size"

    cuts = computeBDTCut(numpy.array(target_efficiencies), is_signal, prediction_proba)

    efficiencies = [list() for eff in target_efficiencies]
    groups_sizes = numpy.array([len(x) for x in groups])

    for group_indices in groups:
        if len(group_indices) == 0:
            continue
        assert numpy.all(is_signal[group_indices]), "the provided groups contain bg events"
        group_predictions = numpy.take(prediction_proba[:, 1], group_indices)
        for i, (eff, cut) in enumerate(zip(efficiencies, cuts)):
            efficiencies[i].append(numpy.sum(group_predictions > cut) / float(len(group_indices)))

    result = 0.
    for cut, efficiencies_at_cut in zip(cuts, efficiencies):
        mean_efficiency = numpy.mean(prediction_proba[is_signal, 1] > cut)
        result += numpy.sum(groups_sizes * numpy.abs(numpy.array(efficiencies_at_cut) - mean_efficiency) ** power)

    # Minkowski distance trick
    return 10 * (result / len(target_efficiencies) / numpy.sum(groups_sizes)) ** (1. / power)


def testComputeMseVariation(size=1000, n_bins=10):
    # testX = pandas.DataFrame(numpy.random.random(size=(size, 1)))
    testY = numpy.random.random(size) > 0.5
    preds = numpy.random.random((size, 2))

    bins = numpy.random.randint(0, n_bins, size)
    target_efficiencies = [0.5, 0.6]
    groups = [numpy.where(numpy.logical_and(testY, bins == bin))[0] for bin in range(n_bins)]
    x1 = computeMseVariationOnBins(preds, testY, bin_indices=bins, target_efficiencies=target_efficiencies)
    x2 = computeMseVariationOnGroups(preds, testY, groups=groups, target_efficiencies=target_efficiencies)
    assert abs(x1 - x2) < 1e-6, "MSE are different"
    print "MSE variation is ok"

testComputeMseVariation()


def computeMseVariation(predict_proba, answers, testX, var_names, efficiencies, n_bins=30, bin_limits=None):
    if bin_limits is None:
        bin_limits = []
        for var_name in var_names:
            var_data = testX[var_name]
            bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])
    bin_indices = computeBinIndices(testX, var_names, bin_limits)
    return computeMseVariationOnBins(predict_proba, answers > 0.5, bin_indices, efficiencies)


def computeStagedMseVariation(staged_predict_proba, answers, testX, var_names, stages, target_efficiencies, n_bins=30,
                              power=2.):
    is_signal = answers > 0.5

    bin_limits = []
    for var_name in var_names:
        var_data = testX[var_name]
        bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])

    bin_indices = computeBinIndices(testX, var_names, bin_limits)

    results = []
    for stage in stages:
        predict_probas = getStageOfStagedProbaDict(staged_predict_proba, stage)
        stage_variations = []
        for name, predict_proba in predict_probas.iteritems():
            stage_variations.append(computeMseVariationOnBins(predict_proba, is_signal, bin_indices=bin_indices,
                                                              target_efficiencies=target_efficiencies, power=power))
        results.append(stage_variations)
    return pandas.DataFrame(results, columns=staged_predict_proba.keys(), index=stages)




def plotEfficiency2D(probas_dict, var_names, testX, testY, target_efficiency, n_bins=30, xlim=None, ylim=None,
                     draw_difference=True):
    """This function plots the efficiency on 2D plot
    - var_name1 is name of first variable
    - var_name2 is name of second variable
    - target_efficiency: float between zero and one,
        the global cut is chosen to give this efficiency
    - order is list of strings, names of classifiers to compare
    - xlim - tuple (x_min, x_max) or None, just as for plot
    - ylim - tuple (y_min, y_max) or None
    """
    assert len(var_names) == 2
    var_name1, var_name2 = var_names
    if xlim is None:
        xlim = numpy.min(testX[var_name1]), numpy.max(testX[var_name1])
    if ylim is None:
        ylim = numpy.min(testX[var_name2]), numpy.max(testX[var_name2])

    x_limits = numpy.linspace(xlim[0], xlim[1], (n_bins + 1))
    y_limits = numpy.linspace(ylim[0], ylim[1], (n_bins + 1))

    fig = pylab.figure(figsize=(5 + 5 * len(probas_dict), 7))
    for i, name in enumerate(probas_dict):
        predict_proba = probas_dict[name]
        cut = computeBDTCut(target_efficiency, testY, predict_proba)
        bin_indices = computeBinIndices(testX, [var_name1, var_name2], [x_limits[1:-1], y_limits[1:-1]])

        local_efficiencies = computeLocalEfficienciesOfBins(predict_proba, testY, bin_indices=bin_indices,
                                                            n_total_bins=n_bins ** 2, cut=cut).reshape((n_bins, n_bins))
        if draw_difference:
            local_efficiencies[local_efficiencies < 0] = target_efficiency
            local_efficiencies -= target_efficiency
        ax = fig.add_subplot(1, len(probas_dict), i + 1)
        if draw_difference:
            p = ax.pcolor(x_limits, y_limits, local_efficiencies, cmap=cm.get_cmap("RdBu"), vmin=-0.2, vmax=+0.2)
        else:
            p = ax.pcolor(x_limits, y_limits, local_efficiencies, cmap=cm.get_cmap("Blue"), vmin=0.0, vmax=1.0)
        ax.set_xlabel(var_name1), ax.set_ylabel(var_name2)
        ax.set_title(name)
        fig.colorbar(p, ax=ax)


def plotStagedMseCurves(classifiers, testX, testY, uniform_variables,
                        target_efficiencies=None, step=1, power=2., n_bins=15):
    if target_efficiencies is None:
        target_efficiencies = [0.6, 0.7, 0.8, 0.9]
    if isinstance(classifiers[list(classifiers.keys())[0]], list):
        staged_predict_proba_dict = classifiers
    else:
        staged_predict_proba_dict = getClassifiersStagedPredictionProba(classifiers, testX)
    stages = range(min([len(pred) for name, pred in staged_predict_proba_dict.iteritems()]))[::step]
    mse_df = computeStagedMseVariation(staged_predict_proba_dict, testY, testX, uniform_variables, stages,
                                       target_efficiencies=target_efficiencies, n_bins=n_bins, power=power)
    for column in mse_df.columns:
        pylab.plot(stages, mse_df[column], label=column)
    pylab.ylim(ymin=0)
    pylab.ylabel("MSE"), pylab.xlabel("stage")
    pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)


def plotClassesDistribution(X, y, var_names):
        y = numpy.array(y)
        classes = numpy.unique(y)
        if len(var_names) == 1:
            pylab.figure(figsize=(14, 7))
            pylab.title('Distribution of classes')
            for y_val in classes:
                pylab.hist(numpy.ravel(X.ix[y == y_val, var_names]), label='class=%i' % y_val, histtype='step')
                pylab.xlabel(var_names[0])
        elif len(var_names) == 2:
            pylab.figure(figsize=(12, 10))
            x_var, y_var = var_names
            xmin = numpy.min(X[x_var])
            xmax = numpy.max(X[x_var])
            ymin = numpy.min(X[y_var])
            ymax = numpy.max(X[y_var])

            pylab.title('Distribution of classes')
            pylab.xlim(xmin, xmax), pylab.ylim(ymin, ymax)
            for i, y_val in enumerate(classes):
                alpha = numpy.clip(2000. / numpy.sum(y == y_val), 0.02, 1)
                pylab.plot(X.ix[y == y_val, x_var], X.ix[y == y_val, y_var], '.', alpha=alpha, label='class=' + str(y_val))
        else:
            raise ValueError("More than tow variables are not implemented")


def testComputeMseAndBins(size=500):
    columns = ['var1', 'var2']
    X = pandas.DataFrame(numpy.random.random((size, 2)), columns=columns)
    y = numpy.random.random(size) > 0.5
    proba = numpy.random.random((size, 2))
    computeMseVariation(proba, y, X, columns, [0.3, 0.5, 0.7])
    n_bins = 5
    x_limits = numpy.linspace(0, 1, n_bins + 1)[1:-1]
    bins = computeBinIndices(X, columns, [x_limits, x_limits])
    assert numpy.all(0 <= bins) and numpy.all(bins < n_bins * n_bins), "whooops"

    effs = computeLocalEfficienciesOfBins(proba, y, bins, n_bins * n_bins, .3)



def testAll():
    from commonutils import generateSample
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    trainX, trainY = generateSample(1000, 10)
    testX, testY = generateSample(1000, 10)

    for low_memory in [True, False]:
        classifiers = ClassifiersDict()
        classifiers['ada'] = AdaBoostClassifier(n_estimators=20)
        classifiers['forest'] = RandomForestClassifier(n_estimators=20)

        classifiers.fit(trainX, trainY).test_on(testX, testY, low_memory=low_memory)\
            .efficiency(trainX.columns[:1], n_bins=7).show() \
            .efficiency(trainX.columns[:2], n_bins=12).show() \
            .roc(stages=[10, 15]).show() \
            .learning_curves().show() \
            .mse_curves(['column0']).show() \
            .hist(['column0']).show()\
            .roc().show().print_mse(['column0'], in_html=False)\
            .correlation(['column0']).show() \
            .compute_metrics(stages=[5, 10], metrics=roc_auc_score, in_html=False) \

if __name__ == "__main__":
    testAll()


testComputeMseAndBins()