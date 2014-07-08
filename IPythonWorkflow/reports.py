# About

# This module contains functions to build reports:
# training, getting predictions,
# building various plots, calculating metrics

from __future__ import print_function
from __future__ import division
from itertools import islice
from numpy.random.mtrand import RandomState

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import time
import numpy
import pandas
import pylab
from sklearn.metrics import auc
from sklearn.utils.validation import check_arrays, column_or_1d
from matplotlib import cm
from scipy.stats import pearsonr

from commonutils import compute_bdt_cut, Binner, roc_curve, roc_auc_score, check_sample_weight, build_normalizer, \
    compute_cut_for_efficiency


__author__ = 'Alex Rogozhnikov'


# Score functions
# Some notation used here
# IsSignal - is really signal
# AsSignal - classified as signal
# IsBackgroundAsSignal - background, but classified as signal
# ... and so on. Cute, right?

def efficiency_score(y_true, y_pred, sample_weight=None):
    """Efficiency = right classified signal / everything that is really signal
    Efficiency == recall, returns -0.1 when ill-defined"""
    sample_weight = check_sample_weight(y_true, sample_weight=sample_weight)
    assert len(y_true) == len(y_pred), "Different size of arrays"
    isSignal = numpy.sum(y_true * sample_weight) - 1e-6
    isSignalAsSignal = numpy.sum(y_true * y_pred * sample_weight) + 1e-7
    return isSignalAsSignal / isSignal
    # the same, but with notifications
    # return recall_score(answer, prediction)


def background_efficiency_score(y_true, y_pred, sample_weight=None):
    """BackgroundEfficiency == right classified bg / everything that is really bg == fpr"""
    return efficiency_score(1 - y_true, 1 - y_pred, sample_weight=sample_weight)


def as_signal_score(y_true, y_pred, sample_weight=None):
    """Part of is signal = classified as signal / total amount of events"""
    sample_weight = check_sample_weight(y_true, sample_weight)
    assert len(y_true) == len(y_pred), "Different size of arrays"
    return numpy.sum(y_pred * sample_weight) / numpy.sum(sample_weight)


def train_classifier(name_classifier, X, y, sample_weight=None):
    """ Trains one classifier on a separate node,
    :param name_classifier: 2-tuple (name, classifiers)
    """
    if sample_weight is None:
        name_classifier[1].fit(X, y)
    else:
        name_classifier[1].fit(X, y, sample_weight=sample_weight)
    return name_classifier


class ClassifiersDict(OrderedDict):
    """This class is a collection of classifiers, which will be trained simultaneously
    and will be
    """
    def fit(self, X, y, sample_weight=None, ipc_profile=None):
        """Trains all classifiers on the same train data,
        if ipc_profile in not None, it is used as a name of IPython cluster to use for parallel computations"""
        if ipc_profile is None:
            for name, classifier in self.iteritems():
                start_time = time.time()
                if sample_weight is None:
                    classifier.fit(X, y)
                else:
                    classifier.fit(X, y, sample_weight=sample_weight)
                print("Classifier %12s is learnt in %.2f seconds" % (name, time.time() - start_time))
        else:
            from IPython.parallel import Client
            client = Client(profile=ipc_profile)
            start_time = time.time()
            lb_view = client.load_balanced_view()

            result = lb_view.map_sync(train_classifier, self.iteritems(), [X] * len(self),  [y] * len(self),
                                      sample_weight=[sample_weight] * len(self))
            print("We spent %.2f seconds on parallel training" % (time.time() - start_time))
            for name, classifier in result:
                self[name] = classifier
        return self

    def test_on(self, X, y, sample_weight=None, low_memory=True):
        return Predictions(self, X, y, sample_weight=sample_weight, low_memory=low_memory)


class Predictions(object):
    def __init__(self, classifiers_dict, X, y, sample_weight=None, low_memory=True):
        """The main object for different reports and plots, computes predictions of different classifiers
        on the sane test data sets and makes it possible to compute different metrics,
        plot some quality curves and so on
        """
        assert isinstance(classifiers_dict, OrderedDict)
        self.X = X
        self.y = numpy.array(y, dtype=int)
        self.sample_weight = sample_weight
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
                    self.staged_predictions[name] = list([numpy.copy(x) for x in classifier.staged_predict_proba(X)])
                    self.predictions[name] = self.staged_predictions[name][-1]
                except AttributeError:
                    self.predictions[name] = classifier.predict_proba(X)

    @staticmethod
    def _check_efficiencies(efficiencies):
        if efficiencies is None:
            return numpy.array([0.6, 0.7, 0.8, 0.9])
        else:
            return numpy.array(efficiencies)

    def _check_mask(self, mask):
        """Checks whether the mask is appropriate and normalizes it"""
        if mask is None:
            return numpy.ones(len(self.y), dtype=numpy.bool)
        assert len(mask) == len(self.y), 'wrong size of mask'
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
                    pass
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
        returns: {name: Series[stage_name, result]}
        :param function: should take the only argument, predict_proba of shape [n_samples, 2]
        :param step: int, the function is applied to every step'th iteration
        """
        result = OrderedDict()
        for name, staged_proba in self._get_staged_proba().iteritems():
            result[name] = pandas.Series()
            for stage, pred in islice(enumerate(staged_proba), step - 1, None, step):
                result[name].loc[stage] = function(pred)
        return result

    def _map_on_stages(self, function, stages=None):
        """
        :type function: takes prediction proba of shape [n_samples, n_classes] and returns something
        :type stages: list(int) | NoneType, the list of stages we calculate metrics on
        :rtype: pandas.DataFrame, with calculated results"""
        # TODO rewrite without get_stages
        selected_stages = self._get_stages(stages)
        result = OrderedDict()
        for name, staged_proba in selected_stages.iteritems():
            result[name] = staged_proba.apply(function)
        return result

    def _plot_on_stages(self, plotting_function, stages=None):
        """Plots in each line results for the same stage,
        plotting_function should have following interface:
        plotting_function(y_true, y_proba, sample_weight), y_proba has shape [n_samples, n_features] """
        selected_stages = pandas.DataFrame(self._get_stages(stages))
        for stage_name, stage_predictions in selected_stages.iterrows():
            print('Stage ' + str(stage_name))
            self._strip_figure(len(stage_predictions))
            for i, (name, probabilities) in enumerate(stage_predictions.iteritems()):
                pylab.subplot(1, len(stage_predictions), i + 1)
                pylab.title(name)
                plotting_function(self.y, probabilities, sample_weight=self.sample_weight)
            pylab.show()

    def _compute_bin_indices(self, var_names, n_bins=20, mask=None):
        """Mask is used to show events that will be binned after"""
        for var in var_names:
            assert var in self.X.columns, "the variable %i is not in dataset" % var
        mask = self._check_mask(mask)
        bin_limits = []
        for var_name in var_names:
            var_data = self.X.loc[mask, var_name]
            bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])
        return compute_bin_indices(self.X, var_names, bin_limits)

    def _compute_bin_centers(self, var_names, n_bins=20, mask=None):
        """Mask is used to show events that will be binned after"""
        bin_centers = []
        mask = self._check_mask(mask)
        for var_name in var_names:
            var_data = self.X.loc[mask, var_name]
            bin_centers.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), 2 * n_bins + 1)[1::2])
            assert len(bin_centers[-1]) == n_bins
        return bin_centers

    def _compute_staged_mse(self, var_names, target_efficiencies=None, step=3, n_bins=20, power=2., label=1):
        target_efficiencies = self._check_efficiencies(target_efficiencies)
        mask = self.y == label
        bin_indices = self._compute_bin_indices(var_names, n_bins=n_bins, mask=mask)
        compute_mse = lambda pred: \
            compute_msee_on_bins(pred[:, label], mask, bin_indices, target_efficiencies=target_efficiencies,
                                 power=power, sample_weight=self.sample_weight)
        return self._map_on_staged_proba(compute_mse, step)

    def _compute_mse(self, var_names, target_efficiencies=None, stages=None, n_bins=20, power=2., label=1):
        target_efficiencies = self._check_efficiencies(target_efficiencies)
        mask = self.y == label
        bin_indices = self._compute_bin_indices(var_names, n_bins=n_bins, mask=mask)
        compute_mse = lambda pred: \
            compute_msee_on_bins(pred[:, label], mask, bin_indices, target_efficiencies=target_efficiencies,
                                 power=power, sample_weight=self.sample_weight)
        return self._map_on_stages(compute_mse, stages=stages)

    def print_mse(self, uniform_variables, efficiencies=None, stages=None, in_html=True, label=1):
        result = pandas.DataFrame(self._compute_mse(uniform_variables, efficiencies, stages=stages, label=label))
        if in_html:
            from IPython.display import display_html
            display_html("<b>Staged MSE variation</b>", raw=True)
            display_html(result)
        else:
            print("Staged MSE variation")
            print(result)
        return self

    def mse_curves(self, uniform_variables, target_efficiencies=None, n_bins=20, step=3, power=2., label=1):
        result = self._compute_staged_mse(uniform_variables, target_efficiencies, step=step,
                                          n_bins=n_bins, power=power, label=label)
        for name, mse_stages in result.iteritems():
            pylab.plot(mse_stages.keys(), mse_stages, label=name)
            pylab.xlabel("stage"), pylab.ylabel("MSE")
        pylab.ylim(ymin=0)
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        return self

    def efficiency(self, uniform_variables, stages=None, target_efficiencies=None, n_bins=20, label=1):
        target_efficiencies = self._check_efficiencies(target_efficiencies)
        if len(uniform_variables) not in {1, 2}:
            raise ValueError("More than two variables are not implemented, you have a 3d-monitor?")

        mask = self.y == label
        bin_indices = self._compute_bin_indices(uniform_variables, n_bins, mask=mask)

        def compute_bin_efficiencies(prediction_proba, target_eff):
            cut = compute_bdt_cut(target_eff, mask, prediction_proba[:, label])
            return compute_efficiencies_on_bins(prediction_proba[:, label], mask, bin_indices,
                                                n_total_bins=n_bins ** len(uniform_variables), cut=cut)

        if len(uniform_variables) == 1:
            effs = self._map_on_stages(stages=stages,
                    function=lambda pred: [compute_bin_efficiencies(pred, eff) for eff in target_efficiencies])
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
                        pylab.ylim(ymin=0.)
        else:
            x_limits, y_limits = self._compute_bin_centers(uniform_variables, n_bins=n_bins, mask=mask)
            for target_efficiency in target_efficiencies:
                staged_results = self._map_on_stages(lambda x: compute_bin_efficiencies(x, target_efficiency),
                                                     stages=stages)
                staged_results = pandas.DataFrame(staged_results)
                for stage_name, stage_data in staged_results.iterrows():
                    print("Stage %s, efficiency=%.2f" % (str(stage_name), target_efficiency))
                    self._strip_figure(len(stage_data))
                    for i, (name, local_efficiencies) in enumerate(stage_data.iteritems()):
                        if isinstance(local_efficiencies, float) and pandas.isnull(local_efficiencies):
                            continue
                        local_efficiencies = local_efficiencies.reshape((n_bins, n_bins)).transpose()
                        # drawing difference, the efficiency in empty bins will be replaced with mean value
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

    def roc(self, stages=None):
        proba_on_stages = pandas.DataFrame(self._get_stages(stages))
        n_stages = len(proba_on_stages)
        self._strip_figure(n_stages)
        for i, (stage_name, proba_on_stage) in enumerate(proba_on_stages.iterrows()):
            pylab.subplot(1, n_stages, i + 1), pylab.title("stage " + str(stage_name))
            plot_roc_curves(predict_proba_dict=proba_on_stage, y_true=self.y, sample_weight=self.sample_weight)
            pylab.title('ROC at stage ' + str(stage_name))
        return self

    def learning_curves(self, metrics=roc_auc_score, step=1, label=1):
        y_true = (self.y == label) * 1
        function = lambda predictions: metrics(y_true, predictions[:, label], sample_weight=self.sample_weight)
        result = self._map_on_staged_proba(function=function, step=step)

        for classifier_name, staged_roc in result.iteritems():
            pylab.plot(staged_roc.keys(), staged_roc, label=classifier_name)
        pylab.legend(loc="lower right")
        pylab.xlabel("stage"), pylab.ylabel("ROC AUC")
        return self

    def correlation(self, var_name, stages=None, metrics=efficiency_score, n_bins=20, thresholds=None):
        """ Plots the dependence of efficiency / sensitivity / whatever vs one of the variables
        :type var_name: str, the name of variable
        :type stages: list(int) | NoneType
        :type metrics: function
        :type n_bins: int, the number of bins
        :type thresholds: list(float) | NoneType
        :rtype: Predictions, returns self
        """
        thresholds = [0.2, 0.4, 0.5, 0.6, 0.8] if thresholds is None else thresholds
        for stage, predictions in pandas.DataFrame(self._get_stages(stages=stages)).iterrows():
            self._strip_figure(len(predictions))
            print('stage ' + str(stage))
            for i, (name, proba) in enumerate(predictions.iteritems()):
                pylab.subplot(1, len(predictions), i + 1)
                plot_score_variable_correlation(self.y, proba, numpy.ravel(self.X[var_name]), cuts=thresholds,
                                                classifier_name=name, var_name=var_name, score_function=metrics,
                                                bins_number=n_bins)
        return self

    def correlation_curves(self, var_name, center=None, step=1, label=1):
        """ Correlation is built only for signal (by now, will be extended soon)
        :param var_name: str, correlation is computed for this variable
        :param center: float|None, if float, the correlation is measured between |x - center| and prediction
        :param step: int
        :param label: int, label of class, the correlation is computed for the events of this class
        :return: Predictions, returns self
        """
        mask = self.y == label
        pylab.title("Pearson correlation with " + str(var_name))
        data = self.X.loc[mask, var_name]
        if center is not None:
            data = numpy.abs(data - center)

        weight = check_sample_weight(self.y, self.sample_weight)[mask]

        def compute_correlation(prediction_proba):
            pred = prediction_proba[mask, label]
            pred = build_normalizer(pred, sample_weight=weight)(pred)
            return pearsonr(pred, data)[0]
        correlations = self._map_on_staged_proba(compute_correlation, step=step)

        for classifier_name, staged_correlation in correlations.iteritems():
            pylab.plot(staged_correlation.keys(), staged_correlation, label=classifier_name)
        pylab.legend(loc="lower left")
        pylab.xlabel("stage"), pylab.ylabel("Pearson correlation")
        return self

    def compute_metrics(self, stages=None, metrics=roc_auc_score, label=1):
        """ Computes arbitrary metrics on selected stages
        :param stages: array-like of stages or None
        :param metrics: (numpy.array, numpy.array, numpy.array | None) -> float,
            any metrics with interface (y_true, y_pred, sample_weight=None), where y_pred of shape [n_samples] of float
        :return: pandas.DataFrame with computed values
        """
        def compute_metrics(proba):
            return metrics((self.y == label) * 1, proba[:, label], sample_weight=self.sample_weight)
        return pandas.DataFrame(self._map_on_stages(compute_metrics, stages=stages))

    def hist(self, var_names):
        """ Plots 1 and 2-dimensional distributions
        :param var_names: array-like of length 1 or 2 with name of variables to plot
        :return: self """
        plot_classes_distribution(self.X, self.y, var_names)
        return self

    @staticmethod
    def _strip_figure(n):
        x_size = 12 if n == 1 else 12 + 3 * n
        y_size = 10 - n if n <= 5 else 4
        pylab.figure(figsize=(x_size, y_size))

    def show(self):
        pylab.show()
        return self


# the same in old good functions (will be removed soon)
def plot_score_variable_correlation(y_true, y_pred, correlation_values, cuts, sample_weight=None, classifier_name="",
                                    var_name="", score_function=efficiency_score, bins_number=20):
    """
    Different score functions available: Efficiency, Precision, Recall, F1Score, and other things from sklearn.metrics
    :param y_pred: numpy.array, of shape [n_samples]
    :param y_true: numpy.array, of shape [n_samples] with float predictions
    :param correlation_values: numpy.array of shape [n_samples], usually that is masses of events
    :param cuts: array-like of cuts, for each cut a separate
    :param sample_weight: numpy.array or None, shape = [n_samples]
    :param classifier_name: str, used only in label
    :param var_name: str, i.e. 'mass'
    :param score_function: any function with signature (y_true, y_pred, sample_weight=None)
    :param bins_number: int, the number of bins
    """
    y_true, y_pred, correlation_values = check_arrays(y_true, y_pred, correlation_values)
    sample_weight = check_sample_weight(y_true, sample_weight=sample_weight)

    binner = Binner(correlation_values, n_bins=bins_number)
    bins_data = binner.split_into_bins(correlation_values, y_true, y_pred, sample_weight)
    for cut in cuts:
        x_values = []
        y_values = []
        for bin_data in bins_data:
            bin_masses, bin_y_true, bin_proba, bin_weight = bin_data
            y_values.append(score_function(bin_y_true, bin_proba[:, 1] > cut, sample_weight=bin_weight))
            x_values.append(numpy.mean(bin_masses))
        pylab.plot(x_values, y_values, '.-', label="cut = %0.3f" % cut)

    pylab.title("Correlation with results of " + classifier_name)
    pylab.xlabel(var_name)
    pylab.ylabel(score_function.__name__)
    pylab.legend(loc="lower right")


def plot_roc(y_true, y_pred, sample_weight=None, classifier_name=""):
    """Plots ROC curve in the way physicist like it
    :param y_true: numpy.array, shape=[n_samples]
    :param y_pred: numpy.array, shape=[n_samples]
    :param sample_weight: numpy.array | None, shape = [n_samples]
    :param classifier_name: str, the name of classifier for label
    """
    y_true, y_pred = check_arrays(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, sample_weight=sample_weight)
    # tpr = recall = isSasS / isS = signal efficiency
    # fpr = isBasS / isB = 1 - specificity ?=?  1 - backgroundRejection
    bg_rejection = 1. - numpy.array(fpr)
    roc_auc = auc(fpr, tpr)
    pylab.plot(tpr, bg_rejection, label='%s (area = %0.3f)' % (classifier_name, roc_auc))


def plot_roc_curves(predict_proba_dict, y_true, sample_weight=None):
    """TestAnswer in numpy.array with zeros and ones
    :param predict_proba_dict: dictionary {name: predict_proba}, predict_proba is of shape [n_samples, 2]
    """
    for classifier_name, predictions in predict_proba_dict.iteritems():
        plot_roc(y_true, predictions[:, 1], sample_weight=sample_weight, classifier_name=classifier_name)
    pylab.plot([0, 1], [1, 0], 'k--')
    pylab.xlim([0., 1.003]),    pylab.xlabel('Signal Efficiency')
    pylab.ylim([0., 1.003]),    pylab.ylabel('Background Rejection')
    pylab.title('Receiver operating characteristic (ROC)')
    pylab.legend(loc="lower left")


def compute_bin_indices(X, var_names, bin_limits):
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


def bin_to_group_indices(bin_indices, mask):
    """ Transforms bin_indices into group indices
    :type bin_indices: numpy.array, each element in index of bin this event belongs, shape = [n_samples]
    :type mask: numpy.array, boolean mask of indices to split into bins, shape = [n_samples]
    :rtype: list(numpy.array), each element is indices of elements in some bin
    """
    assert len(bin_indices) == len(mask), "Different length"
    bins_id = numpy.unique(bin_indices)
    result = list()
    for bin_id in bins_id:
        result.append(numpy.where(mask & (bin_indices == bin_id))[0])
    return result


def test_bin_to_group_indices(size=100, bins=10):
    bin_indices = RandomState().randint(0, bins, size=size)
    mask = RandomState().randint(0, 1, size=size) > 0.5
    group_indices = bin_to_group_indices(bin_indices, mask=mask)
    assert numpy.sum([len(group) for group in group_indices]) == numpy.sum(mask)
    indices = set()
    for group in group_indices:
        assert numpy.all(mask[group])
        for index in group:
            assert index not in indices
        indices.update(group)

test_bin_to_group_indices()





def compute_efficiencies_on_bins(signal_proba, signal_mask, bin_indices, n_total_bins, cut, sample_weight=None):
    assert len(signal_mask) == len(signal_proba) == len(bin_indices), "different size"
    sample_weight = check_sample_weight(signal_mask, sample_weight=sample_weight)
    bin_total = numpy.bincount(bin_indices[signal_mask], weights=sample_weight[signal_mask], minlength=n_total_bins) + 1e-6
    signal_proba = column_or_1d(signal_proba)

    passed_cut = signal_proba > cut
    bin_passed_cut = numpy.bincount(bin_indices[signal_mask & passed_cut],
                                    weights=sample_weight[signal_mask & passed_cut], minlength=n_total_bins) - 1e-10
    return bin_passed_cut / bin_total


def compute_msee_on_bins(y_pred, mask, bin_indices, target_efficiencies, power=2., sample_weight=None):
    """ An efficient function to compute MSE, the splitting into bins should be given in bin_indices """
    assert len(y_pred) == len(bin_indices) == len(mask), "different size of arrays"
    # needed in case if in some bins there are no signal events
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    n_bins = numpy.max(bin_indices[mask]) + 1
    target_efficiencies = numpy.array(target_efficiencies)

    signal_proba = y_pred[mask]
    signal_answers = numpy.ones(len(signal_proba), dtype=numpy.int)
    signal_bins = bin_indices[mask]
    signal_weights = sample_weight[mask]

    bin_total = numpy.bincount(signal_bins, weights=signal_weights, minlength=n_bins) + 1e-6
    cuts = compute_cut_for_efficiency(target_efficiencies, signal_answers, y_pred=signal_proba,
                                      sample_weight=signal_weights)
    result = 0.
    for cut, efficiency in zip(cuts, target_efficiencies):
        passed_cut = signal_proba > cut
        mean_efficiency = numpy.average(passed_cut, weights=signal_weights)
        bin_passed_cut = numpy.bincount(signal_bins[passed_cut], weights=signal_weights[passed_cut], minlength=n_bins)
        bin_efficiency = bin_passed_cut / bin_total
        result += numpy.sum(bin_total * numpy.abs(bin_efficiency - mean_efficiency) ** power)
    # TODO probably we should norm on the weights
    # Minkowski distance trick with powers
    return 10 * (result / len(target_efficiencies) / numpy.sum(mask)) ** (1. / power)


def compute_msee_on_groups(y_pred, mask, groups, target_efficiencies, sample_weight=None, power=2.):
    """ An efficient function to compute MSE, the splitting into groups should be given
     in the format of list, each item is a list of indices inside bin"""
    assert len(y_pred) == len(mask), "different size"
    sample_weight = check_sample_weight(y_pred, sample_weight)
    y_pred = column_or_1d(y_pred)

    cuts = compute_cut_for_efficiency(target_efficiencies, mask, y_pred=y_pred, sample_weight=sample_weight)

    efficiencies = [list() for eff in target_efficiencies]
    groups_sizes = numpy.array([len(x) for x in groups])
    groups_weights = numpy.array([numpy.sum(numpy.take(sample_weight, g)) for g in groups])
    signal_weight = sample_weight[mask]

    for group_indices in groups:
        if len(group_indices) == 0:
            continue
        assert numpy.all(mask[group_indices]), "The provided groups contain bg events"
        group_predictions = numpy.take(y_pred, group_indices)
        group_weights = numpy.take(sample_weight, group_indices)

        for i, (eff, cut) in enumerate(zip(efficiencies, cuts)):
            efficiencies[i].append(numpy.average(group_predictions > cut, weights=group_weights))

    result = 0.
    for cut, efficiencies_at_cut in zip(cuts, efficiencies):
        mean_efficiency = numpy.average(y_pred[mask] > cut, weights=signal_weight)
        result += numpy.sum(groups_weights * numpy.abs(efficiencies_at_cut - mean_efficiency) ** power)

    # Minkowski distance trick with powers
    return 10 * (result / len(target_efficiencies) / numpy.sum(groups_sizes)) ** (1. / power)


def test_msee_computations(size=1000, n_bins=10):
    random = RandomState()
    testY = random.uniform(size=size) > 0.5
    pred = random.uniform(size=(size, 2))
    weights = random.exponential(size=size)

    bins = random.randint(0, n_bins, size)
    target_efficiencies = [0.5, 0.6]
    groups = [numpy.where(testY & (bins == bin))[0] for bin in range(n_bins)]
    x1 = compute_msee_on_bins(pred[:, 1], testY, bin_indices=bins,
                              target_efficiencies=target_efficiencies, sample_weight=weights)
    x2 = compute_msee_on_groups(pred[:, 1], testY, groups=groups,
                                target_efficiencies=target_efficiencies, sample_weight=weights)
    assert abs(x1 - x2) < 1e-6, "MSE are different"
    print("MSE variation is ok")

test_msee_computations()


def plot_classes_distribution(X, y, var_names):
        y = column_or_1d(y)
        labels = numpy.unique(y)
        if len(var_names) == 1:
            pylab.figure(figsize=(14, 7))
            pylab.title('Distribution of classes')
            for label in labels:
                pylab.hist(numpy.ravel(X.ix[y == label, var_names]), label='class=%i' % label, histtype='step')
                pylab.xlabel(var_names[0])

        elif len(var_names) == 2:
            pylab.figure(figsize=(12, 10))
            pylab.title('Distribution of classes')
            x_var, y_var = var_names
            for label in labels:
                alpha = numpy.clip(2000. / numpy.sum(y == label), 0.02, 1)
                pylab.plot(X.loc[y == label, x_var], X.loc[y == label, y_var], '.',
                           alpha=alpha, label='class=' + str(label))
        else:
            raise ValueError("More than tow variables are not implemented")


def test_bins(size=500):
    n_bins = 10
    columns = ['var1', 'var2']
    df = pandas.DataFrame(numpy.random.random((size, 2)), columns=columns)
    x_limits = numpy.linspace(0, 1, n_bins + 1)[1:-1]
    bins = compute_bin_indices(df, columns, [x_limits, x_limits])
    assert numpy.all(0 <= bins) and numpy.all(bins < n_bins * n_bins), "the bins with wrong indices appeared"

test_bins()


def test_all():
    from commonutils import generate_sample
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    trainX, trainY = generate_sample(1000, 10)
    testX, testY = generate_sample(1000, 10)

    for low_memory in [True]:
        classifiers = ClassifiersDict()
        classifiers['ada'] = AdaBoostClassifier(n_estimators=20)
        classifiers['forest'] = RandomForestClassifier(n_estimators=20)

        classifiers.fit(trainX, trainY).test_on(testX, testY, low_memory=low_memory)\
            .correlation(['column0']).show() \
            .correlation_curves('column1', ).show() \
            .learning_curves().show() \
            .efficiency(trainX.columns[:1], n_bins=7).show() \
            .efficiency(trainX.columns[:2], n_bins=12, target_efficiencies=[0.5]).show() \
            .roc(stages=[10, 15]).show() \
            .mse_curves(['column0']).show() \
            .hist(['column0']).show()\
            .roc().show().print_mse(['column0'], in_html=False)\
            .compute_metrics(stages=[5, 10], metrics=roc_auc_score)

if __name__ == "__main__":
    from matplotlib.cbook import Null
    pylab = Null()
    test_all()