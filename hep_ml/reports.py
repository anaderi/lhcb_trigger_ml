# About

# This module contains functions to build reports:
# training, getting predictions,
# building various plots, calculating metrics

from __future__ import print_function, division, absolute_import
from itertools import islice

from collections import OrderedDict
import time
import warnings
import numpy
import pandas
import matplotlib.pyplot as pylab
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.utils.validation import check_arrays, column_or_1d
from matplotlib import cm
from scipy.stats import pearsonr

from .commonutils import compute_bdt_cut, \
    check_sample_weight, build_normalizer, computeSignalKnnIndices, map_on_cluster

from .metrics_utils import compute_sde_on_bins, compute_sde_on_groups, compute_theil_on_bins, \
    bin_based_cvm, bin_based_ks

from .metrics_utils import compute_bin_efficiencies, compute_bin_weights, compute_bin_indices


__author__ = 'Alex Rogozhnikov'


def train_classifier(name_classifier, X, y, sample_weight=None):
    """ Trains one classifier on a separate node in cluster,
    :param name_classifier: 2-tuple (name, classifier)
    """
    start_time = time.time()
    if sample_weight is None:
        name_classifier[1].fit(X, y)
    else:
        name_classifier[1].fit(X, y, sample_weight=sample_weight)
    spent_time = time.time() - start_time
    return name_classifier, spent_time


class ClassifiersDict(OrderedDict):
    """A collection of classifiers, which will be trained simultaneously
    and after that will be compared"""

    def fit(self, X, y, sample_weight=None, ipc_profile=None):
        """Trains all classifiers on the same train data,
        if ipc_profile in not None, it is used as a name of IPython cluster to use for parallel computations"""
        start_time = time.time()
        result = map_on_cluster(ipc_profile, train_classifier,
                                self.items(),
                                [X] * len(self),
                                [y] * len(self),
                                [sample_weight] * len(self))
        total_train_time = time.time() - start_time
        for (name, classifier), clf_time in result:
            self[name] = classifier
            print("Classifier %12s is learnt in %.2f seconds" % (name, clf_time))

        if ipc_profile is None:
            print("Totally spent %.2f seconds on training" % total_train_time)
        else:
            print("Totally spent %.2f seconds on parallel training" % total_train_time)
        return self

    def test_on(self, X, y, sample_weight=None):
        return Predictions(self, X, y, sample_weight=sample_weight)


class Predictions(object):
    def __init__(self, classifiers_dict, X, y, sample_weight=None, low_memory=None):
        """The main object for different reports and plots,
        computes predictions of different classifiers on the same test data sets
        and makes it possible to compute different metrics,
        plot some quality curves and so on
        """
        assert isinstance(classifiers_dict, OrderedDict)
        if low_memory is not None:
            warnings.warn("Low memory argument is deprecated", DeprecationWarning)

        self.X = X
        self.y = column_or_1d(numpy.array(y, dtype=int))
        self.sample_weight = sample_weight
        assert len(X) == len(y), 'Different lengths'
        self.n_samples = len(y)
        self.checked_sample_weight = check_sample_weight(y, sample_weight=sample_weight)

        self.predictions = OrderedDict([(name, classifier.predict_proba(X))
                                        for name, classifier in classifiers_dict.items()])
        self.staged_predictions = None
        self.classifiers = classifiers_dict

    # region Checks
    @staticmethod
    def _check_efficiencies(efficiencies):
        if efficiencies is None:
            return numpy.array([0.6, 0.7, 0.8, 0.9])
        else:
            return numpy.array(efficiencies, dtype=numpy.float)

    def _check_mask(self, mask):
        """Checks whether the mask is appropriate and normalizes it"""
        if mask is None:
            return numpy.ones(len(self.y), dtype=numpy.bool)
        assert len(mask) == len(self.y), 'wrong size of mask'
        assert numpy.result_type(mask) == numpy.bool, 'the mask should be boolean'
        return mask

    # endregion

    # region Mappers - function that apply functions to predictions
    def _get_staged_proba(self):
        result = OrderedDict()
        for name, classifier in self.classifiers.items():
            try:
                result[name] = classifier.staged_predict_proba(self.X)
            except AttributeError:
                pass
        return result

    def _get_stages(self, stages):
        result = OrderedDict()
        if stages is None:
            for name, preds in self.predictions.items():
                result[name] = pandas.Series(data=[preds], index=['result'])
        else:
            stages = set(stages)
            for name, stage_preds in self._get_staged_proba().items():
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
        :param int step: the function is applied to every step'th iteration
        """
        result = OrderedDict()
        for name, staged_proba in self._get_staged_proba().items():
            result[name] = pandas.Series()
            for stage, pred in islice(enumerate(staged_proba), step - 1, None, step):
                result[name].loc[stage] = function(pred)
        return result

    def _map_on_stages(self, function, stages=None):
        """
        :type function: takes prediction proba of shape [n_samples, n_classes] and returns something
        :type stages: list(int) | NoneType, the list of stages we calculate metrics on
        :rtype: dict[str, pandas.Series]"""
        selected_stages = self._get_stages(stages)
        result = OrderedDict()
        for name, staged_proba in selected_stages.items():
            result[name] = staged_proba.apply(function)
        return result

    def _plot_on_stages(self, plotting_function, stages=None):
        """Plots in each line results for the same stage,
        plotting_function should have following interface:
        plotting_function(y_true, y_proba, sample_weight),  y_proba has shape [n_samples, n_features] """
        selected_stages = pandas.DataFrame(self._get_stages(stages))
        for stage_name, stage_predictions in selected_stages.iterrows():
            print('Stage ' + str(stage_name))
            self._strip_figure(len(stage_predictions))
            for i, (name, probabilities) in enumerate(stage_predictions.items(), start=1):
                pylab.subplot(1, len(stage_predictions), i)
                pylab.title(name)
                plotting_function(self.y, probabilities, sample_weight=self.sample_weight)
            pylab.show()

    def _plot_curves(self, function, step):
        """
        :param function: should take proba od shape [n_samples, n_classes]
        """
        result = self._map_on_staged_proba(function=function, step=step)
        for name, values in result.items():
            pylab.plot(values.keys(), values, label=name)
        pylab.xlabel('stage')
        return result

    # endregion

    # region Quality-related methods

    def roc(self, stages=None, new_figure=True):
        proba_on_stages = pandas.DataFrame(self._get_stages(stages))
        n_stages = len(proba_on_stages)
        if new_figure:
            self._strip_figure(n_stages)
        for i, (stage_name, proba_on_stage) in enumerate(proba_on_stages.iterrows()):
            pylab.subplot(1, n_stages, i + 1), pylab.title("stage " + str(stage_name))
            pylab.title('ROC at stage ' + str(stage_name))
            pylab.plot([0, 1], [1, 0], 'k--')
            pylab.xlim([0., 1.003]), pylab.xlabel('Signal Efficiency')
            pylab.ylim([0., 1.003]), pylab.ylabel('Background Rejection')
            for classifier_name, predictions in proba_on_stage.iteritems():
                plot_roc(self.y, predictions[:, 1], sample_weight=self.sample_weight,
                         classifier_name=classifier_name)
            pylab.legend(loc="lower left")
        return self

    def prediction_pdf(self, stages=None, histtype='step', bins=30, show_legend=False):
        proba_on_stages = pandas.DataFrame(self._get_stages(stages))
        for stage_name, proba_on_stage in proba_on_stages.iterrows():
            self._strip_figure(len(proba_on_stage))
            for i, (clf_name, predict_proba) in enumerate(proba_on_stage.iteritems(), 1):
                pylab.subplot(1, len(proba_on_stage), i)
                for label in numpy.unique(self.y):
                    pylab.hist(predict_proba[self.y == label, label], histtype=histtype, bins=bins, label=str(label))
                pylab.title('Predictions of %s at stage %s' % (clf_name, str(stage_name)))
                if show_legend:
                    pylab.legend()
            pylab.show()

    def learning_curves(self, metrics=roc_auc_score, step=1, label=1, mask=None):
        y_true = (self.y == label) * 1
        mask = self._check_mask(mask)
        self._plot_curves(lambda p: metrics(y_true[mask], p[mask, label], sample_weight=self.sample_weight), step=step)
        pylab.legend(loc="lower right")
        pylab.xlabel("stage"), pylab.ylabel("ROC AUC")

    def compute_metrics(self, stages=None, metrics=roc_auc_score, label=1):
        """ Computes arbitrary metrics on selected stages
        :param stages: array-like of stages or None
        :param metrics: (numpy.array, numpy.array, numpy.array | None) -> float,
            any metrics with interface (y_true, y_pred, sample_weight=None), where y_pred of shape [n_samples] of float
        :return: pandas.DataFrame with computed values
        """
        def _compute_metrics(proba):
            return metrics((self.y == label) * 1, proba[:, label], sample_weight=self.sample_weight)

        return pandas.DataFrame(self._map_on_stages(_compute_metrics, stages=stages))

    #endregion

    #region Uniformity-related methods

    def _compute_bin_indices(self, var_names, n_bins=20, mask=None):
        """Mask is used to show events that will be binned afterwards
        (for instance if only signal events will be binned, then mask= y == 1)"""
        for var in var_names:
            assert var in self.X.columns, "the variable %i is not in dataset" % var
        mask = self._check_mask(mask)
        bin_limits = []
        for var_name in var_names:
            var_data = self.X.loc[mask, var_name]
            bin_limits.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), n_bins + 1)[1: -1])
        return compute_bin_indices(self.X.ix[:, var_names].values, bin_limits=bin_limits)

    def _compute_nonempty_bins_mask(self, var_names, n_bins=20, mask=None):
        return numpy.bincount(self._compute_bin_indices(var_names, n_bins=n_bins, mask=mask),
                              minlength=n_bins ** len(var_names)) > 0

    def _compute_bin_masscenters(self, var_name, n_bins=20, mask=None):
        bin_indices = self._compute_bin_indices([var_name], n_bins=n_bins, mask=mask)
        result = []
        for bin in range(numpy.max(bin_indices) + 1):
            result.append(numpy.median(self.X.ix[(bin_indices == bin) & mask, var_name]))
        return numpy.array(result)

    def _compute_bin_centers(self, var_names, n_bins=20, mask=None):
        """Mask is used to show events that will be binned after"""
        bin_centers = []
        mask = self._check_mask(mask)
        for var_name in var_names:
            var_data = self.X.loc[mask, var_name]
            bin_centers.append(numpy.linspace(numpy.min(var_data), numpy.max(var_data), 2 * n_bins + 1)[1::2])
            assert len(bin_centers[-1]) == n_bins
        return bin_centers

    def sde_curves(self, uniform_variables, target_efficiencies=None, n_bins=20, step=3, power=2., label=1,
                   return_data=False):
        mask = self.y == label
        bin_indices = self._compute_bin_indices(uniform_variables, n_bins=n_bins, mask=mask)
        target_efficiencies = self._check_efficiencies(target_efficiencies)

        def compute_sde(pred):
            return compute_sde_on_bins(pred[:, label], mask=mask, bin_indices=bin_indices,
                                       target_efficiencies=target_efficiencies, power=power,
                                       sample_weight=self.checked_sample_weight)

        result = self._plot_curves(compute_sde, step=step)
        pylab.xlabel("stage"), pylab.ylabel("SDE")
        pylab.ylim(0, pylab.ylim()[1] * 1.15)
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True, shadow=True)
        if return_data:
            return result

    def sde_knn_curves(self, uniform_variables, target_efficiencies=None, knn=30, step=3, power=2, label=1,
                       return_data=True):
        """Warning: this functions is very slow, specially on large datasets"""
        mask = self.y == label
        knn_indices = computeSignalKnnIndices(uniform_variables, self.X, is_signal=mask, n_neighbors=knn)
        knn_indices = knn_indices[mask, :]
        target_efficiencies = self._check_efficiencies(target_efficiencies)

        def compute_sde(pred):
            return compute_sde_on_groups(pred[:, label], mask, groups_indices=knn_indices,
                                         target_efficiencies=target_efficiencies,
                                         power=power, sample_weight=self.sample_weight)

        result = self._plot_curves(compute_sde, step=step)
        pylab.xlabel("stage"), pylab.ylabel("SDE")
        pylab.ylim(0, pylab.ylim()[1] * 1.15)
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True, shadow=True)
        if return_data:
            return result

    def theil_curves(self, uniform_variables, target_efficiencies=None, n_bins=20, label=1, step=3, return_data=True):
        mask = self.y == label
        bin_indices = self._compute_bin_indices(uniform_variables, n_bins=n_bins, mask=mask)
        target_efficiencies = self._check_efficiencies(target_efficiencies)

        def compute_theil(pred):
            return compute_theil_on_bins(pred[:, label], mask=mask, bin_indices=bin_indices,
                                         target_efficiencies=target_efficiencies,
                                         sample_weight=self.checked_sample_weight)

        result = self._plot_curves(compute_theil, step=step)
        pylab.ylabel("Theil Index")
        pylab.ylim(0, pylab.ylim()[1] * 1.15)
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True, shadow=True)
        if return_data:
            return result

    def ks_curves(self, uniform_variables, n_bins=20, label=1, step=3, return_data=True):
        mask = self.y == label
        bin_indices = self._compute_bin_indices(uniform_variables, n_bins=n_bins, mask=mask)

        def compute_ks(pred):
            return bin_based_ks(pred[:, label], mask=mask, bin_indices=bin_indices,
                                sample_weight=self.checked_sample_weight)

        result = self._plot_curves(compute_ks, step=step)
        pylab.ylabel("KS flatness")
        pylab.ylim(0, pylab.ylim()[1] * 1.15)
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True, shadow=True)
        if return_data:
            return result

    def cvm_curves(self, uniform_variables, n_bins=20, label=1, step=3, power=1., return_data=True):
        """power = 0.5 to compare with SDE"""
        mask = self.y == label
        bin_indices = self._compute_bin_indices(uniform_variables, n_bins=n_bins, mask=mask)

        def compute_cvm(pred):
            return bin_based_cvm(pred[mask, label], bin_indices=bin_indices[mask],
                                 sample_weight=self.checked_sample_weight[mask]) ** power

        result = self._plot_curves(compute_cvm, step=step)
        pylab.ylabel('CvM flatness')
        pylab.ylim(0, pylab.ylim()[1] * 1.15)
        pylab.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True, shadow=True)
        if return_data:
            return result

    def rcp(self, variable, global_rcp=None, n_bins=20, label=1,
            new_plot=True, ignored_sidebands=0., range=None, marker='.',
            show_legend=True, multiclassification=False, adjust_n_bins=True, mask=None,
            median_centers=True, compute_cuts_for_other_class=False, print_cut=False):
        """
        Right-classified part. This is efficiency for signal events, background rejection for background ones.
        In case of more than two classes this is the part of events of that class that was correctly classified.
        This function is needed to control correlation in more than one dimension.

        :param variable: feature name or array with values for each event in dataset
        :param stage: at which stage to compute (default=None, means after all stages)
        :param global_rcp: right-classified parts, for which cuts are computed (default=[0.5, 0.6, 0.7, 0.8, 0.9])
        :param cuts: in addition to global_rcp one can pass the precise values of cuts that will be used
        :param n_bins: number of bins (default 20)
        :param label: 1 for signal, 0 for background, or label of interested class if multiclassification
        :param new_plot: if False, will use the existing figure (default=True)
        :param ignored_sidebands: float, part of events from the left and right
            that will be ignored (default 0.001 = 0.1%)
        :param range: tuple or None, events with values of variable outside this range will be ignored
        :param multiclassification: bool, if False, 'physical' names will be used (efficiency, rejection)
        :param median_centers: bool, if True, the x of point is median of masses inside bin,
            otherwise mean of the bounds
        :param compute_cuts_for_other_class: if True, the computed cuts will correspond to rcp of opposite class
            (available only for binary classification)
        """
        if multiclassification:
            assert not compute_cuts_for_other_class, 'this option is unavailable for multiclassification'
        if not multiclassification:
            assert label in {0, 1}, 'for binary classification label should be in [0, 1]'
        mask = self._check_mask(mask)
        inner_mask = (mask > 0.5) & (self.y == label)

        if range is not None:
            left, right = range
        else:
            signal_masses = self.X.loc[mask, variable].values
            left, right = numpy.percentile(signal_masses, [100 * ignored_sidebands, 100 * (1. - ignored_sidebands)])
            left -= 0.5
            right += 0.5

        masses = self.X.loc[:, variable].values
        inner_mask &= (masses >= left) & (masses <= right)
        if adjust_n_bins:
            n_bins = min(n_bins, len(numpy.unique(masses[mask])))

        bin_indices = self._compute_bin_indices([variable], n_bins=n_bins, mask=inner_mask)
        if median_centers:
            bin_centers = self._compute_bin_masscenters(variable, n_bins=n_bins, mask=inner_mask)
        else:
            bin_centers, = self._compute_bin_centers([variable], n_bins=n_bins, mask=inner_mask)

        # Leave only non-empty
        bin_mask = self._compute_nonempty_bins_mask([variable], n_bins=n_bins, mask=inner_mask)

        global_rcp = self._check_efficiencies(global_rcp)

        n_classifiers = len(self.predictions)
        if new_plot:
            self._strip_figure(n_classifiers)

        if multiclassification:
            ylabel = 'right-classified part'
            legend_label = 'rcp={rcp:.2f}'
        elif label == 1:
            ylabel = 'signal efficiency'
            legend_label = 'avg eff={rcp:.2f}' if not compute_cuts_for_other_class else 'bck rej={rcp:.2f}'
        else:
            ylabel = 'background rejection'
            legend_label = 'avg rej={rcp:.2f}' if not compute_cuts_for_other_class else 'avg eff={rcp:.2f}'
        if print_cut:
            legend_label += '(cut={cut:.2f})'

        for i, (name, proba) in enumerate(self.predictions.items(), start=1):
            ax = pylab.subplot(1, n_classifiers, i)
            for eff in global_rcp:
                if not compute_cuts_for_other_class:
                    cut = compute_bdt_cut(eff, y_true=mask, y_pred=proba[:, label],
                                          sample_weight=self.checked_sample_weight)
                else:
                    cut = 1 - compute_bdt_cut(eff, y_true=mask, y_pred=proba[:, 1 - label],
                                              sample_weight=self.checked_sample_weight)

                bin_effs = compute_bin_efficiencies(proba[mask, label], bin_indices=bin_indices[mask], cut=cut,
                                                    sample_weight=self.checked_sample_weight[mask], minlength=n_bins)
                ax.plot(bin_centers[bin_mask], bin_effs[bin_mask], label=legend_label.format(rcp=eff, cut=cut),
                        marker=marker)

            ax.set_ylim(0, 1)
            ax.set_title(name)
            ax.set_xlabel(variable)
            ax.set_ylabel(ylabel)
            if show_legend:
                ax.legend(loc='best')

    def efficiency(self, uniform_variables, stages=None, target_efficiencies=None, n_bins=20, label=1):
        warnings.warn("This implementation of efficiency is considered outdated, consider using RCP",
                      DeprecationWarning)

        target_efficiencies = self._check_efficiencies(target_efficiencies)
        if len(uniform_variables) not in {1, 2}:
            raise ValueError("More than two variables are not implemented, you have a 3d-monitor? :)")

        mask = self.y == label
        bin_indices = self._compute_bin_indices(uniform_variables, n_bins, mask=mask)
        total_bins = n_bins ** len(uniform_variables)

        def compute_bin_effs(prediction_proba, target_eff):
            cut = compute_bdt_cut(target_eff, y_true=mask, y_pred=prediction_proba[:, label],
                                  sample_weight=self.checked_sample_weight)
            return compute_bin_efficiencies(prediction_proba[mask, label], bin_indices=bin_indices[mask],
                                            cut=cut, sample_weight=self.checked_sample_weight[mask],
                                            minlength=total_bins)

        if len(uniform_variables) == 1:
            effs = self._map_on_stages(stages=stages,
                                       function=lambda pred: [compute_bin_effs(pred, eff) for eff in
                                                              target_efficiencies])
            effs = pandas.DataFrame(effs)
            x_limits, = self._compute_bin_centers(uniform_variables, n_bins=n_bins, mask=mask)
            for stage_name, stage in effs.iterrows():
                self._strip_figure(len(stage))
                for i, (name, eff_stage_data) in enumerate(stage.iteritems()):
                    if isinstance(eff_stage_data, float) and pandas.isnull(eff_stage_data):
                        continue
                    ax = pylab.subplot(1, len(stage), i + 1)
                    for eff, local_effs in zip(target_efficiencies, eff_stage_data):
                        ax.set_ylim(0, 1)
                        ax.plot(x_limits, local_effs, label='eff=%.2f' % eff)
                        ax.set_title(name)
                        ax.set_xlabel(uniform_variables[0])
                        ax.set_ylabel('efficiency')
                        ax.legend(loc='best')
        else:
            x_limits, y_limits = self._compute_bin_centers(uniform_variables, n_bins=n_bins, mask=mask)
            bin_weights = compute_bin_weights(bin_indices, sample_weight=self.checked_sample_weight)
            bin_weights.resize(total_bins)
            for target_efficiency in target_efficiencies:
                staged_results = self._map_on_stages(lambda x: compute_bin_effs(x, target_efficiency), stages=stages)
                staged_results = pandas.DataFrame(staged_results)
                for stage_name, stage_data in staged_results.iterrows():
                    print("Stage %s, efficiency=%.2f" % (str(stage_name), target_efficiency))
                    self._strip_figure(len(stage_data))
                    for i, (name, local_efficiencies) in enumerate(stage_data.iteritems(), start=1):
                        if isinstance(local_efficiencies, float) and pandas.isnull(local_efficiencies):
                            continue
                        local_efficiencies[bin_weights <= 0] = target_efficiency
                        local_efficiencies = local_efficiencies.reshape([n_bins, n_bins], ).transpose()
                        # drawing difference, the efficiency in empty bins will be replaced with mean value
                        ax = pylab.subplot(1, len(stage_data), i)
                        p = ax.pcolor(x_limits, y_limits, local_efficiencies, cmap=cm.get_cmap("RdBu"),
                                      vmin=target_efficiency - 0.2, vmax=target_efficiency + 0.2)
                        ax.set_xlabel(uniform_variables[0]), ax.set_ylabel(uniform_variables[1])
                        ax.set_title(name)
                        pylab.colorbar(p, ax=ax)
                    pylab.show()
        return self

    def correlation_curves(self, var_name, center=None, step=1, label=1):
        """ Correlation between normalized(!) predictions on some class and a variable
        :type var_name: str, correlation is computed for this variable
        :type center: float|None, if float, the correlation is measured between |x - center| and prediction
        :type step: int
        :type label: int, label of class, the correlation is computed for the events of this class
        :rtype: Predictions, returns self
        """
        pylab.title("Pearson correlation with " + str(var_name))
        mask = self.y == label
        data = self.X.loc[mask, var_name]
        if center is not None:
            data = numpy.abs(data - center)
        weight = check_sample_weight(self.y, self.sample_weight)[mask]

        def compute_correlation(prediction_proba):
            pred = prediction_proba[mask, label]
            pred = build_normalizer(pred, sample_weight=weight)(pred)
            return pearsonr(pred, data)[0]

        correlations = self._map_on_staged_proba(compute_correlation, step=step)

        for classifier_name, staged_correlation in correlations.items():
            pylab.plot(staged_correlation.keys(), staged_correlation, label=classifier_name)
        pylab.legend(loc="lower left")
        pylab.xlabel("stage"), pylab.ylabel("Pearson correlation")
        return self

    #endregion

    def hist(self, var_names, n_bins=20, new_plot=True):
        """ Plots 1 and 2-dimensional distributions
        :param var_names: array-like of length 1 or 2 with name of variables to plot
        :param int n_bins: number of bins for histogram()
        :return: self """
        plot_classes_distribution(self.X, self.y, var_names, n_bins=n_bins, new_plot=new_plot)
        return self

    @staticmethod
    def _strip_figure(n):
        x_size = 12 if n == 1 else 12 + 3 * n
        y_size = 10 - n if n <= 5 else 4
        pylab.figure(figsize=(x_size, y_size))

    def show(self):
        pylab.show()
        return self


# Helpful functions that can be used separately

def plot_roc(y_true, y_pred, sample_weight=None, classifier_name="", is_cut=False, mask=None):
    """Plots ROC curve in the way physicists like it
    :param y_true: numpy.array, shape=[n_samples]
    :param y_pred: numpy.array, shape=[n_samples]
    :param sample_weight: numpy.array | None, shape = [n_samples]
    :param classifier_name: str, the name of classifier for label
    :param is_cut: predictions are binary
    :param mask: plot ROC curve only for events that have mask=True
    """
    if is_cut:
        assert len(numpy.unique(y_pred)) == 2, 'Cut assumes that prediction are 0 and 1 (or True/False)'

    MAX_STEPS = 500
    y_true, y_pred = check_arrays(y_true, y_pred)
    if mask is not None:
        mask = numpy.array(mask, dtype=bool)  # converting to bool, just in case
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]

    fpr, tpr, thresholds = check_arrays(*roc_curve(y_true, y_pred, sample_weight=sample_weight))
    roc_auc = auc(fpr, tpr)
    # tpr = recall = isSasS / isS = signal efficiency
    # fpr = isBasS / isB = 1 - specificity = 1 - backgroundRejection
    bg_rejection = 1. - fpr

    if len(fpr) > MAX_STEPS:
        # decreasing the number of points in plot
        targets = numpy.linspace(0, 1, MAX_STEPS)
        x_ids = numpy.searchsorted(tpr, targets)
        y_ids = numpy.searchsorted(fpr, targets)
        indices = numpy.concatenate([x_ids, y_ids, [0, len(tpr) - 1]], )
        indices = numpy.unique(indices)
        tpr = tpr[indices]
        bg_rejection = bg_rejection[indices]
    if not is_cut:
        pylab.plot(tpr, bg_rejection, label='%s (area = %0.3f)' % (classifier_name, roc_auc))
    else:
        pylab.plot(tpr[1:2], bg_rejection[1:2], 'o', label='%s' % classifier_name)


def plot_classes_distribution(X, y, var_names, n_bins=20, new_plot=True):
    y = column_or_1d(y)
    labels = numpy.unique(y)
    if len(var_names) == 1:
        if new_plot:
            pylab.figure(figsize=(14, 7))
        pylab.title('Distribution of classes')
        for label in labels:
            pylab.hist(numpy.ravel(X.ix[y == label, var_names]), label='class=%i' % label, alpha=0.3, bins=n_bins)
            pylab.xlabel(var_names[0])
        pylab.legend()

    elif len(var_names) == 2:
        if new_plot:
            pylab.figure(figsize=(12, 10))
        pylab.title('Distribution of classes')
        x_var, y_var = var_names
        for label in labels:
            alpha = numpy.clip(2000. / numpy.sum(y == label), 0.02, 1)
            pylab.plot(X.loc[y == label, x_var], X.loc[y == label, y_var], '.',
                       alpha=alpha, label='class=' + str(label))
    else:
        raise ValueError("More than two variables are not implemented")


def plot_features_pdf(X, y, n_bins=20, n_columns=3, ignored_sideband=0.001, mask=None,
                      sig_label='sig', bck_label='bck', adjust_n_bins=True, normed=True):
    """
    Plots in concise form distributions of all features
    """
    columns = sorted(X.columns)
    mask = numpy.ones(len(X), dtype=bool) if mask is None else mask
    for i, column in enumerate(columns, 1):
        pylab.subplot((len(columns) + n_columns - 1) // n_columns, n_columns, i)
        feature_bins = n_bins
        if adjust_n_bins:
            feature_bins = min(n_bins, len(numpy.unique(X.ix[:, column])))

        limits = numpy.percentile(X.loc[mask, column], [100 * ignored_sideband, 100 * (1. - ignored_sideband)])
        pylab.hist(X.ix[(y == 1) & mask, column].values, bins=feature_bins, normed=normed,
                   range=limits, alpha=0.3, label=sig_label, color='b')
        pylab.hist(X.ix[(y == 0) & mask, column].values, bins=feature_bins, normed=normed,
                   range=limits, alpha=0.3, label=bck_label, color='r')
        pylab.legend(loc='best')
        pylab.title(column)


