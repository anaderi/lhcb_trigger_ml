#!/usr/bin/env python2
"""Tests whether the given classifier lives up to its probability estimates.
Counts the real estimation correctness and compares with the predicted
probability. For example, we expect that among entries with
probability=0.6 around 6/10 should be class 1 and 4/10 class 0. Of
course, this is not a sufficient condition for correctness of the
probability assessment, but a necessary one.
"""

import pylab as pl
import numpy as np


__author__ = "Nikita Kazeev"
__copyright__ = "Copyright 2014, Yandex"
__all__ = ['proba_test']


def proba_test(classifier, x_test, y_test, n_bins=10,
               axis=None, figure=None, name=None):
    """Plots histogram for the number of records correctly classified for
    different classifier predicted probabilities. Returns:
    error - averaged weighted deviation of the true class correctness from the
      estimated proba; axis - matplotplib Axes with a histogram plot;
      figure - matplotlib Figure
    :param BaseClassifier:classifier classifier to test.
       Must support fit, predict_proba and classes_.
       Must be fitted beforehand by the caller.
    :param np.array:x_test - x test points.
    :param np.array:y_test - y test points.
    :param int:n_bins number of bins for the histogram.
    :param matplotlib.Axes:axis Axes to plot at. If None
        it will be created inside the function.
    :param matplotlib.Figure:figure figure containing the plot.
        If None will be created inside the function.
    :param str:name string to name the plot.
    """
    proba = classifier.predict_proba(x_test)
    raw_predict = np.argmax(proba, axis=1)
    predict = classifier.classes_.take(raw_predict, axis=0)

    # Probability assigned by the classifier to it's predicted correct class
    # TODO(kazeevn) do better slicing
    proba_correct_class = proba[np.arange(len(proba)), raw_predict]
    proba_for_hits = proba_correct_class[predict == y_test]

    histogram, bins = np.histogram(proba_for_hits, bins=n_bins)
    histogram_total, bins_total = np.histogram(proba_correct_class, bins=bins)

    assert np.array_equal(bins, bins_total)

    histogram = np.true_divide(histogram, histogram_total)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2.

    if axis is None:
        figure, axis = pl.subplots()
        axis.set_xlabel("Predicted proba")
        axis.set_ylabel("Real share")
        ax_is_ours = True
    else:
        ax_is_ours = False

    if name is None:
        name = ''

    error = np.sum(np.abs(
        ((histogram - center)*histogram_total)[histogram_total != 0])) \
        / len(proba_correct_class)
    axis.bar(center, histogram, align='center', width=width,
             label='%s: %f' % (name, error))
    axis.plot([center[0], center[-1]], [center[0], center[-1]],
              lw=10, c='r', alpha=0.5)

    if ax_is_ours:
        axis.legend()

    return error, axis, figure
