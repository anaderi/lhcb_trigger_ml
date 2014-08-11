#!/usr/bin/env python2
"""
Tests whether the given classifer lives up to its probability estimates.
Counts the real estimation correctness and compares with the predictied probability.
For example, we expect that among entries with probability=0.6 around 6/10 should be class 1 and
4/10 class 0. Of course, this is not a sufficient condition for correctness of the
probability assesment, but a necessary one.
"""

import pylab as pl
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm, datasets
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

from commonutils import generate_sample
from reports import ClassifiersDict
from uboost import uBoostClassifier

def proba_test(classifier, x_test, y_test, n_bins=10, axis=None, figure=None, name=None):
    """Plots histogram for the number of records correctly classified for
    different classifier predicted probabilities. Returns:
    error - averaged weighted deviation of the true class correctenss from the
      estimated proba; axis - matplotplib Axes with a histogram plot;
      figure - matplotlib Figure
    :param BaseClassifier:classifier classifier to test.
       Must support fit, predict_proba and classes_. Must be fitted beforehand by the caller.
    :param np.array:x_test - x test points.
    :param np.array:y_test - y test points.
    :param int:n_bins number of bins for the histogram.
    :param matplotlib.Axes:axis Axes to plot at. If None will be created inside the function.
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

    if name is None:
       name = ''

    error = np.sum(np.abs(((histogram - center)*histogram_total)[histogram_total != 0])) \
            / len(proba_correct_class)
    axis.bar(center, histogram, align='center', width=width,
             label='%s: %f' % (name, error))
    axis.plot([center[0], center[-1]], [center[0], center[-1]],
             lw=10, c='r', alpha=0.5)

    if ax_is_ours:
        axis.legend()

    return error, axis, figure

def main():
    np.random.seed(42)
    clf_dict = ClassifiersDict({
        'Ada_SAMME': AdaBoostClassifier(algorithm="SAMME"),
        'Ada_SAMME_R: AdaBoostClassifier(algorithm="SAMME.R"),
        'Dummy': DummyClassifier("uniform"),
        'clf_uBoost_SAMME_R': uBoostClassifier(
            uniform_variables=['column0'],
            n_neighbors=50,
            efficiency_steps=5,
            n_estimators=50,
            algorithm="SAMME.R"),
        'clf_uBoost_SAMME': uBoostClassifier(
            uniform_variables=['column0'],
            n_neighbors=50,
            efficiency_steps=5,
            n_estimators=50,
            algorithm="SAMME"),
        'GradientBoosting': GradientBoostingClassifier(),
        'Bayes': GaussianNB()
        })
    x_train, y_train = generate_sample(5000, 10, 0.42)
    x_test, y_test = generate_sample(3000, 10, 0.42)
    clf_dict.fit(x_train, y_train)
    for name, classifier in clf_dict.iteritems():
        print("%s: %f" % (name,
                          proba_test(classifier, x_test, y_test, name=name)[0]))

    pl.show()

if __name__ == '__main__':
    main()
