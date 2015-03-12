from __future__ import division, print_function, absolute_import
import pandas

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from matplotlib.cbook import Null

from sklearn.metrics import roc_auc_score
from hep_ml import reports
from hep_ml.reports import ClassifiersDict
from hep_ml.commonutils import generate_sample


class MyNull(Null):
    def ylim(self, *args):
        return [0, 1]


trainX, trainY = generate_sample(1000, 10)
testX, testY = generate_sample(1000, 10)

classifiers = ClassifiersDict()
classifiers['ada'] = AdaBoostClassifier(n_estimators=20)
classifiers['forest'] = RandomForestClassifier(n_estimators=20)
predictions = classifiers.fit(trainX, trainY).test_on(testX, testY)


def test_reports(null_pylab=True):
    if null_pylab:
        reports.pylab = MyNull()

    predictions.sde_curves(['column0'])

    predictions.correlation_curves('column1', ).show()
    predictions.learning_curves()
    predictions.show()
    predictions.hist(['column0']).show()

    rocs = predictions.compute_metrics(stages=[5, 10], metrics=roc_auc_score)
    assert isinstance(rocs, pandas.DataFrame)

    reports.pylab.figure(figsize=[18, 10])
    reports.plot_features_pdf(trainX, trainY, n_columns=4)
    predictions.show()


def test_roc(null_pylab=True):
    if null_pylab:
        reports.pylab = MyNull()
    predictions.roc().show()
    predictions.roc(stages=[10, 15]).show()
    reports.plot_roc(trainY, trainY, is_cut=True)
    predictions.show()


def test_reports_efficiency(null_pylab=True):
    if null_pylab:
        reports.pylab = MyNull()

    predictions.efficiency(trainX.columns[:1], n_bins=7).show()

    predictions.rcp('column0', n_bins=7)
    predictions.show()

    predictions.rcp('column0', n_bins=30)
    predictions.show()

    predictions.efficiency(trainX.columns[:2], n_bins=12, target_efficiencies=[0.5]).show()


