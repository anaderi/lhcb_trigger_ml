from __future__ import division, print_function, absolute_import

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from matplotlib.cbook import Null

from hep_ml import reports
from hep_ml.metrics import roc_auc_score
from hep_ml.reports import ClassifiersDict
from hep_ml.commonutils import generate_sample


class MyNull(Null):
    def ylim(self, *args):
        return [0, 1]


def test_reports(null_pylab=True):
    if null_pylab and __name__ != '__main__':
        reports.pylab = MyNull()

    print(reports.pylab.ylim)

    trainX, trainY = generate_sample(1000, 10)
    testX, testY = generate_sample(1000, 10)

    for low_memory in [True]:
        classifiers = ClassifiersDict()
        classifiers['ada'] = AdaBoostClassifier(n_estimators=20)
        classifiers['forest'] = RandomForestClassifier(n_estimators=20)

        pred = classifiers.fit(trainX, trainY).test_on(testX, testY, low_memory=low_memory)
        pred.roc().show().sde_curves(['column0'])

        pred.correlation_curves('column1', ).show() \
            .learning_curves().show() \
            .efficiency(trainX.columns[:1], n_bins=7).show() \
            .efficiency(trainX.columns[:2], n_bins=12, target_efficiencies=[0.5]).show() \
            .roc(stages=[10, 15]).show() \
            .hist(['column0']).show() \
            .compute_metrics(stages=[5, 10], metrics=roc_auc_score)

        pred.rcp('column0', n_bins=7)
        pred.show()

    reports.plot_features_pdf(trainX, trainY)

