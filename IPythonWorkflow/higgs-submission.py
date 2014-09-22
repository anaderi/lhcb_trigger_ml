from __future__ import print_function
__author__ = 'axelr'

import numpy
import gradient_boosting as gb
import pandas
import transformations
import cPickle as pickle

trainX, trainY, trainW = gb.get_higgs_data()
testX = pandas.read_csv('/Users/axelr/ipython/datasets/higgs/test.csv', index_col='EventId')


clf = gb.ReweightingGB(loss=gb.AdaLossFunction()).set_params(
    loss__signal_curvature=0.7, learning_rate=0.004, min_samples_leaf=125,
    n_estimators=5000, max_features=14,
    update_tree=True, max_depth=12, subsample=0.15, sig_weight=0.1,
    weights_in_loss=False, update_on='all', recount_step=10000, smearing=0.)

clf.fit(transformations.extend_data(trainX), trainY, trainW)
p_old = clf.predict_proba(transformations.extend_data(testX))[:, 1]
with open('submission-classifier2.pkl', 'wb') as output:
    pickle.dump(clf, output)

with open('submission-classifier2.pkl', 'rb') as input:
    clf = pickle.load(input)

p = clf.predict_proba(transformations.extend_data(testX))[:, 1]

assert numpy.all(p == p_old)

threshold = 0.5

labels = numpy.array(['b'] * len(p))
labels[p >= threshold] = 's'
result_csv = pandas.DataFrame()
result_csv['EventId'] = numpy.array(testX.index)
result_csv['RankOrder'] = numpy.argsort(numpy.argsort(p)) + 1
result_csv['Class'] = labels
result_csv.to_csv('submission_2.csv', sep=',',  index=False)

print('signal_events: ' + str(numpy.sum(p > threshold)) + ' from ' + str(len(p)))