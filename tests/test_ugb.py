from __future__ import division, print_function, absolute_import
import numpy
from hep_ml.commonutils import generate_sample
from hep_ml.losses import compute_positions, BinomialDevianceLossFunction, SimpleKnnLossFunction, \
    BinFlatnessLossFunction, KnnFlatnessLossFunction
from hep_ml.ugradientboosting import uGradientBoostingClassifier


def check_orders(size=40):
    effs1 = compute_positions(numpy.arange(size), numpy.ones(size))
    p = numpy.random.permutation(size)
    effs2 = compute_positions(numpy.arange(size)[p], numpy.ones(size))
    assert numpy.all(effs1[p] == effs2), 'Efficiencies are wrong'
    assert numpy.all(effs1 == numpy.sort(effs1))


def test_gb_with_ada(n_samples=1000, n_features=10, distance=0.6):
    testX, testY = generate_sample(n_samples, n_features, distance=distance)
    trainX, trainY = generate_sample(n_samples, n_features, distance=distance)
    loss = BinomialDevianceLossFunction()
    clf = uGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=.2,
                                      subsample=0.7, n_estimators=10, train_variables=None)
    clf.fit(trainX, trainY)
    assert clf.n_features == n_features
    assert len(clf.feature_importances_) == n_features
    # checking that predict proba works
    for p in clf.staged_predict_proba(testX):
        assert p.shape == (n_samples, 2)
    assert numpy.all(p == clf.predict_proba(testX))


def test_gradient_boosting(n_samples=1000):
    # Generating some samples correlated with first variable
    distance = 0.6
    testX, testY = generate_sample(n_samples, 10, distance)
    trainX, trainY = generate_sample(n_samples, 10, distance)
    # We will try to get uniform distribution along this variable
    uniform_variables = ['column0']
    n_estimators = 20

    loss1 = SimpleKnnLossFunction(uniform_variables)
    # loss2 = PairwiseKnnLossFunction(uniform_variables, knn=10)
    loss3 = BinomialDevianceLossFunction()
    # loss4 = RandomKnnLossFunction(uniform_variables, samples * 2, knn=5, knn_factor=3)
    # loss5 = DistanceBasedKnnFunction(uniform_variables, knn=10, distance_dependence=lambda r: numpy.exp(-0.1 * r))
    loss6bin = BinFlatnessLossFunction(uniform_variables, ada_coefficient=0.5)
    loss7bin = BinFlatnessLossFunction(uniform_variables, ada_coefficient=0.5, uniform_label=[0, 1])
    loss6knn = KnnFlatnessLossFunction(uniform_variables, ada_coefficient=0.5)
    loss7knn = KnnFlatnessLossFunction(uniform_variables, ada_coefficient=0.5, uniform_label=[0, 1])
    # loss8 = NewFlatnessLossFunction(uniform_variables, ada_coefficient=0.5, uniform_label=1)
    # loss9 = NewFlatnessLossFunction(uniform_variables, ada_coefficient=0.5, uniform_label=[0, 1])

    for loss in [loss1, loss3, loss6bin, loss7bin, loss6knn, loss7knn]:
        result = uGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=.2,
                                             subsample=0.7, n_estimators=n_estimators, train_variables=None) \
            .fit(trainX[:n_samples], trainY[:n_samples]).score(testX, testY)
        assert result >= 0.7, "The quality is too poor: %.3f" % result

