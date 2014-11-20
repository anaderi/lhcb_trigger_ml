from __future__ import division, print_function, absolute_import
import numpy
from hep_ml.commonutils import generate_sample
from hep_ml.losses import compute_positions, AdaLossFunction, SimpleKnnLossFunction, \
    BinFlatnessLossFunction, KnnFlatnessLossFunction
from hep_ml.ugradientboosting import uGradientBoostingClassifier


def check_orders(size=40):
    effs1 = compute_positions(numpy.arange(size), numpy.ones(size))
    p = numpy.random.permutation(size)
    effs2 = compute_positions(numpy.arange(size)[p], numpy.ones(size))
    assert numpy.all(effs1[p] == effs2), 'Efficiencies are wrong'
    assert numpy.all(effs1 == numpy.sort(effs1))


def check_gradient(loss, size=1000):
    X, y = generate_sample(size, 10)
    sample_weight = numpy.ones(size)
    loss.fit(X, y, sample_weight=sample_weight)
    pred = numpy.random.random(size)
    epsilon = 1e-7
    val = loss(pred)
    gradient = numpy.zeros_like(pred)

    for i in range(size):
        pred2 = pred.copy()
        pred2[i] += epsilon
        val2 = loss(pred2)
        gradient[i] = (val2 - val) / epsilon

    n_gradient = loss.negative_gradient(pred)
    assert numpy.all(abs(n_gradient + gradient) < 1e-3), "Problem with functional gradient"


def test_gradient_boosting(samples=1000):
    # Generating some samples correlated with first variable
    distance = 0.6
    testX, testY = generate_sample(samples, 10, distance)
    trainX, trainY = generate_sample(samples, 10, distance)
    # We will try to get uniform distribution along this variable
    uniform_variables = ['column0']
    n_estimators = 20

    loss1 = SimpleKnnLossFunction(uniform_variables)
    # loss2 = PairwiseKnnLossFunction(uniform_variables, knn=10)
    loss3 = AdaLossFunction()
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
            .fit(trainX[:samples], trainY[:samples]).score(testX, testY)
        assert result >= 0.7, "The quality is too poor: %.3f" % result

    for loss in [loss1, loss3, ]:
        check_gradient(loss)

    print('uniform gradient boosting is ok')


# TODO test that in the bins/groups we have only events of the needed class

