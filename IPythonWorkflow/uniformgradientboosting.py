import scipy.sparse as sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import LossFunction, LOSS_FUNCTIONS, MultinomialDeviance, MeanEstimator, \
    LogOddsEstimator, BinomialDeviance
import numpy
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.tree.tree import DecisionTreeClassifier
from commonutils import generateSample
import commonutils


# TODO move computing of knn_indices to fit function of classifier ot transmit it as fit parameter
class KnnLossFunction(LossFunction):
    def __init__(self, n_classes, coefficients_matrix, initial_weights=None):
        if n_classes != 2:
            raise NotImplementedError("Only 2 classes supported!")
        LossFunction.__init__(self, 1)
        self.coefficients_matrix = coefficients_matrix
        self.coefficients_matrix_t = sparse.csr_matrix(coefficients_matrix.transpose())
        if initial_weights is None:
            initial_weights = numpy.ones(coefficients_matrix.shape[0])
        else:
            assert len(initial_weights) == coefficients_matrix.shape[0], "Different size"
        self.initial_weights = numpy.array(initial_weights)

    def __call__(self, y, pred):
        """Computing the loss itself"""
        assert len(y) == len(pred) == self.coefficients_matrix.shape[1], "something is wrong with sizes"
        y_signed = 2 * y - 1
        exponents = numpy.exp(- self.coefficients_matrix.dot(y_signed * numpy.ravel(pred)))
        return (self.initial_weights * exponents).sum()

    def negative_gradient(self, y, pred, **kwargs):
        assert len(y) == len(pred) == self.coefficients_matrix.shape[1], "something is wrong with sizes"
        y_signed = 2 * y - 1
        exponents = numpy.exp(- self.coefficients_matrix.dot(y_signed * numpy.ravel(pred)))
        result = self.coefficients_matrix_t.dot(self.initial_weights * exponents) * y_signed
        return result

    def init_estimator(self):
        return LogOddsEstimator()

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_mask, learning_rate=1.0, k=0):
        y_signed = 2 * y - 1
        self.update_exponents = self.initial_weights * numpy.exp(- self.coefficients_matrix.dot(y_signed * numpy.ravel(y_pred)))
        LossFunction.update_terminal_regions(self, tree, X, y, residual, y_pred, sample_mask, learning_rate, k)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        # terminal_region = numpy.where(terminal_regions == leaf)[0]
        y_signed = 2 * y - 1
        z = self.coefficients_matrix.dot((terminal_regions == leaf) * y_signed)
        alpha = numpy.sum(self.update_exponents * z) / (numpy.sum(self.update_exponents * z * z) + 1e-10)
        tree.value[leaf, 0, 0] = alpha


class PairwiseKnnLossFunction(KnnLossFunction):
    def __init__(self, trainX, trainY, uniform_variables, knn=5):
        is_signal = trainY > 0.5
        knn_signal = commonutils.computeSignalKnnIndices(uniform_variables, trainX, is_signal, knn)
        knn_bg = commonutils.computeSignalKnnIndices(uniform_variables, trainX, ~is_signal, knn)
        knn_bg[is_signal, :] = knn_signal[is_signal, :]

        rows = xrange(len(trainX) * knn)
        columns1 = numpy.repeat(numpy.arange(0, len(trainX)), knn)
        columns2 = knn_bg.flatten()
        data = numpy.ones(len(rows))

        coefficients_matrix = sparse.csr_matrix((data, (rows, columns1)), shape=[len(trainX) * knn, len(trainX)]) + \
            sparse.csr_matrix((data, (rows, columns2)), shape=[len(trainX) * knn, len(trainX)])

        KnnLossFunction.__init__(self, 2, coefficients_matrix)


class SimpleKnnLossFunction(KnnLossFunction):
    def __init__(self, trainX, trainY, uniform_variables, knn=5):
        is_signal = trainY > 0.5
        knn_indices = commonutils.computeKnnIndicesOfSameClass(uniform_variables, trainX, is_signal, knn)
        ind_ptr = numpy.arange(0, len(trainX) * knn + 1, knn)
        column_indices = knn_indices.flatten()
        data = numpy.ones(len(trainX) * knn)
        coefficients_matrix = sparse.csr_matrix((data, column_indices, ind_ptr), shape=(len(trainX), len(trainX)))
        KnnLossFunction.__init__(self, 2, coefficients_matrix)


class RandomKnnLossFunction(KnnLossFunction):
    def __init__(self, trainX, trainY, uniform_variables, rows, knn=5):
        is_signal = trainY > 0.5
        knn_indices = commonutils.computeKnnIndicesOfSameClass(uniform_variables, trainX, is_signal, knn)
        selected_originals = numpy.random.randint(0, len(trainX), rows)
        selected_knns = knn_indices[selected_originals, :]
        # TODO implement
        # for row in rows:






class MyGradientBoostingClassifier(GradientBoostingClassifier):
    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")

        if isinstance(self.loss, LossFunction):
            self.loss_ = self.loss
        else:
            if self.loss not in LOSS_FUNCTIONS:
                raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

            if self.loss == 'deviance':
                loss_class = (MultinomialDeviance
                              if len(self.classes_) > 2
                              else BinomialDeviance)
            else:
                loss_class = LOSS_FUNCTIONS[self.loss]

            if self.loss in ('huber', 'quantile'):
                self.loss_ = loss_class(self.n_classes_, self.alpha)
            else:
                self.loss_ = loss_class(self.n_classes_)

        if self.subsample <= 0.0 or self.subsample > 1:
            raise ValueError("subsample must be in (0,1]")

        if self.init is not None:
            if (not hasattr(self.init, 'fit')
                    or not hasattr(self.init, 'predict')):
                raise ValueError("init must be valid estimator")
            self.init_ = self.init
        else:
            self.init_ = self.loss_.init_estimator()

        if not (0.0 < self.alpha and self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0)")



def testGradient(loss, size=1000):
    y = numpy.random.random(size) > 0.5
    pred = numpy.random.random(size)
    epsilon = 1e-6
    val = loss(y, pred)
    gradient = numpy.zeros_like(pred)

    for i in range(size):
        pred2 = pred.copy()
        pred2[i] += epsilon
        val2 = loss(y, pred2)
        gradient[i] = (val2 - val) / epsilon

    n_gradient = loss.negative_gradient(y, pred)
    assert numpy.all(abs(n_gradient + gradient) < 1e-4), "Problem with functional gradient"
    print "loss is ok"

testGradient(KnnLossFunction(2, 3 * sparse.eye(1000, 1000)))

def testGradientBoosting():
    # Generating some samples correlated with first variable
    dist = 0.6
    testX, testY = generateSample(2000, 10, dist)
    trainX, trainY = generateSample(2000, 10, dist)
    # We will try to get uniform distribution along this variable
    uniform_variables = ['column0']
    base_estimator = DecisionTreeClassifier(min_samples_split=20, max_depth=None)
    n_estimators = 40
    samples = 2000
    samples_min = 200

    loss2 = SimpleKnnLossFunction(trainX, trainY, uniform_variables)
    print MyGradientBoostingClassifier(min_samples_split=20, loss=loss2, max_depth=None, learning_rate=.2,
        n_estimators=n_estimators).fit(trainX, trainY).score(testX, testY),

    loss3 = PairwiseKnnLossFunction(trainX[:samples_min], trainY[:samples_min], uniform_variables)
    print MyGradientBoostingClassifier(min_samples_split=20, loss=loss3, max_depth=None, learning_rate=.2,
        n_estimators=n_estimators).fit(trainX[:samples_min], trainY[:samples_min]).score(testX, testY),

    print MyGradientBoostingClassifier(min_samples_split=20, loss=KnnLossFunction(2, sparse.eye(1000, samples)),
                                       max_depth=None, learning_rate=.2, n_estimators=n_estimators)\
        .fit(trainX[:samples], trainY[:samples]).score(testX, testY),

    print AdaBoostClassifier(n_estimators=n_estimators, base_estimator=base_estimator).fit(trainX, trainY)\
        .score(testX, testY)

    print 'uniform gradient boosting is ok'

testGradientBoosting()