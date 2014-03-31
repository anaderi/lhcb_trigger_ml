import scipy.sparse as sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import LossFunction, LOSS_FUNCTIONS, MultinomialDeviance, MeanEstimator, \
    LogOddsEstimator, BinomialDeviance
import numpy
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.tree.tree import DecisionTreeClassifier
from commonutils import generateSample
import commonutils


class KnnLossFunction(LossFunction):
    def __init__(self, uniform_variables):
        """KnnLossFunction is a base class to be inherited by other loss functions,
        which choose the particular A matrix and w vector. The formula of loss is:
        loss = \sum_i w_i * exp(- \sum_j a_ij y_j score_j)
        """
        LossFunction.__init__(self, 1)
        self.uniform_variables = uniform_variables
        # real matrix and vector will be computed during fitting
        self.A = None
        self.A_t = None
        self.w = None

    def __call__(self, y, pred):
        """Computing the loss itself"""
        assert len(y) == len(pred) == self.A.shape[1], "something is wrong with sizes"
        y_signed = 2 * y - 1
        exponents = numpy.exp(- self.A.dot(y_signed * numpy.ravel(pred)))
        return (self.w * exponents).sum()

    def negative_gradient(self, y, pred, **kwargs):
        """Computing negative gradient"""
        assert len(y) == len(pred) == self.A.shape[1], "something is wrong with sizes"
        y_signed = 2 * y - 1
        exponents = numpy.exp(- self.A.dot(y_signed * numpy.ravel(pred)))
        result = self.A_t.dot(self.w * exponents) * y_signed
        return result

    def fit(self, X, y):
        """This method is used to compute A matrix and w based on train dataset"""
        assert len(X) == len(y), "different size of arrays"
        A, w = self.compute_parameters(X, y)
        self.A = sparse.csr_matrix(A)
        self.A_t = sparse.csr_matrix(self.A.transpose())
        self.w = numpy.array(w)
        assert A.shape[0] == len(w), "inconsistent sizes"
        assert A.shape[1] == len(X), "wrong size of matrix"
        return self

    def compute_parameters(self, trainX, trainY):
        """This method should be overloaded in descendant,
         and should return A, w (matrix and vector)"""
        raise NotImplementedError()

    def init_estimator(self, X=None, y=None):
        return LogOddsEstimator()

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_mask, learning_rate=1.0, k=0):
        y_signed = 2 * y - 1
        self.update_exponents = self.w * numpy.exp(- self.A.dot(y_signed * numpy.ravel(y_pred)))
        LossFunction.update_terminal_regions(self, tree, X, y, residual, y_pred, sample_mask, learning_rate, k)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, pred):
        # terminal_region = numpy.where(terminal_regions == leaf)[0]
        y_signed = 2 * y - 1
        z = self.A.dot((terminal_regions == leaf) * y_signed)
        alpha = numpy.sum(self.update_exponents * z) / (numpy.sum(self.update_exponents * z * z) + 1e-10)
        tree.value[leaf, 0, 0] = alpha


# Descendants of KnnLossFunction - particular cases, each has its own
# algorithm of generating A and w


class PairwiseKnnLossFunction(KnnLossFunction):
    def __init__(self, uniform_variables, knn):
        """ A is rectangular matrix, in each row we have only two '1's,
        all other elements are zeros, these two '1's are placed in the columns, corresponding to neighbours"""
        self.knn = knn
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        knn = self.knn
        knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, knn)

        rows = xrange(len(trainX) * knn)
        columns1 = numpy.repeat(numpy.arange(0, len(trainX)), knn)
        columns2 = knn_indices.flatten()
        data = numpy.ones(len(rows))

        A = sparse.csr_matrix((data, (rows, columns1)), shape=[len(trainX) * knn, len(trainX)]) + \
            sparse.csr_matrix((data, (rows, columns2)), shape=[len(trainX) * knn, len(trainX)])
        w = numpy.ones(len(trainX) * knn)
        return A, w



class SimpleKnnLossFunction(KnnLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5

        knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, self.knn)
        ind_ptr = numpy.arange(0, len(trainX) * self.knn + 1, self.knn)
        column_indices = knn_indices.flatten()
        data = numpy.ones(len(trainX) * self.knn)
        A = sparse.csr_matrix((data, column_indices, ind_ptr), shape=(len(trainX), len(trainX)))
        w = numpy.ones(len(trainX))
        return A, w



class RandomKnnLossFunction(KnnLossFunction):
    def __init__(self, uniform_variables, n_rows, knn=5, knn_factor=3):
        """A very general loss,
        at each iteration it takes some random event from train dataset,
        and selects randomly knn of its knn*knn_factor neighbours, the process is repeated 'n_rows' times"""
        self.n_rows = n_rows
        self.knn = knn
        self.knn_factor = knn_factor
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        knn_max = int(self.knn * self.knn_factor)
        knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, knn_max)
        selected_originals = numpy.random.randint(0, len(trainX), self.n_rows)
        selected_knns = knn_indices[selected_originals, :]
        groups_indices = numpy.zeros((self.n_rows, self.knn), dtype=numpy.int)
        for i, event_neighs in enumerate(selected_knns):
            indices = numpy.random.permutation(knn_max)[:self.knn]
            groups_indices[i] = event_neighs[indices]

        ind_ptr = numpy.arange(0, self.n_rows * self.knn + 1, self.knn)
        column_indices = groups_indices.flatten()
        data = numpy.ones(self.n_rows * self.knn)
        A = sparse.csr_matrix((data, column_indices, ind_ptr), shape=(self.n_rows, len(trainX)))
        w = numpy.ones(self.n_rows)
        return A, w


class AdaLossFunction(KnnLossFunction):
    def __init__(self):
        """Good old Ada loss, implemented as version of KnnLostFunction """
        KnnLossFunction.__init__(self, None)

    def compute_parameters(self, trainX, trainY):
        return sparse.eye(len(trainX), len(trainX)), numpy.ones(len(trainX))





class MyGradientBoostingClassifier(GradientBoostingClassifier):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                 max_depth=3, init=None, random_state=None,
                 max_features=None, verbose=0, train_variables=None):
        """
        GradientBoosting from sklearn, which is modified to work with KnnLossFunction and it's versions.
        Subsampling is not supported at the moment.
        :param loss: LossFunction or string
        """
        self.train_variables = train_variables
        GradientBoostingClassifier.__init__(self, loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, init=init, random_state=random_state, max_features=max_features, verbose=verbose)

    def get_train_variables(self, X):
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]

    def fit(self, X, y):
        # we enable to pass simply LossFunction object
        if isinstance(self.loss, LossFunction):
            self.loss_ = self.loss
        else:
            if self.loss not in LOSS_FUNCTIONS:
                raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

            if self.loss == 'deviance':
                loss_class = (MultinomialDeviance if len(self.classes_) > 2 else BinomialDeviance)
            else:
                loss_class = LOSS_FUNCTIONS[self.loss]

            if self.loss in ('huber', 'quantile'):
                self.loss_ = loss_class(self.n_classes_, self.alpha)
            else:
                self.loss_ = loss_class(self.n_classes_)

        if self.subsample <= 0.0 or self.subsample > 1:
            raise ValueError("subsample must be in (0,1]")

        if self.init is not None:
            if (not hasattr(self.init, 'fit') or not hasattr(self.init, 'predict')):
                raise ValueError("init must be valid estimator")
            self.init_ = self.init
        else:
            self.init_ = self.loss_.init_estimator()

        # fitting the loss id it needs
        if isinstance(self.loss_, KnnLossFunction):
            self.loss_.fit(X, y)

        return GradientBoostingClassifier.fit(self, self.get_train_variables(X), y)

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        # evrything connected with loss was moved to self.fit
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")
        if not (0.0 < self.alpha and self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0)")

    def predict(self, X):
        return GradientBoostingClassifier.predict(self, self.get_train_variables(X))

    def predict_proba(self, X):
        return GradientBoostingClassifier.predict_proba(self, self.get_train_variables(X))

    def staged_predict_proba(self, X):
        return GradientBoostingClassifier.staged_predict_proba(self, self.get_train_variables(X))




def testGradient(loss, size=1000):
    loss.fit(numpy.arange(size), numpy.arange(size))
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

testGradient(AdaLossFunction())


def testGradientBoosting():
    # Generating some samples correlated with first variable
    distance = 0.6
    testX, testY = generateSample(2000, 10, distance)
    trainX, trainY = generateSample(2000, 10, distance)
    # We will try to get uniform distribution along this variable
    uniform_variables = ['column0']
    base_estimator = DecisionTreeClassifier(min_samples_split=20, max_depth=5)
    n_estimators = 40
    samples = 2000

    loss2 = SimpleKnnLossFunction(uniform_variables)
    loss3 = PairwiseKnnLossFunction(uniform_variables, knn=10)
    loss4 = AdaLossFunction()
    loss5 = RandomKnnLossFunction(uniform_variables, samples * 2, knn=5, knn_factor=3)

    for loss in [loss2, loss3, loss4, loss5]:
        print MyGradientBoostingClassifier(min_samples_split=20, loss=loss, max_depth=5, learning_rate=.2,
            n_estimators=n_estimators, train_variables=None).fit(trainX[:samples], trainY[:samples]).score(testX, testY),

    print AdaBoostClassifier(n_estimators=n_estimators, base_estimator=base_estimator)\
        .fit(trainX, trainY).score(testX, testY)

    print 'uniform gradient boosting is ok'

testGradientBoosting()