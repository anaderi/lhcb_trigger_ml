__author__ = 'axelr'
from itertools import islice

import numpy
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree.tree import DecisionTreeRegressor, DTYPE
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_arrays
from scipy import special

from commonutils import check_sample_weight
import commonutils


class BinomialDeviance(MyLossFunction):
    """Binomial deviance loss function for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.
    """
    def __init__(self):
        # we only need to fit one tree for binary clf.
        super(BinomialDeviance, self).__init__()

    def __call__(self, y, pred, sample_weight=None):
        """Compute the deviance (= 2 * negative log-likelihood). """
        # numpy.logaddexp(0, v) == log(1.0 + exp(v))
        pred = pred.ravel()
        return -2.0 * numpy.average((y * pred) - numpy.logaddexp(0.0, pred), weights=sample_weight)

    def fit(self, X, y, sample_weight=None):
        pass

    def negative_gradient(self, y, y_pred, sample_weight=None):
        """Compute the residual (= negative gradient). """
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        return (y - expit(y_pred.ravel())) * sample_weight

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Make a single Newton-Raphson step. our node estimate is given by:
            sum(y - prob) / sum(prob * (1 - prob))
        """
        terminal_region = numpy.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        # y = y.take(terminal_region, axis=0)
        prob = expit(pred.take(terminal_region, axis=0))
        sample_weight = sample_weight.take(terminal_region)

        numerator = numpy.sum(residual * sample_weight)
        denominator = numpy.sum(prob * (1. - prob) * sample_weight)
        tree.value[leaf, 0, 0] = numerator / (denominator + 1e-4)




class ReweightingForest(RandomForestClassifier):
    def __init__(self, sig_weight=1., pow_sig=1., pow_bg=1., n_estimators=10,
                 criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="auto",
                 bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None,
                 compute_importances=None):
        RandomForestClassifier.__init__(self)
        # Everything should be set via set_params
        self.sig_weight = sig_weight
        self.pow_bg = pow_bg
        self.pow_sig = pow_sig

    def fit(self, X, y, sample_weight=None):
        sample_weight = normalize_weight(y, sample_weight, sig_weight=self.sig_weight, pow_sig=self.pow_sig,
                                         pow_bg=self.pow_bg)
        return RandomForestClassifier.fit(self, X, y, sample_weight=sample_weight)


class ReweightingForestRegressor(RandomForestRegressor):
    def __init__(self, sig_weight=1., pow_sig=1., pow_bg=1., gap=1., n_estimators=10,
                 criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="auto",
                 bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 min_density=None, compute_importances=None):
        RandomForestRegressor.__init__(self)
        # Everything should be set via set_params
        self.sig_weight = sig_weight
        self.pow_bg = pow_bg
        self.pow_sig = pow_sig
        self.gap = gap

    def fit(self, X, y, sample_weight=None):
        sample_weight = normalize_weight(y, sample_weight, sig_weight=self.sig_weight, pow_sig=self.pow_sig,
                                         pow_bg=self.pow_bg)
        target = sample_weight + self.gap
        target[y == 0] *= -1
        RandomForestRegressor.fit(self, X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        pred = RandomForestRegressor.predict(self, X)
        result = numpy.zeros([len(X), 2])
        result[:, 1] = special.expit(pred / 1000.)
        result[:, 0] = 1. - result[:, 1]
        return result




class StochasticBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, train_part=0.05, max_depth=10, min_samples_leaf=40,
                 min_samples_leaf_control=100,
                 criterion='mse', splitter='best', max_features=None, test_size=None,
                 min_signal_part=0.95, learning_rate=0.1, sig_weight=1., random_state=None):
        self.n_estimators = n_estimators
        self.train_part = train_part
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.test_size = test_size
        self.min_signal_part = min_signal_part
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.sig_weight = sig_weight
        self.min_samples_leaf_control = min_samples_leaf_control

    def update_terminal_regions(self, tree, X, y, sample_weight, score):
        # residual is negative gradient
        # compute leaf for each sample in ``X``.
        # score = numpy.copy(score)
        # assert numpy.all(score >= 0.)
        # score[y > 0.5] /= numpy.mean(score[y > 0.5]) * 10 + 1e-6
        # score[y < 0.5] /= numpy.mean(score[y < 0.5]) * 10 + 1e-6

        # sample_weight = sample_weight * numpy.exp(-score)
        terminal_regions = tree.apply(X)
        n_regions = numpy.max(terminal_regions) + 1
        mask = y > 0.5
        w_sig = numpy.bincount(terminal_regions[mask], weights=sample_weight[mask], minlength=n_regions)
        w_bck = numpy.bincount(terminal_regions[~mask], weights=sample_weight[~mask], minlength=n_regions)
        n_sig = numpy.bincount(terminal_regions[mask], minlength=n_regions)
        n_bck = numpy.bincount(terminal_regions[~mask], minlength=n_regions)

        n_com = n_sig + n_bck

        # print(n_sig, n_bck)

        self.w_sig.append(w_sig * (n_com < self.min_samples_leaf_control))
        self.w_bck.append(w_bck * (n_com < self.min_samples_leaf_control)) #

        # update each leaf (= perform line search)
        for leaf in numpy.where(tree.children_left == TREE_LEAF)[0]:
            print(w_sig[leaf], w_bck[leaf], n_sig[leaf], n_bck[leaf], n_com[leaf])
            # n = n_sig[leaf] + n_bck[leaf]
            tree.value[leaf, 0, 0] = leaf
            # w = w_sig[leaf] + w_bck[leaf]
            # # n = n_sig[leaf] + n_bck[leaf]
            # # if n < self.min_samples_leaf_control:
            # #     tree.value[leaf, 0, 0] = 0.
            # # elif w_sig[leaf] <
            # if w_sig[leaf] < self.min_signal_part * w:
            #     tree.value[leaf, 0, 0] = 0.
            # elif n_sig[leaf] < self.min_samples_leaf_control:
            #     tree.value[leaf, 0, 0] = 0.
            # else:
            #     tree.value[leaf, 0, 0] = numpy.sqrt(n_sig[leaf]) * (w_sig[leaf] / w - self.min_signal_part) ** 2.

    def fit(self, X, y, sample_weight=None):
        X, y = check_arrays(X, y, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        sample_weight = normalize_weight(y, sample_weight, sig_weight=self.sig_weight)
        self.random_state = check_random_state(self.random_state)
        self.estimators = []
        score = numpy.zeros(len(X), dtype=float)
        y_signed = 2 * y - 1

        self.w_sig = []
        self.w_bck = []

        for _ in range(self.n_estimators):
            residual = y_signed
            # numpy.exp(- y_signed * score)
            # residual[y > 0.5] /= numpy.mean(residual[y > 0.5])
            # residual[y < 0.5] /= -numpy.mean(residual[y < 0.5])

            trainX, testX, trainY, testY, trainW, testW, trainR, testR, trainS, testS = \
                train_test_split(X, y, sample_weight, residual, score,
                                 train_size=self.train_part, test_size=self.test_size, random_state=self.random_state)

            tree = DecisionTreeRegressor(criterion=self.criterion, splitter=self.splitter,
                                         max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features, random_state=self.random_state)

            # fitting
            tree.fit(trainX, trainR, sample_weight=trainW, check_input=False)

            # post-pruning
            self.update_terminal_regions(tree.tree_, testX, testY, testW, testS)

            # updating score
            # score += self.learning_rate * tree.predict(X)
            self.estimators.append(tree)

    # def staged_predict_proba(self, X):
    #     score = numpy.zeros(len(X), dtype=float)
    #     proba = numpy.zeros([len(X), 2], dtype=float)
    #     results=numpy.zeros([len(X), self.n_estimators])
    #     for estimator in self.estimators:
    #         score += estimator.predict(X) / self.n_estimators
    #         proba[:, 1] = expit(score)
    #         proba[:, 0] = expit(-score)
    #         yield proba

    def staged_predict_proba(self, X):
        results=numpy.zeros([len(X), self.n_estimators], dtype=float)
        X, = check_arrays(X, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        w_s = numpy.zeros(len(X))
        w_b = numpy.zeros(len(X))

        for i, (estimator, e_ws, e_wb) in enumerate(zip(self.estimators, self.w_sig, self.w_bck)):
            indices = estimator.predict(X).astype(int)
            assert numpy.all(indices == estimator.tree_.apply(X))
            results[:, i] = e_ws[indices] / e_wb[indices]

        for i in range(1, self.n_estimators):
            score = numpy.median(results[:, :i], axis=1)
            proba = numpy.zeros([len(X), 2], dtype=float)
            proba[:, 1] = expit(score)
            proba[:, 0] = expit(-score)
            yield proba

    def predict_proba(self, X, percentile=50):
        results=numpy.zeros([len(X), self.n_estimators], dtype=float)
        X, = check_arrays(X, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        w_s = numpy.zeros(len(X))
        w_b = numpy.zeros(len(X))

        for i, (estimator, e_ws, e_wb) in enumerate(zip(self.estimators, self.w_sig, self.w_bck)):
            indices = estimator.predict(X).astype(int)

            assert numpy.all(indices == estimator.tree_.apply(X))
            results[:, i] = e_ws[indices] / e_wb[indices]
            w_s += e_ws[indices]
            w_b += e_wb[indices]

        score = numpy.percentile(results, percentile, axis=1)
        # score = w_s / (w_s + w_b + 0.01)

        proba = numpy.zeros([len(X), 2], dtype=float)
        proba[:, 1] = expit( score)
        proba[:, 0] = expit(-score)
        return proba



def test_stochastic():
    sb = StochasticBoosting(min_signal_part=0.6, min_samples_leaf=100, train_part=0.2, criterion='friedman_mse',
                            sig_weight=1., min_samples_leaf_control=10000, n_estimators=10)

    data, answers, weights = get_higgs_data()
    trainX, testX, trainY, testY, trainW, testW = commonutils.train_test_split(data, answers, weights, train_size=0.8)

    # trainX, trainY = commonutils.generate_sample(1000, 10)
    # testX, testY = commonutils.generate_sample(1000, 10)
    # weights = numpy.ones(1000)

    sb.fit(trainX, trainY, trainW)
    print(optimal_AMS(trainY, sb.predict_proba(trainX)[:, 1], sample_weight=trainW))
    print(commonutils.roc_auc_score(testY, sb.predict_proba(testX)[:, 1], sample_weight=testW))
    print(optimal_AMS(testY, sb.predict_proba(testX)[:, 1], sample_weight=testW))

    for perc in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print(perc)
        print(optimal_AMS(testY, sb.predict_proba(testX, percentile=perc)[:, 1], sample_weight=testW))

    for stage in islice(sb.staged_predict_proba(testX), None, None, 5):
        print(optimal_AMS(testY, stage[:, 1], sample_weight=testW))

# test_stochastic()




class ProjClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=1, knn=100):
        self.n_components = n_components
        self.knn = knn

    def fit(self, X, y, sample_weight=None):
        self.classes_ = numpy.array([0, 1])
        self.proj = GaussianRandomProjection(n_components=self.n_components)
        # self.knner = KNeighborsClassifier(n_neighbors=self.knn)
        self.knner = Knn1dClassifier(self.knn)
        self.proj.fit(X)
        X_new = self.proj.transform(X)
        # TODO sample weight!!
        self.knner.fit(X_new, y, sample_weight=sample_weight)
        print('ok')
        return self

    def predict_proba(self, X):
        X_new = self.proj.transform(X)
        return self.knner.predict_proba(X_new)

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)


class Knn1dClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, knn=100):
        self.knn = knn

    def fit(self, X, y, sample_weight=None):
        sample_weight = check_sample_weight(y,  sample_weight=sample_weight)
        X, y = check_arrays(X, y)
        assert X.shape[1] == 1
        X = numpy.ravel(X)
        indices = numpy.argsort(X)
        self.sorted_x = X[indices]
        self.sorted_y = y[indices]
        self.sorted_w = sample_weight[indices]
        window = numpy.hamming(2 * self.knn + 1)
        window[self.knn] = 0
        self.sig_w = numpy.convolve(self.sorted_w * self.sorted_y,       window, mode='same')
        self.bck_w = numpy.convolve(self.sorted_w * (1 - self.sorted_y), window, mode='same')
        assert len(self.sig_w) == len(self.bck_w) == len(self.sorted_y) == len(X)

    def predict_proba(self, X):
        indices = numpy.searchsorted(self.sorted_x, numpy.ravel(X))
        indices = numpy.clip(indices, 0, len(self.sig_w) - 1)
        s = self.sig_w[indices]
        b = self.bck_w[indices]
        proba = numpy.zeros([len(X), 2], dtype=float)
        proba[:, 0] = b / (s + b)
        proba[:, 1] = s / (s + b)
        return proba

