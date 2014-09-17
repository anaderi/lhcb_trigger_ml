from __future__ import print_function
from __future__ import division

from collections import defaultdict
import copy
import numbers
import warnings

import pandas
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy

from sklearn.tree.tree import DecisionTreeRegressor, DTYPE
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_arrays, column_or_1d

from commonutils import check_sample_weight, computeSignalKnnIndices
import commonutils
from metrics import compute_group_weights
import metrics


__author__ = 'Alex Rogozhnikov'


#region utilities

def compute_positions(y_pred, sample_weight):
    """For each event computes it position among other events by prediction.
    position = part of elements with lower predictions"""
    order = numpy.argsort(y_pred)
    ordered_weights = sample_weight[order]
    ordered_weights /= float(numpy.sum(ordered_weights))
    efficiencies = (numpy.cumsum(ordered_weights) - 0.5 * ordered_weights)
    return efficiencies[numpy.argsort(order)]


def check_orders(size=10):
    effs1 = compute_positions(numpy.arange(size), numpy.ones(size))
    p = numpy.random.permutation(size)
    effs2 = compute_positions(numpy.arange(size)[p], numpy.ones(size))
    assert numpy.all(effs1[p] == effs2), 'Efficiencies are wrong'
    assert numpy.all(effs1 == numpy.sort(effs1))

check_orders()


def random_sample_mask(size, n_inbag, random_state):
    if n_inbag >= size:
        return numpy.ones(size, dtype=bool)
    indices = random_state.choice(size, n_inbag, replace=False)
    result = numpy.zeros(size, dtype=bool)
    result[indices] = True
    return result

#endregion


class AbstractLossFunction(BaseEstimator):
    def fit(self, X, y, sample_weight):
        """ This method is optional, it is called before all the others"""
        pass

    def negative_gradient(self, y_pred):
        """The y_pred should contain all the events passed to `fit` method,
        moreover, the order should be the same"""
        raise NotImplementedError()

    def __call__(self, y_pred):
        """The y_pred should contain all the events passed to `fit` method,
        moreover, the order should be the same"""
        raise NotImplementedError()

    def update_tree(self, tree, X, y, y_pred, sample_weight, update_mask, residual):
        """This method may be not called at all, so it shouldn't
        modify y_pred (unlike LossFunction from sklearn)"""

        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~update_mask] = -1

        for leaf, indices_in_leaf in commonutils.indices_of_values(masked_terminal_regions):
            if leaf == -1:
                continue
            self.update_tree_leaf(tree, leaf=leaf, indices_in_leaf=indices_in_leaf,
                                  X=X, y=y, y_pred=y_pred,
                                  sample_weight=sample_weight, update_mask=update_mask,
                                  residual=residual)

    def update_tree_leaf(self, tree, leaf, indices_in_leaf,
                         X, y, y_pred, sample_weight, update_mask, residual):
        pass


class AdaLossFunction(AbstractLossFunction):
    def fit(self, X, y, sample_weight):
        self.y = y
        self.sample_weight = sample_weight
        self.y_signed = 2 * y - 1

    def __call__(self, y_pred):
        return numpy.sum(self.sample_weight * numpy.exp(- self.y_signed * y_pred))

    def negative_gradient(self, y_pred):
        return self.y_signed * self.sample_weight * numpy.exp(- self.y_signed * y_pred)

    def update_tree_leaf(self, tree, leaf, indices_in_leaf,
                         X, y, y_pred, sample_weight, update_mask, residual):
        leaf_ans = y[indices_in_leaf]
        leaf_exp = sample_weight[indices_in_leaf] * numpy.exp(- self.y_signed[indices_in_leaf] * y_pred[indices_in_leaf])
        w1 = numpy.sum(leaf_exp[leaf_ans == 1])
        w2 = numpy.sum(leaf_exp[leaf_ans == 0])
        # regularization
        w_sum = w1 + w2
        w1 += 1e-5 * w_sum
        w2 += 1e-5 * w_sum
        # minimization of w1 * e^(-x) + w2 * e^x
        return 0.5 * numpy.log(w1 / w2)

#region MatrixLossFunction

class KnnLossFunction(AbstractLossFunction):
    def __init__(self, uniform_variables):
        """KnnLossFunction is a base class to be inherited by other loss functions,
        which choose the particular A matrix and w vector. The formula of loss is:
        loss = \sum_i w_i * exp(- \sum_j a_ij y_j score_j)
        """
        self.uniform_variables = uniform_variables
        # real matrix and vector will be computed during fitting
        self.A = None
        self.A_t = None
        self.w = None

    def fit(self, X, y, sample_weight):
        """This method is used to compute A matrix and w based on train dataset"""
        assert len(X) == len(y), "different size of arrays"
        A, w = self.compute_parameters(X, y)
        self.A = sparse.csr_matrix(A)
        self.A_t = sparse.csr_matrix(self.A.transpose())
        self.w = numpy.array(w)
        assert A.shape[0] == len(w), "inconsistent sizes"
        assert A.shape[1] == len(X), "wrong size of matrix"
        self.y_signed = 2 * y - 1
        return self

    def __call__(self, y_pred):
        """Computing the loss itself"""
        assert len(y_pred) == self.A.shape[1], "something is wrong with sizes"
        exponents = numpy.exp(- self.A.dot(self.y_signed * y_pred))
        return numpy.sum(self.w * exponents)

    def negative_gradient(self, y_pred):
        """Computing negative gradient"""
        assert len(y_pred) == self.A.shape[1], "something is wrong with sizes"
        exponents = numpy.exp(- self.A.dot(self.y_signed * y_pred))
        result = self.A_t.dot(self.w * exponents) * self.y_signed
        return result

    def compute_parameters(self, trainX, trainY):
        """This method should be overloaded in descendant, and should return A, w (matrix and vector)"""
        raise NotImplementedError()

    def update_tree(self, tree, X, y, y_pred, sample_weight, update_mask, residual):
        self.update_exponents = self.w * numpy.exp(- self.A.dot(self.y_signed * y_pred))
        AbstractLossFunction.update_tree(self, tree, X, y, y_pred, sample_weight, update_mask, residual)

    def update_tree_leaf(self, tree, leaf, indices_in_leaf,
                         X, y, y_pred, sample_weight, update_mask, residual):
        terminal_region = numpy.zeros(len(X), dtype=float)
        terminal_region[indices_in_leaf] += 1
        z = self.A.dot(terminal_region * self.y_signed)
        # optimal value here by several steps?
        alpha = numpy.sum(self.update_exponents * z) / (numpy.sum(self.update_exponents * z * z) + 1e-10)
        tree.value[leaf, 0, 0] = alpha


class SimpleKnnLossFunction(KnnLossFunction):
    def __init__(self, uniform_variables, knn=10, uniform_label=1, distinguish_classes=True, row_norm=1.):
        """A matrix is square, each row corresponds to a single event in train dataset, in each row we put ones
        to the closest neighbours of that event if this event from class along which we want to have uniform prediction.
        :param list[str] uniform_variables: the features, along which uniformity is desired
        :param int knn: the number of nonzero elements in the row, corresponding to event in 'uniform class'
        :param int|list[int] uniform_label: the label (labels) of 'uniform classes'
        :param bool distinguish_classes: if True, 1's will be placed only for
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.row_norm = row_norm
        self.uniform_label = [uniform_label] if isinstance(uniform_label, numbers.Number) else uniform_label
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        sample_weight = numpy.ones(len(trainX))
        A_parts = []
        w_parts = []
        for label in self.uniform_label:
            label_mask = trainY == label
            n_label = numpy.sum(label_mask)
            if self.distinguish_classes:
                mask = label_mask
            else:
                mask = numpy.ones(len(trainY), dtype=numpy.bool)
            knn_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, trainX, mask, self.knn)
            knn_indices = knn_indices[label_mask, :]
            ind_ptr = numpy.arange(0, n_label * self.knn + 1, self.knn)
            column_indices = knn_indices.flatten()
            data = numpy.ones(n_label * self.knn, dtype=float) * self.row_norm / self.knn
            A_part = sparse.csr_matrix((data, column_indices, ind_ptr), shape=[n_label, len(trainX)])
            w_part = numpy.mean(numpy.take(sample_weight, knn_indices), axis=1)
            assert A_part.shape[0] == len(w_part)
            A_parts.append(A_part)
            w_parts.append(w_part)

        for label in set(trainY) - set(self.uniform_label):
            label_mask = trainY == label
            n_label = numpy.sum(label_mask)
            ind_ptr = numpy.arange(0, n_label + 1)
            column_indices = numpy.where(label_mask)[0].flatten()
            data = numpy.ones(n_label, dtype=float) * self.row_norm
            A_part = sparse.csr_matrix((data, column_indices, ind_ptr), shape=[n_label, len(trainX)])
            w_part = sample_weight[label_mask]
            A_parts.append(A_part)
            w_parts.append(w_part)

        A = sparse.vstack(A_parts, format='csr', dtype=float)
        w = numpy.concatenate(w_parts)
        assert A.shape == (len(trainX), len(trainX))
        return A, w

#endregion

#region FlatnessLossFunction


def exp_margin(margin):
    """ margin = - y_signed * y_pred """
    return numpy.exp(numpy.clip(margin, -1e5, 2))


class AbstractFlatnessLossFunction(AbstractLossFunction):
    def __init__(self, uniform_variables, uniform_label=1, power=2., ada_coefficient=1.,
                 allow_wrong_signs=True, use_median=False,
                 keep_debug_info=False):
        """
        This loss function contains separately penalty for non-flatness and ada_coefficient.
        The penalty for non-flatness is using bins.

        :type uniform_variables: the vars, along which we want to obtain uniformity
        :type n_bins: the number of bins along each axis
        :type uniform_label: int | list(int), the labels for which we want to obtain uniformity
        :type power: the loss contains the difference | F - F_bin |^p, where p is power
        :type ada_coefficient: coefficient of ada_loss added to this one. The greater the coefficient,
            the less we tend to uniformity.
        :type allow_wrong_signs: defines whether gradient may different sign from the "sign of class"
            (i.e. may have negative gradient on signal)
        """
        self.uniform_variables = uniform_variables
        if isinstance(uniform_label, numbers.Number):
            self.uniform_label = numpy.array([uniform_label])
        else:
            self.uniform_label = numpy.array(uniform_label)
        self.power = power
        self.ada_coefficient = ada_coefficient
        self.allow_wrong_signs = allow_wrong_signs
        self.keep_debug_info = keep_debug_info
        self.use_median = use_median

    def fit(self, X, y, sample_weight=None):
        sample_weight = check_sample_weight(y,  sample_weight=sample_weight)
        assert len(X) == len(y), 'lengths are different'
        X = pandas.DataFrame(X)

        self.group_indices = dict()
        self.group_weights = dict()

        occurences = numpy.zeros(len(X))
        for label in self.uniform_label:
            self.group_indices[label] = self.compute_groups_indices(X, y, label=label)
            self.group_weights[label] = compute_group_weights(self.group_indices[label], sample_weight=sample_weight)
            for group in self.group_indices[label]:
                occurences[group] += 1

        out_of_bins = (occurences == 0) & numpy.in1d(y, self.uniform_label)
        if numpy.mean(out_of_bins) > 0.01:
            warnings.warn("%i events out of all bins " % numpy.sum(out_of_bins), UserWarning)

        self.y = y
        self.y_signed = 2 * y - 1
        self.sample_weight = sample_weight
        self.divided_weight = sample_weight / numpy.maximum(occurences, 1)

        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)
        return self

    def compute_groups_indices(self, X, y, label):
        raise NotImplementedError()

    def __call__(self, pred):
        #TODO implement
        return 0

    def negative_gradient(self, y_pred):
        y_pred = numpy.ravel(y_pred)
        neg_gradient = numpy.zeros(len(self.y), dtype=numpy.float)

        for label in self.uniform_label:
            label_mask = self.y == label
            global_positions = numpy.zeros(len(y_pred), dtype=float)
            global_positions[label_mask] = \
                compute_positions(y_pred[label_mask], sample_weight=self.sample_weight[label_mask])

            for indices_in_bin in self.group_indices[label]:
                # TODO delete
                assert numpy.all(label_mask[indices_in_bin]), "Not all events in bin of appropriate class"

                local_pos = compute_positions(y_pred[indices_in_bin],
                                              sample_weight=self.sample_weight[indices_in_bin])
                global_pos = global_positions[indices_in_bin]
                bin_gradient = self.power * numpy.sign(local_pos - global_pos) * \
                               numpy.abs(local_pos - global_pos) ** (self.power - 1)

                neg_gradient[indices_in_bin] += bin_gradient

        neg_gradient *= self.divided_weight

        assert numpy.all(neg_gradient[~numpy.in1d(self.y, self.uniform_label)] == 0)

        y_signed = self.y_signed
        if self.keep_debug_info:
            self.debug_dict['pred'].append(numpy.copy(y_pred))
            self.debug_dict['fl_grad'].append(numpy.copy(neg_gradient))
            self.debug_dict['ada_grad'].append(y_signed * self.sample_weight * numpy.exp(-y_signed*y_pred))

        # adding ada
        neg_gradient += self.ada_coefficient * y_signed * self.sample_weight * exp_margin(-y_signed*y_pred)

        if not self.allow_wrong_signs:
            neg_gradient = y_signed * numpy.clip(y_signed * neg_gradient, 0, 1e5)

        return neg_gradient

    def update_tree_leaf(self, tree, leaf, indices_in_leaf,
                         X, y, y_pred, sample_weight, update_mask, residual):
        if self.use_median:
            residual = residual[indices_in_leaf]
            tree.value[leaf, 0, 0] = numpy.median(residual)
        else:
            tree.value[leaf, 0, 0] = numpy.clip(tree.value[leaf, 0, 0], -10, 10)


class BinFlatnessLossFunction(AbstractFlatnessLossFunction):
    def __init__(self, uniform_variables, n_bins=10, uniform_label=1, power=2., ada_coefficient=1.,
                 allow_wrong_signs=True, use_median=False, keep_debug_info=False):
        self.n_bins = n_bins
        AbstractFlatnessLossFunction.__init__(self, uniform_variables,
            uniform_label=uniform_label, power=power, ada_coefficient=ada_coefficient,
            allow_wrong_signs=allow_wrong_signs, use_median=use_median,
            keep_debug_info=keep_debug_info)

    def compute_groups_indices(self, X, y, label):
        """Returns a list, each element is events' indices in some group."""
        label_mask = y == label
        extended_bin_limits = []
        for var in self.uniform_variables:
            f_min, f_max = numpy.min(X[var][label_mask]), numpy.max(X[var][label_mask])
            extended_bin_limits.append(numpy.linspace(f_min, f_max, 2 * self.n_bins + 1))
        groups_indices = list()
        for shift in [0, 1]:
            bin_limits = []
            for axis_limits in extended_bin_limits:
                bin_limits.append(axis_limits[1 + shift:-1:2])
            bin_indices = metrics.compute_bin_indices(X, self.uniform_variables, bin_limits)
            groups_indices += list(metrics.bin_to_group_indices(bin_indices, mask=label_mask))
        return groups_indices


class KnnFlatnessLossFunction(AbstractFlatnessLossFunction):
    def __init__(self, uniform_variables, n_neighbours=100, uniform_label=1, power=2., ada_coefficient=1.,
                 max_groups_on_iteration=3000, allow_wrong_signs=True, use_median=False, keep_debug_info=False,
                 random_state=None):
        self.n_neighbours = n_neighbours
        self.max_group_on_iteration = max_groups_on_iteration
        self.random_state = random_state
        AbstractFlatnessLossFunction.__init__(self, uniform_variables,
            uniform_label=uniform_label, power=power, ada_coefficient=ada_coefficient,
            allow_wrong_signs=allow_wrong_signs, use_median=use_median,
            keep_debug_info=keep_debug_info)

    def compute_groups_indices(self, X, y, label):
        mask = y == label
        self.random_state = check_random_state(self.random_state)
        groups_mask = random_sample_mask(numpy.sum(mask), self.max_group_on_iteration, self.random_state)
        return computeSignalKnnIndices(self.uniform_variables, X, mask,
                                       n_neighbors=self.n_neighbours)[mask, :][groups_mask, :]


#endregion

#region Gradient Boosting classifier

#TODO different kinds of updating (all, other, random and so on)

class uGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss=None, n_estimators=10, learning_rate=0.1,
                 subsample=1., min_samples_split=2, min_samples_leaf=1,
                 max_features=None, max_leaf_nodes=None,
                 max_depth=3, init_estimator=None, update_tree=False,
                 criterion='mse', splitter='best', train_variables=None, random_state=None):

        """This class supports only two-class classification and only special losses
        derived from AbstractLossFunction.
        :type loss: AbstractLossFunction
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.init_estimator = init_estimator
        self.update_tree = update_tree
        self.train_variables = train_variables
        self.random_state = random_state
        self.criterion = criterion
        self.splitter = splitter

    def check_params(self):
        assert isinstance(self.loss, AbstractLossFunction), \
            'LossFunction should be derived from AbstractLossFunction'
        assert self.n_estimators > 0, 'n_estimators should be positive'
        assert 0 < self.subsample <= 1., 'subsample should be in (0, 1]'
        self.random_state = check_random_state(self.random_state)

    def fit(self, X, y, sample_weight=None):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        assert len(X) == len(y), 'Different lengths of X and y'
        X = pandas.DataFrame(X)
        y = numpy.array(column_or_1d(y), dtype=int)
        assert numpy.all(numpy.in1d(y, [0, 1])), \
            'Only two-class classification supported'
        self.check_params()

        self.estimators = []
        self.scores = []

        n_samples = len(X)
        n_inbag = int(self.subsample * len(X))
        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)

        # preparing for fitting in trees
        X = self.get_train_vars(X)
        X, y = check_arrays(X, y, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        y_pred = numpy.zeros(len(X), dtype=float)

        if self.init_estimator is not None:
            y_signed = 2 * y - 1
            self.init_estimator.fit(X, y_signed, sample_weight=sample_weight)
            y_pred += numpy.ravel(self.init_estimator.predict(X))

        for stage in range(self.n_estimators):
            # tree creation
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes)

            # tree learning
            residual = self.loss.negative_gradient(y_pred)
            train_mask = random_sample_mask(n_samples, n_inbag, self.random_state)

            tree.fit(X[train_mask], residual[train_mask], sample_weight=sample_weight[train_mask], check_input=False)
            # update tree leaves
            if self.update_tree:
                self.loss.update_tree(tree.tree_, X=X, y=y, y_pred=y_pred, sample_weight=sample_weight,
                                      update_mask=numpy.ones(len(X), dtype=bool), residual=residual)

            y_pred += self.learning_rate * tree.predict(X)
            self.estimators.append(tree)
            self.scores.append(self.loss(y_pred))

    def get_train_vars(self, X):
        if self.train_variables is None:
            return X
        else:
            return X.loc[:, self.train_variables]

    def score_to_proba(self, score):
        result = numpy.zeros([len(score), 2], dtype=float)
        result[:, 1] = commonutils.sigmoid_function(score, width=1.)
        result[:, 0] = 1. - result[:, 1]
        return result

    def staged_predict_score(self, X):
        X = self.get_train_vars(X)
        y_pred = numpy.zeros(len(X))
        if self.init_estimator is not None:
            y_pred += numpy.ravel(self.init_estimator.predict(X))
        yield y_pred
        for estimator in self.estimators:
            y_pred += self.learning_rate * estimator.predict(X)
            yield y_pred

    def predict_score(self, X):
        result = None
        for score in self.staged_predict_score(X):
            result = score
        return result

    def staged_predict_proba(self, X):
        for score in self.staged_predict_score(X):
            yield self.score_to_proba(score)

    def predict_proba(self, X):
        return self.score_to_proba(self.predict_score(X))

#endregion

