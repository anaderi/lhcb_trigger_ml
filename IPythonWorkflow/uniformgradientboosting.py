from __future__ import print_function
from __future__ import division

from collections import defaultdict, OrderedDict
from itertools import izip
import numbers
from time import time
import itertools
import math
import scipy.sparse as sparse
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier as GBClassifier
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import LossFunction, LOSS_FUNCTIONS, MultinomialDeviance, \
    LogOddsEstimator, BinomialDeviance
import numpy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.tree._tree import DTYPE
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_arrays, column_or_1d
from commonutils import generate_sample, check_sample_weight
import commonutils
import reports

__author__ = 'Alex Rogozhnikov'

# TODO updating tree in FL and NFL


class KnnLossFunction(LossFunction, BaseEstimator):
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
        return numpy.sum(self.w * exponents)

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
        """This method should be overloaded in descendant, and should return A, w (matrix and vector)"""
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
                knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, label_mask,
                                                                       n_neighbours=self.knn)
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

        for label in set(trainY).difference(self.uniform_label):
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
        return A, w


class SimpleKnnLossFunctionEyeBg(KnnLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for signal.
        For background we have identity matrix.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        if self.distinguish_classes:
            knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, self.knn)
        if not self.distinguish_classes:
            is_signal = numpy.ones(len(trainY), dtype=numpy.bool)
            knn_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, trainX, is_signal, self.knn)

        bg_index = numpy.where(~ is_signal)[0]

        j = 0
        k = 0
        ind_ptr = [0]
        x = set(bg_index)
        column_indices_help = []
        for i in range(len(trainX)):
            if i in x:
                column_indices_help.append(bg_index[j])
                ind_ptr.append(k + 1)
                k += 1
                j += 1
            else:
                for n in knn_indices[i]:
                    column_indices_help.append(n)
                ind_ptr.append(k + self.knn)
                k += self.knn

        column_indices = numpy.array(column_indices_help)

        data = numpy.ones(len(column_indices))

        A = sparse.csr_matrix((data, column_indices, ind_ptr), shape=(len(trainX), len(trainX)))
        w = numpy.ones(len(trainX))
        return A, w


class SimpleKnnLossFunctionKnnOnDiagonalSignal(KnnLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for signal. For background we
        have identity matrix times self.knn.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        if self.distinguish_classes:
            knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, self.knn)
        if not self.distinguish_classes:
            is_signal = numpy.ones(len(trainY), dtype=numpy.bool)
            knn_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, trainX, is_signal, self.knn)

        bg_index = numpy.where(is_signal == False)[0]

        j = 0
        k = 0
        ind_ptr = [0]
        x = set(bg_index)
        column_indices_help = []
        for i in range(len(trainX)):
            if i in x:
                column_indices_help.append(bg_index[j])
                ind_ptr.append(k + 1)
                k += 1
                j += 1
            else:
                for n in knn_indices[i]:
                    column_indices_help.append(n)
                ind_ptr.append(k + self.knn)
                k += self.knn

        column_indices = numpy.array(column_indices_help)

        data = numpy.ones(len(column_indices))
        data[bg_index] = self.knn

        A = sparse.csr_matrix((data, column_indices, ind_ptr), shape=(len(trainX), len(trainX)))
        w = numpy.ones(len(trainX))
        return A, w


class SimpleKnnLossFunctionKnnOnDiagonalBg(KnnLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for signal. For background we
        have identity matrix times self.knn.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        if self.distinguish_classes:
            knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, self.knn)
        if not self.distinguish_classes:
            is_signal = numpy.ones(len(trainY), dtype=numpy.bool)
            knn_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, trainX, is_signal, self.knn)

        bg_index = numpy.where(is_signal == True)[0]

        j = 0
        k = 0
        ind_ptr = [0]
        x = set(bg_index)
        column_indices_help = []
        for i in range(len(trainX)):
            if i in x:
                column_indices_help.append(bg_index[j])
                ind_ptr.append(k + 1)
                k += 1
                j += 1
            else:
                for n in knn_indices[i]:
                    column_indices_help.append(n)
                ind_ptr.append(k + self.knn)
                k += self.knn

        column_indices = numpy.array(column_indices_help)

        data = numpy.ones(len(column_indices))
        data[bg_index] = self.knn

        A = sparse.csr_matrix((data, column_indices, ind_ptr), shape=(len(trainX), len(trainX)))

        w = numpy.ones(len(trainX))
        return A, w


class SimpleKnnLossFunctionEyeSignal(KnnLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for background.
        For signal we have identity matrix.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        if self.distinguish_classes:
            knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, self.knn)
        if not self.distinguish_classes:
            is_signal = numpy.ones(len(trainY), dtype=numpy.bool)
            knn_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, trainX, is_signal, self.knn)

        bg_index = numpy.where(is_signal)[0]

        j = 0
        k = 0
        ind_ptr = [0]
        x = set(bg_index)
        column_indices_help = []
        for i in range(len(trainX)):
            if i in x:
                column_indices_help.append(bg_index[j])
                ind_ptr.append(k + 1)
                k += 1
                j += 1
            else:
                for n in knn_indices[i]:
                    column_indices_help.append(n)
                ind_ptr.append(k + self.knn)
                k += self.knn

        column_indices = numpy.array(column_indices_help)
        data = numpy.ones(len(column_indices))

        A = sparse.csr_matrix((data, column_indices, ind_ptr), shape=(len(trainX), len(trainX)))

        w = numpy.ones(len(trainX))
        return A, w



class PairwiseKnnLossFunction(KnnLossFunction):
    def __init__(self, uniform_variables, knn, exclude_self=True, penalize_large_preds=True):
        """ A is rectangular matrix, in each row we have only two '1's,
        all other elements are zeros, these two '1's are placed in the columns, corresponding to neighbours
        exclude_self: bool, exclude self from knn?
        """
        self.knn = knn
        self.exclude_self = exclude_self
        self.penalize_large_preds = penalize_large_preds
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        knn = self.knn
        if self.exclude_self:
            knn_indices = \
                commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, knn+1)[:, 1:]
        else:
            knn_indices = commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, knn)

        rows = xrange(len(trainX) * knn)
        columns1 = numpy.repeat(numpy.arange(0, len(trainX)), knn)
        columns2 = knn_indices.flatten()
        data = numpy.ones(len(rows))

        A = sparse.csr_matrix((data, (rows, columns1)), shape=[len(trainX) * knn, len(trainX)]) + \
            sparse.csr_matrix((data, (rows, columns2)), shape=[len(trainX) * knn, len(trainX)])

        if self.penalize_large_preds:
            penalty1 = - sparse.eye(len(trainX), len(trainX))
            penalty2 = sparse.eye(len(trainX), len(trainX))
            A = sparse.vstack((A, penalty1, penalty2), format="csr")
        w = numpy.ones(A.shape[0])
        return A, w


class RandomKnnLossFunction(KnnLossFunction):
    def __init__(self, uniform_variables, n_rows, knn=5, knn_factor=3, large_preds_penalty=1.):
        """A general loss,
        at each iteration it takes some random event from train dataset,
        and selects randomly knn of its knn*knn_factor neighbours, the process is repeated 'n_rows' times"""
        self.n_rows = n_rows
        self.knn = knn
        self.knn_factor = knn_factor
        self.large_preds_penalty = large_preds_penalty
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

        if self.large_preds_penalty > 0:
            penalty1 = - self.large_preds_penalty * sparse.eye(len(trainX), len(trainX))
            penalty2 = self.large_preds_penalty * sparse.eye(len(trainX), len(trainX))
            A = sparse.vstack((A, penalty1, penalty2), format="csr")

        w = numpy.ones(A.shape[0])
        return A, w


class AdaLossFunction(KnnLossFunction):
    def __init__(self):
        """Good old Ada loss, implemented as version of KnnLostFunction """
        KnnLossFunction.__init__(self, None)

    def compute_parameters(self, trainX, trainY):
        return sparse.eye(len(trainX), len(trainX)), numpy.ones(len(trainX))


class DistanceBasedKnnFunction(KnnLossFunction):
    def __init__(self, uniform_variables, knn=None, distance_dependence=None, large_preds_penalty=0.,
                 row_normalize=False):
        """If knn is None, the matrix will be filled, otherwise it will be sparse
        with knn as number of nonzero cells,
        distance dependence is function, that takes distance between i-th and j-th
        events and returns a_ij
        """
        self.knn = knn
        self.distance_dependence = distance_dependence
        self.large_pred_penalty = large_preds_penalty
        self.row_normalize = row_normalize
        KnnLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        for variable in self.uniform_variables:
            if variable not in trainX.columns:
                raise ValueError("Dataframe is missing %s column" % variable)

        if self.knn is None:
            A = pairwise_distances(trainX[self.uniform_variables])
            A = self.distance_dependence(A)
            A *= (trainY[:, numpy.newaxis] == trainY[numpy.newaxis, :])
        else:
            is_signal = trainY > 0.5
            # computing knn indices of same type
            uniforming_features_of_signal = numpy.array(trainX.ix[is_signal, self.uniform_variables])
            neighbours = NearestNeighbors(n_neighbors=self.knn, algorithm='kd_tree').fit(uniforming_features_of_signal)
            signal_distances, knn_signal_indices = neighbours.kneighbors(uniforming_features_of_signal)
            knn_signal_indices = numpy.where(is_signal)[0].take(knn_signal_indices)

            uniforming_features_of_bg = numpy.array(trainX.ix[~is_signal, self.uniform_variables])
            neighbours = NearestNeighbors(n_neighbors=self.knn, algorithm='kd_tree').fit(uniforming_features_of_bg)
            bg_distances, knn_bg_indices = neighbours.kneighbors(uniforming_features_of_bg)
            knn_bg_indices = numpy.where(~is_signal)[0].take(knn_bg_indices)

            signal_distances = self.distance_dependence(signal_distances.flatten())
            bg_distances = self.distance_dependence(bg_distances.flatten())

            signal_ind_ptr = numpy.arange(0, sum(is_signal) * self.knn + 1, self.knn)
            bg_ind_ptr = numpy.arange(0, sum(~is_signal) * self.knn + 1, self.knn)
            signal_column_indices = knn_signal_indices.flatten()
            bg_column_indices = knn_bg_indices.flatten()

            A_sig = sparse.csr_matrix(sparse.csr_matrix((signal_distances, signal_column_indices, signal_ind_ptr),
                                                        shape=(sum(is_signal), len(trainX))))
            A_bg = sparse.csr_matrix(sparse.csr_matrix((bg_distances, bg_column_indices, bg_ind_ptr),
                                                       shape=(sum(~is_signal), len(trainX))))

            A = sparse.vstack((A_sig, A_bg), format='csr')

        if self.row_normalize:
            from sklearn.preprocessing import normalize
            A = normalize(A, norm='l1', axis=1)

        return A, numpy.ones(A.shape[0])


def compute_efficiencies(mask, y_pred, sample_weight):
    """For each event computes it position among other events by prediction. """
    order = numpy.argsort(y_pred[mask])
    weights = sample_weight[mask][order]
    efficiencies = (numpy.cumsum(weights) - 0.5 * weights) / numpy.sum(weights)
    return efficiencies[numpy.argsort(order)]


def test_compute_efficiency(size=100):
    y_pred = numpy.random.random(size)
    mask = numpy.random.random(size) > 0.5
    effs = compute_efficiencies(mask, y_pred, sample_weight=numpy.ones(size))
    assert len(effs) == numpy.sum(mask)
    assert len(effs) == len(set(effs))
    assert numpy.all(effs[numpy.argsort(y_pred[mask])] == numpy.sort(effs))
    effs2 = compute_efficiencies(numpy.where(mask)[0], y_pred, sample_weight=numpy.ones(size))
    assert numpy.all(effs == effs2)
    print("Compute efficiency is ok")

test_compute_efficiency()


def exp_margin(margin):
    """ margin = - y_signed * y_pred """
    return numpy.exp(numpy.clip(margin, -1e5, 2))


class FlatnessLossFunction(LossFunction, BaseEstimator):
    def __init__(self, uniform_variables, bins=10, uniform_label=1, power=2., ada_coefficient=1.,
                 allow_wrong_signs=True, keep_debug_info=False):
        """
        This loss function contains separately penalty for non-flatness and ada_coefficient.
        The penalty for non-flatness is using bins.

        :type uniform_variables: the vars, along which we want to obtain uniformity
        :type bins: the number of bins along each axis
        :type uniform_label: int | list(int), the labels for which we want to obtain uniformity
        :type power: the loss contains the difference | F - F_bin |^p, where p is power
        :type ada_coefficient: coefficient of ada_loss added to this one. The greater the coefficient,
            the less we tend to uniformity.
        :type allow_wrong_signs: defines whether gradient may different sign from the "sign of class"
            (i.e. may have negative gradient on signal)
        """
        self.uniform_variables = uniform_variables
        self.bins = bins
        self.uniform_label = numpy.array([uniform_label]) if isinstance(uniform_label, numbers.Number)  \
            else numpy.array(uniform_label)
        self.power = power
        self.ada_coefficient = ada_coefficient
        self.allow_wrong_signs = allow_wrong_signs
        self.keep_debug_info = keep_debug_info
        LossFunction.__init__(self, 1)

    def fit(self, X, y, sample_weight=None):
        assert len(X) == len(y), 'The lengths are different'
        sample_weight = check_sample_weight(y,  sample_weight=sample_weight)

        self.group_indices = defaultdict(list)
        # The weight of bin is mean of weights inside bin
        self.group_weights = defaultdict(list)
        occurences = numpy.zeros(len(X), dtype=int)

        for label in self.uniform_label:
            group_indices = self.compute_groups_indices(X, y, sample_weight=sample_weight, label=label)
            # cleaning the bins - deleting tiny or empty groups, canonizing
            for indices in group_indices:
                if len(indices) < 5:
                    # ignoring very small groups
                    continue
                assert numpy.all((y == label)[indices])
                self.group_indices[label].append(numpy.array(indices))
                self.group_weights[label].append(numpy.mean(sample_weight[indices]))
                occurences[indices] += 1

        y = numpy.array(y, dtype=int)
        needed_indices = numpy.in1d(y, self.uniform_label)
        out_of_bins = numpy.sum((occurences == 0) & needed_indices)
        if out_of_bins > 0.01 * len(X):
            print("warning: %i events are out of all bins" % out_of_bins)

        self.sample_weight = sample_weight
        self.event_weights = sample_weight / (occurences + 1e-10)
        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)
        return self

    def compute_groups_indices(self, X, y, sample_weight, label):
        """Returns a list, each element is events' indices in some group."""
        mask = y == label
        bin_limits = []
        for var in self.uniform_variables:
            bin_limits.append(numpy.linspace(numpy.min(X[var][mask]), numpy.max(X[var][mask]), 2 * self.bins + 1))
        result = list()
        for shift in [0, 1]:
            bin_limits2 = []
            for axis_limits in bin_limits:
                bin_limits2.append(axis_limits[1 + shift:-1:2])
            bin_indices = reports.compute_bin_indices(X, self.uniform_variables, bin_limits2)
            result += reports.bin_to_group_indices(bin_indices, mask=mask)
        return result

    def __call__(self, y, pred):
        # computing the common distribution of signal
        # taking only signal by now
        # this is approximate computation!
        # TODO reimplement, this is wrong implementation
        pred = numpy.ravel(pred)
        loss = 0

        for label in self.uniform_label:
            needed_indices = y == label
            sorted_pred = numpy.sort(pred[needed_indices])

            for bin_weight, indices_in_bin in zip(self.group_weights[label], self.group_indices[label]):
                probs_in_bin = numpy.take(pred, indices_in_bin)
                probs_in_bin = numpy.sort(probs_in_bin)
                positions = numpy.searchsorted(sorted_pred, probs_in_bin)
                global_effs = positions / float(len(sorted_pred))
                local_effs = (numpy.arange(0, len(probs_in_bin)) + 0.5) / len(probs_in_bin)
                bin_loss = numpy.sum((global_effs - local_effs) ** self.power)
                loss += bin_loss * bin_weight

        # Ada loss now
        loss += self.ada_coefficient * numpy.sum(numpy.exp(-y * pred))
        return loss

    def negative_gradient(self, y, y_pred, **kw_args):
        y_pred = numpy.ravel(y_pred)
        neg_gradient = numpy.zeros(len(y))

        for label in self.uniform_label:
            label_mask = y == label
            global_efficiencies = numpy.zeros(len(y_pred), dtype=float)
            global_efficiencies[label_mask] = compute_efficiencies(label_mask, y_pred, sample_weight=self.sample_weight)

            for bin_weight, indices_in_bin in zip(self.group_weights[label], self.group_indices[label]):
                assert numpy.all(label_mask[indices_in_bin]), "TODO delete"
                local_effs = compute_efficiencies(indices_in_bin, y_pred, sample_weight=self.sample_weight)
                global_effs = global_efficiencies[indices_in_bin]
                bin_gradient = self.power * numpy.sign(local_effs - global_effs) \
                               * numpy.abs(local_effs - global_effs) ** (self.power - 1)

                # TODO multiply by derivative of F_global ?
                neg_gradient[indices_in_bin] += bin_weight * bin_gradient

        assert numpy.all(neg_gradient[~numpy.in1d(y, self.uniform_label)] == 0)

        y_signed = 2 * y - 1
        if self.keep_debug_info:
            self.debug_dict['pred'].append(numpy.copy(y_pred))
            self.debug_dict['fl_grad'].append(numpy.copy(neg_gradient))
            self.debug_dict['ada_grad'].append(y_signed * self.sample_weight * numpy.exp(- y_signed * y_pred))

        # adding ada
        neg_gradient += self.ada_coefficient * y_signed * self.sample_weight \
                        * exp_margin(-self.ada_coefficient * y_signed * y_pred)

        if not self.allow_wrong_signs:
            neg_gradient = y_signed * numpy.clip(y_signed * neg_gradient, 0, 1e5)

        return neg_gradient

    # def update_terminal_regions(self, tree, X, y, residual, y_pred, sample_mask, learning_rate=1.0, k=0):
        # the standard version is used

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, pred):
        # terminal_region = numpy.where(terminal_regions == leaf)[0]
        tree.value[leaf, 0, 0] = numpy.clip(tree.value[leaf, 0, 0], -10, 10)
        # TODO think of real minimization

    def init_estimator(self, X=None, y=None):
        return LogOddsEstimator()


class NewRF(RandomForestRegressor):
    """Just a random forest regressor, that returns a two-dimensional array"""
    def predict(self, X):
        return RandomForestRegressor.predict(self, X)[:, numpy.newaxis]


class NewFlatnessLossFunction(FlatnessLossFunction, BaseEstimator):
    def __init__(self, uniform_variables, n_neighbours=100, uniform_label=1,  ada_coefficient=1.,
                 allow_wrong_signs=True, keep_debug_info=False, uniforming_factor=1., update_tree=True):
        """
        :param int|list[int] uniform_label: labels of classes for which the uniformity of predictions is desired
        """
        self.uniform_variables = uniform_variables
        self.n_neighbours = n_neighbours
        self.uniform_label = numpy.array([uniform_label]) if isinstance(uniform_label, numbers.Number)  \
            else numpy.array(uniform_label)
        self.ada_coefficient = ada_coefficient
        self.allow_wrong_signs = allow_wrong_signs
        self.keep_debug_info = keep_debug_info
        self.uniforming_factor = uniforming_factor
        self.update_tree = update_tree
        LossFunction.__init__(self, 1)

    def fit(self, X, y, sample_weight=None):
        assert len(X) == len(y), 'The lengths are different'
        # sample_weight = check_sample_weight(y,  sample_weight=sample_weight)
        y = column_or_1d(y)
        assert set(y) == {0,1}, "Only two classes are supported, their labels should be 0 and 1"
        self.knn_indices = defaultdict(list)
        for label in self.uniform_label:
            label_mask = y == label
            knn_indices = commonutils.computeSignalKnnIndices(self.uniform_variables, X, label_mask, n_neighbors=self.n_neighbours)
            # taking only rows, corresponding to this class
            self.knn_indices[label] = knn_indices[label_mask, :]

        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)
        return self

    def __call__(self, y, pred):
        return 1

    def init_estimator(self, X=None, y=None):
        return NewRF()

    def negative_gradient(self, y, y_pred, sample_weight=None, **kw_args):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        y_pred = numpy.ravel(y_pred)
        neg_gradient = numpy.zeros(len(y))

        for label in self.uniform_label:
            label_mask = y == label
            assert sum(label_mask) == len(self.knn_indices[label])
            # global_efficiencies = numpy.zeros(len(y_pred), dtype=float)
            # global_efficiencies[label_mask] = compute_efficiencies(label_mask, y_pred, sample_weight=self.sample_weight)
            values = y_pred[label_mask]
            knn_values = numpy.take(y_pred, self.knn_indices[label])
            knn_weights = numpy.take(sample_weight, self.knn_indices[label])
            # TODO use heaviside here?
            local_efficiencies = numpy.average(knn_values > values[:, numpy.newaxis], axis=1, weights=knn_weights)
            global_targets = commonutils.weighted_percentile(values, local_efficiencies,
                                                             sample_weight=sample_weight[label_mask])

            neg_gradient[label_mask] += self.uniforming_factor * (global_targets - values)

        assert numpy.all(neg_gradient[~numpy.in1d(y, self.uniform_label)] == 0)

        y_signed = 2 * y - 1
        if self.keep_debug_info:
            self.debug_dict['pred'].append(numpy.copy(y_pred))
            self.debug_dict['fl_grad'].append(numpy.copy(neg_gradient))
            self.debug_dict['ada_grad'].append(y_signed * sample_weight * numpy.exp(- y_signed * y_pred))

        # adding ada
        neg_gradient += self.ada_coefficient * y_signed * sample_weight \
                        * exp_margin(- self.ada_coefficient * y_signed * y_pred)

        if not self.allow_wrong_signs:
            neg_gradient = y_signed * numpy.clip(y_signed * neg_gradient, 0, 1e5)

        return neg_gradient

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, pred):
        if not self.update_tree:
            return
        terminal_region = numpy.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        tree.value[leaf, 0, 0] = numpy.median(residual)


class MyGradientBoostingClassifier(GBClassifier):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                 max_depth=3, init=None, random_state=None,
                 max_features=None, verbose=0, train_variables=None):
        """
        GradientBoosting from sklearn, which is modified to work with KnnLossFunction and it's versions.
        Train variables are variables used in training trees.

        :param LossFunction|str loss:
        """
        self.train_variables = train_variables
        GBClassifier.__init__(self, loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, init=init, random_state=random_state, max_features=max_features, verbose=verbose)

    def get_train_variables(self, X):
        if self.train_variables is None:
            return X
        else:
            return X[self.train_variables]

    def fit(self, X, y, sample_weight=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes
            ``0, 1, ..., n_classes_-1``

        sample_weight: array-like, shape = [n_samples], default None,
            positive weights if they are needed

        Returns
        -------
        self : object
            Returns self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = numpy.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        assert self.n_classes_ == 2, "at this moment only two-class classification is supported"

        self._check_params()
        # fitting the loss if it needs
        if isinstance(self.loss_, KnnLossFunction) or isinstance(self.loss_, FlatnessLossFunction):
            self.loss_.fit(X, y)
        X = self.get_train_variables(X)

        # Check input
        X, = check_arrays(X, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        n_samples, n_features = X.shape
        self.n_features = n_features
        random_state = check_random_state(self.random_state)

        # pull frequently used parameters into local scope
        subsample = self.subsample
        do_oob = subsample < 1.0

        # allocate model state data structures
        self.estimators_ = numpy.empty((self.n_estimators, self.loss_.K), dtype=numpy.object)
        self.train_score_ = numpy.zeros((self.n_estimators,), dtype=numpy.float64)

        sample_mask = numpy.ones((n_samples,), dtype=numpy.bool)
        n_inbag = max(1, int(subsample * n_samples))

        if self.verbose:
            # header fields and line format str
            header_fields = ['Iter', 'Train Loss']
            verbose_fmt = ['{iter:>10d}', '{train_score:>16.4f}']
            if do_oob:
                header_fields.append('OOB Improve')
                verbose_fmt.append('{oob_impr:>16.4f}')
            header_fields.append('Remaining Time')
            verbose_fmt.append('{remaining_time:>16s}')
            verbose_fmt = ' '.join(verbose_fmt)
            # print the header line
            print(('%10s ' + '%16s ' * (len(header_fields) - 1)) % tuple(header_fields))
            # plot verbose info each time i % verbose_mod == 0
            verbose_mod = 1
            start_time = time()

        # fit initial model
        self.init_.fit(X, y)

        # init predictions
        y_pred = self.init_.predict(X)

        # perform boosting iterations
        for i in range(self.n_estimators):
            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)

            # fit next stage of tree
            args = {}
            # TODO write own gradient boosting
            if sklearn.__version__ >= '0.15':
                args = {'criterion': 'mse', 'splitter': 'best', }
            y_pred = self._fit_stage(i, X, y, y_pred=y_pred, sample_mask=sample_mask, random_state=random_state, **args)

            self.train_score_[i] = self.loss_(y, y_pred)

            if self.verbose > 0:
                if (i + 1) % verbose_mod == 0:
                    remaining_time = (self.n_estimators - (i + 1)) * (time() - start_time) / float(i + 1)
                    if remaining_time > 60:
                        remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
                    else:
                        remaining_time = '{0:.2f}s'.format(remaining_time)
                    print(verbose_fmt.format(iter=i + 1,
                                             train_score=self.train_score_[i],
                                             remaining_time=remaining_time))
                if self.verbose == 1 and ((i + 1) // (verbose_mod * 10) > 0):
                    # adjust verbose frequency (powers of 10)
                    verbose_mod *= 10

        return self

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        # everything connected with loss was moved to self.fit
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0)")

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

    def predict(self, X):
        return GBClassifier.predict(self, self.get_train_variables(X))

    def predict_proba(self, X):
        return GBClassifier.predict_proba(self, self.get_train_variables(X))

    def staged_predict_proba(self, X):
        return GBClassifier.staged_predict_proba(self, self.get_train_variables(X))


def test_gradient(loss, size=1000):
    X, y = commonutils.generate_sample(size, 10)
    loss.fit(X, y)
    pred = numpy.random.random(size)
    epsilon = 1e-7
    val = loss(y, pred)
    gradient = numpy.zeros_like(pred)

    for i in range(size):
        pred2 = pred.copy()
        pred2[i] += epsilon
        val2 = loss(y, pred2)
        gradient[i] = (val2 - val) / epsilon

    n_gradient = loss.negative_gradient(y, pred)
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
    loss2 = PairwiseKnnLossFunction(uniform_variables, knn=10)
    loss3 = AdaLossFunction()
    loss4 = RandomKnnLossFunction(uniform_variables, samples * 2, knn=5, knn_factor=3)
    loss5 = DistanceBasedKnnFunction(uniform_variables, knn=10, distance_dependence=lambda r: numpy.exp(-0.1 * r))
    loss6 = FlatnessLossFunction(uniform_variables, ada_coefficient=0.5)
    loss7 = FlatnessLossFunction(uniform_variables, ada_coefficient=0.5, uniform_label=[0,1])
    loss8 = NewFlatnessLossFunction(uniform_variables, ada_coefficient=0.5, uniform_label=1)
    loss9 = NewFlatnessLossFunction(uniform_variables, ada_coefficient=0.5, uniform_label=[0, 1])

    for loss in [loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9]:
        result = MyGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=.2,
                                              subsample=0.7, n_estimators=n_estimators, train_variables=None)\
            .fit(trainX[:samples], trainY[:samples]).score(testX, testY)
        assert result >= 0.7, "The quality is too poor: %.3f" % result

    # TODO return this code and test losses
    # for loss in [loss1, loss2, loss3, loss4, loss5]:
    #     testGradient(loss)

    print('uniform gradient boosting is ok')

test_gradient_boosting()

