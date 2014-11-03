from __future__ import print_function
from __future__ import division

from collections import defaultdict
import numbers

import scipy.sparse as sparse
from sklearn.base import BaseEstimator
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import LogOddsEstimator
import numpy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.utils.validation import column_or_1d

from .. import commonutils, reports, metrics
from ..commonutils import check_sample_weight
from ..ugradientboosting import AbstractLossFunction, KnnLossFunction, compute_positions


__author__ = 'Alex Rogozhnikov'

# TODO updating tree in FL and NFL


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



def exp_margin(margin):
    """ margin = - y_signed * y_pred """
    return numpy.exp(numpy.clip(margin, -1e5, 2))


class FlatnessLossFunction(AbstractLossFunction, BaseEstimator):
    def __init__(self, uniform_variables, bins=10, uniform_label=1, power=2., ada_coefficient=1.,
                 allow_wrong_signs=True, median=False, keep_debug_info=False):
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
        self.median = median
        AbstractLossFunction.__init__(self, 1)

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
            result += metrics.bin_to_group_indices(bin_indices, mask=mask)
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
        if self.median:
            terminal_region = numpy.where(terminal_regions == leaf)[0]
            residual = residual.take(terminal_region, axis=0)
            tree.value[leaf, 0, 0] = numpy.median(residual)
        else:
            tree.value[leaf, 0, 0] = numpy.clip(tree.value[leaf, 0, 0], -10, 10)

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
        AbstractLossFunction.__init__(self, 1)

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
