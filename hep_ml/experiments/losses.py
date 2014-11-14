from __future__ import print_function, division, absolute_import

import numpy
import scipy.sparse as sparse
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.unsupervised import NearestNeighbors

from .. import commonutils
from ..commonutils import check_sample_weight
from hep_ml.commonutils import check_uniform_label
from hep_ml.losses import SimpleKnnLossFunction, KnnFlatnessLossFunction
from hep_ml.ugradientboosting import AbstractMatrixLossFunction


__author__ = 'Alex Rogozhnikov'

# TODO updating tree in FL and NFL


# Descendants of KnnLossFunction - particular cases, each has its own
# algorithm of generating A and w


class ExperimentalSimpleKnnLossFunction(SimpleKnnLossFunction):
    def __init__(self, uniform_variables, knn=10, uniform_label=1, row_norm=1., diagonal=0., distinguish_classes=True):
        """A matrix is square, each row corresponds to a single event in train dataset, in each row we put ones
        to the closest neighbours of that event if this event from class along which we want to have uniform prediction.
        :param list[str] uniform_variables: the features, along which uniformity is desired
        :param int knn: the number of nonzero elements in the row, corresponding to event in 'uniform class'
        :param int|list[int] uniform_label: the label (labels) of 'uniform classes'
        :param int diagonal: float, A + diagonal * Identity is used.
        """
        self.knn = knn
        self.row_norm = row_norm
        self.diagonal = diagonal
        self.uniform_label = check_uniform_label(uniform_label)
        self.distinguish_classes = distinguish_classes
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        n_samples = len(trainX)
        A_parts = []
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
            A_parts.append(A_part)

        for label in set(trainY).difference(self.uniform_label):
            label_mask = trainY == label
            n_label = numpy.sum(label_mask)
            ind_ptr = numpy.arange(0, n_label + 1)
            column_indices = numpy.where(label_mask)[0].flatten()
            data = numpy.ones(n_label, dtype=float) * self.row_norm
            A_part = sparse.csr_matrix((data, column_indices, ind_ptr), shape=[n_label, len(trainX)])
            A_parts.append(A_part)

        A = sparse.vstack(A_parts, format='csr', dtype=float) + sparse.eye(n_samples) * self.diagonal
        return A, numpy.ones(len(trainX))


class SimpleKnnLossFunctionEyeBg(AbstractMatrixLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for signal.
        For background we have identity matrix.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

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


class SimpleKnnLossFunctionKnnOnDiagonalSignal(AbstractMatrixLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for signal. For background we
        have identity matrix times self.knn.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

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


class SimpleKnnLossFunctionKnnOnDiagonalBg(AbstractMatrixLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for signal. For background we
        have identity matrix times self.knn.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

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


class SimpleKnnLossFunctionEyeSignal(AbstractMatrixLossFunction):
    def __init__(self, uniform_variables, knn=5, distinguish_classes=True, diagonal=0.):
        """A matrix is square, each row corresponds to a single event in train dataset,
        in each row we put ones to the closest neighbours of that event for background.
        For signal we have identity matrix.

        If distinguish_classes==True, only events of the same class are chosen.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.diagonal = diagonal
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

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


class PairwiseKnnLossFunction(AbstractMatrixLossFunction):
    def __init__(self, uniform_variables, knn, exclude_self=True, penalize_large_preds=True):
        """ A is rectangular matrix, in each row we have only two '1's,
        all other elements are zeros, these two '1's are placed in the columns, corresponding to neighbours
        exclude_self: bool, exclude self from knn?
        """
        self.knn = knn
        self.exclude_self = exclude_self
        self.penalize_large_preds = penalize_large_preds
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY):
        is_signal = trainY > 0.5
        knn = self.knn
        if self.exclude_self:
            knn_indices = \
                commonutils.computeKnnIndicesOfSameClass(self.uniform_variables, trainX, is_signal, knn + 1)[:, 1:]
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


class RandomKnnLossFunction(AbstractMatrixLossFunction):
    def __init__(self, uniform_variables, n_rows, knn=5, knn_factor=3, large_preds_penalty=1.):
        """A general loss,
        at each iteration it takes some random event from train dataset,
        and selects randomly knn of its knn*knn_factor neighbours, the process is repeated 'n_rows' times"""
        self.n_rows = n_rows
        self.knn = knn
        self.knn_factor = knn_factor
        self.large_preds_penalty = large_preds_penalty
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

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


class DistanceBasedKnnFunction(AbstractMatrixLossFunction):
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
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

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


class NewRF(RandomForestRegressor):
    """Just a random forest regressor, that returns a two-dimensional array"""

    def predict(self, X):
        return RandomForestRegressor.predict(self, X)[:, numpy.newaxis]


class NewFlatnessLossFunction(KnnFlatnessLossFunction):
    def __init__(self, uniform_variables, n_neighbours=100, uniform_label=1, ada_coefficient=1.,
                 allow_wrong_signs=True, power=2.,
                 keep_debug_info=False, uniforming_factor=1.):
        KnnFlatnessLossFunction.__init__(uniform_variables=uniform_variables,
                                         uniform_label=uniform_label,
                                         power=power,
                                         ada_coefficient=ada_coefficient,
                                         allow_wrong_signs=allow_wrong_signs,
                                         n_neighbours=n_neighbours,
                                         use_median=False,
                                         keep_debug_info=keep_debug_info,
        )
        self.uniforming_factor = uniforming_factor

    def negative_gradient(self, y_pred):
        sample_weight = check_sample_weight(self.y, sample_weight=self.sample_weight)
        y_pred = numpy.ravel(y_pred)
        neg_gradient = numpy.zeros(len(self.y))

        for label in self.uniform_label:
            label_mask = self.y == label
            assert sum(label_mask) == len(self.knn_indices[label])
            values = y_pred[label_mask]
            knn_values = numpy.take(y_pred, self.knn_indices[label])
            knn_weights = numpy.take(sample_weight, self.knn_indices[label])
            # TODO use smoothing here?
            local_efficiencies = numpy.average(knn_values > values[:, numpy.newaxis], axis=1, weights=knn_weights)
            global_targets = commonutils.weighted_percentile(values, local_efficiencies,
                                                             sample_weight=sample_weight[label_mask])

            neg_gradient[label_mask] += self.uniforming_factor * (global_targets - values) ** (self.power - 1)

        assert numpy.all(neg_gradient[~numpy.in1d(self.y, self.uniform_label)] == 0)

        y_signed = self.y_signed
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
