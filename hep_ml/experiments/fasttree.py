"""
This is fast version of DecisionTreeRegressor for only one target function.
(This is the most simple case, but even multi-class boosting doesn't need more complicated things)

I need numpy implementation mostly for further experiments, rather than for real speedup.
This tree shouldn't be used by itself, only in boosting techniques
"""
from __future__ import division, print_function, absolute_import
import numpy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_random_state
from hep_ml.commonutils import reorder_by_first


__author__ = 'Alex Rogozhnikov'


class MseCriterion(object):
    @staticmethod
    def pre_estimate(bin_indices, y, sample_weight):
        """estimates the lowest possible MSE we can achieve at this feature,
        it is used to select feature"""
        assert len(bin_indices) == len(y) == len(sample_weight)
        # computing weights to the left and right side of cut:
        left_weights, right_weights = MseCriterion._compute_split_sums(bin_indices, sample_weight)
        # computing means to the left and right side of cut:
        left_means, right_means = MseCriterion._compute_split_sums(bin_indices, y * sample_weight)
        # computing squares to the left and right side of cut:
        # left_sq, right_sq = MseCriterion._compute_split_sums(bin_indices, y ** 2 * sample_weight)
        # computing finally MSE for each split point
        # mse = left_sq + right_sq + left_mean ** 2 / left_weights + right_mean ** 2 / right_weights
        # one can see that left_sq + right_sq is constant, and can be omitted, so we have:
        return - numpy.max(left_means ** 2 / left_weights + right_means ** 2 / right_weights)

    @staticmethod
    def _compute_split_sums(bin_indices, values):
        """Provided an array of bin_indices and values returns
        two arrays of length n-1, where n is number of
        each one corresponds to a split of initial array (between to bins),
        left contains sum of `values` to the left of split value (in first k bins),
        right contains sum of `values` to the right of split value (in last n-k bins)
        """
        bin_sums = numpy.bincount(bin_indices, weights=values)
        left = numpy.cumsum(bin_sums[:-1])
        right = bin_sums[-1] - left
        return left, right

    @staticmethod
    def compute_split(feature_values, y, sample_weight):
        """computes optimal split for selected feature """
        # TODO greedy search here
        feature_values, y, sample_weight = reorder_by_first(feature_values, y, sample_weight)
        left_means = numpy.cumsum(y * sample_weight)
        right_means = left_means[-1] - left_means
        left_weights = numpy.cumsum(sample_weight) + 1e-10
        right_weights = left_weights[-1] - left_weights + 1e-10
        sde_values = left_means ** 2 / left_weights + right_means ** 2 / right_weights
        optimal_position = numpy.argmax(sde_values)
        optimal_cut = feature_values[optimal_position]
        # TODO return optimal SDE
        return optimal_cut


criterions = {'mse': MseCriterion}


class FastTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 max_depth=5,
                 max_features=None,
                 min_samples_split=40,
                 max_events_used=1000,
                 n_steps=10,
                 criterion='mse',
                 random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_events_used = max_events_used
        self.criterion = criterion
        self.random_state = random_state
        self.steps = n_steps
        # keeps the indices of features and the values at which we split them.
        # dict{node_index -> (feature_index, split_value) or (leaf_value)}
        # left: `0`bit, right: `1`bit.
        # Indices of some nodes: root - 1b, left-root-children: 10b, it's right child: 101b.
        self.nodes_data = dict()

    def print_tree_stats(self):
        print(len(self.nodes_data), ' nodes in tree')
        leaves = [k for k, v in self.nodes_data.items() if len(v) == 1]
        print(len(leaves), ' leaf nodes in tree')


    @staticmethod
    def _children(node_index):
        left_node = 2 * node_index
        right_node = 2 * node_index + 1
        return left_node, right_node

    def _fit_tree_node(self, X, y, w, node_index, depth, passed_indices):
        """Recursive function to fit tree, rather simple implementation"""
        # indices of events used in pre-estimation
        if len(passed_indices) <= self.min_samples_split or depth >= self.max_depth:
            self.nodes_data[node_index] = (numpy.average(y[passed_indices], weights=w[passed_indices]), )
            return

        pre_events = passed_indices
        if len(passed_indices) > self.max_events_used:
            pre_events = self.random_state.choice(passed_indices, size=self.max_events_used, replace=True)
        # should be minimal for good classification
        estimated_costs = numpy.zeros(self.n_features) + 1e20
        for feature_index in self.random_state.choice(self.n_features, size=self._n_used_features, replace=False):
            cut_events = self.random_state.choice(passed_indices, size=self.steps)
            cuts = numpy.sort(X[cut_events, feature_index])
            bin_indices = numpy.searchsorted(cuts, X[pre_events, feature_index])
            estimated_costs[feature_index] = \
                self._criterion.pre_estimate(bin_indices=bin_indices, y=y[pre_events], sample_weight=w[pre_events])

        # feature that showed best pre-estimated cost
        feature_index = numpy.argmin(estimated_costs)
        split = self._criterion.compute_split(X[pre_events, feature_index], y[pre_events], w[pre_events])
        # computing information for (possible) children
        passed_left_subtree = passed_indices[X[passed_indices, feature_index] <= split]
        passed_right_subtree = passed_indices[X[passed_indices, feature_index] > split]
        left, right = self._children(node_index)
        if len(passed_left_subtree) == 0 or len(passed_right_subtree) == 0:
            # this will be leaf
            self.nodes_data[node_index] = (numpy.average(y[passed_indices], weights=w[passed_indices]), )
        else:
            # non-leaf, recurrent calls
            self.nodes_data[node_index] = (feature_index, split)
            self._fit_tree_node(X, y, w, left, depth + 1, passed_left_subtree)
            self._fit_tree_node(X, y, w, right, depth + 1, passed_right_subtree)

    def _apply_node(self, X, leaf_indices, predictions, node_index, passed_indices):
        """Recursive function to compute the index """
        node_data = self.nodes_data[node_index]
        if len(node_data) == 1:
            # leaf node
            leaf_indices[passed_indices] = node_index
            predictions[passed_indices] = node_data[0]
        else:
            # non-leaf
            feature_index, split = node_data
            passed_left_subtree = passed_indices[X[passed_indices, feature_index] <= split]
            passed_right_subtree = passed_indices[X[passed_indices, feature_index] > split]
            left, right = self._children(node_index)
            self._apply_node(X, leaf_indices, predictions, left, passed_left_subtree)
            self._apply_node(X, leaf_indices, predictions, right, passed_right_subtree)

    def apply(self, X):
        """For each event returns the index of leaf that event belongs to"""
        assert isinstance(X, numpy.ndarray), 'X should be numpy.array'
        leaf_indices = numpy.zeros(len(X), dtype=int)
        predictions = numpy.zeros(len(X), dtype=float)
        # this function fills leaf_indices array
        self._apply_node(X, leaf_indices, predictions, node_index=1, passed_indices=numpy.arange(len(X)))
        return leaf_indices, predictions

    def fit(self, X, y, sample_weight, check_input=True):
        if check_input:
            assert isinstance(X, numpy.ndarray), "X should be numpy.array"
            assert isinstance(y, numpy.ndarray), "y should be numpy.array"
            assert isinstance(sample_weight, numpy.ndarray), "sample_weight should be numpy.array"
            assert len(X) == len(y) == len(sample_weight), 'Size of arrays is different'
        self.n_features = X.shape[1]
        if self.max_features is None:
            self._n_used_features = self.n_features
        else:
            self._n_used_features = min(self.n_features, self.max_features)
        self._criterion = criterions[self.criterion]
        self.random_state = check_random_state(self.random_state)
        self.nodes_data = dict()  # clearing previous fitting
        root_node_index = 1
        self._fit_tree_node(X=X, y=y, w=sample_weight, node_index=root_node_index, depth=0,
                            passed_indices=numpy.arange(len(X)))
        return self

    def predict(self, X):
        leaf_indices, predictions = self.apply(X)
        return predictions