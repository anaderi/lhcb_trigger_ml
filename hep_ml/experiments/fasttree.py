"""
This is fast version of DecisionTreeRegressor for only one target function.
(This is the most simple case, but even multi-class boosting doesn't need more complicated things)

I need numpy implementation mostly for further experiments, rather than for real speedup.
This tree shouldn't be used by itself, only in boosting techniques
"""
from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import numpy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.utils.validation import check_random_state


__author__ = 'Alex Rogozhnikov'


# Criterion is minimized in tree

class MseCriterion(object):
    @staticmethod
    def compute_best_splits(data, y, sample_weight):
        orders = numpy.argsort(data, axis=0)
        answers = y[orders]
        weights = sample_weight[orders]
        left_sum, right_sum = _compute_cumulative_sums(answers * weights)
        left_weights, right_weights = _compute_cumulative_sums(weights + 1e-20)
        # mse = left_sq + right_sq - left_sum ** 2 / left_weights - right_sum ** 2 / right_weights
        # one can see that left_sq + right_sq is constant, and can be omitted, so we have:
        costs = - (left_sum ** 2 / left_weights + right_sum ** 2 / right_weights)
        return _compute_cuts_costs_positions(costs, data=data, orders=orders)


class FriedmanMseCriterion(object):
    @staticmethod
    def compute_best_splits(data, y, sample_weight):
        orders = numpy.argsort(data, axis=0)
        answers = y[orders]
        weights = sample_weight[orders]
        left_sum, right_sum = _compute_cumulative_sums(answers * weights)
        left_weights, right_weights = _compute_cumulative_sums(weights + 1e-50)
        diff = left_sum / left_weights - right_sum / right_weights
        # improvement = n_left * n_right * diff ^ 2 / (n_left + n_right)
        costs = - left_weights * right_weights * (diff ** 2)
        return _compute_cuts_costs_positions(costs, data=data, orders=orders)


class PValueCriterion(object):
    @staticmethod
    def compute_best_splits(data, y, sample_weight):
        y_order = numpy.argsort(numpy.argsort(y))
        # converting to [-1, 1]
        y_order = numpy.linspace(-1, 1, len(y_order))[y_order]
        orders = numpy.argsort(data, axis=0)
        # answers = y[orders]
        pred_orders = y_order[orders]
        weights = sample_weight[orders]
        left_sum, right_sum = _compute_cumulative_sums(pred_orders * weights)

        left_weights, right_weights = _compute_cumulative_sums(weights + 1e-50)
        regularization = 0.01 * numpy.sum(sample_weight)
        # mean = 0, var = left_weights * right_weights
        costs = - numpy.abs(left_sum) / len(y)
        costs /= numpy.sqrt((left_weights + regularization) * (right_weights + regularization))
        return _compute_cuts_costs_positions(costs, data=data, orders=orders)


class AbstractClassificationCriterion(object):
    @staticmethod
    def compute_costs(s_left, s_right, b_left, b_right):
        raise NotImplementedError('Should be overloaded')

    @classmethod
    def compute_best_splits(cls, data, y, sample_weight):
        orders = numpy.argsort(data, axis=0)
        answers = y[orders]
        weights = sample_weight[orders]
        pos_answers = answers * (answers > 0)
        neg_answers = - answers * (answers < 0)
        left_pos_sum, right_pos_sum = _compute_cumulative_sums(pos_answers * weights)
        left_neg_sum, right_neg_sum = _compute_cumulative_sums(neg_answers * weights)
        # using passed function to compute criterion
        costs = cls.compute_costs(left_pos_sum, right_pos_sum, left_neg_sum, right_neg_sum)
        return _compute_cuts_costs_positions(costs, data=data, orders=orders)


class GiniCriterion(AbstractClassificationCriterion):
    @staticmethod
    def compute_costs(s_left, s_right, b_left, b_right):
        return s_left * b_left / (s_left + b_left) + s_right * b_right / (s_right + b_right)


class EntropyCriterion(AbstractClassificationCriterion):
    @staticmethod
    def compute_costs(s_left, s_right, b_left, b_right):
        reg = 10
        total_left = s_left + b_left
        total_right = s_right + b_right
        return -(
            s_left * numpy.log(s_left + reg)
            + s_right * numpy.log(s_right + reg)
            + b_left * numpy.log(b_left + reg)
            + b_right * numpy.log(b_right + reg)
            - total_left * numpy.log(total_left + reg)
            - total_right * numpy.log(total_right + reg)
        )


class SignificanceCriterion(AbstractClassificationCriterion):
    @staticmethod
    def compute_costs(s_left, s_right, b_left, b_right):
        return - (s_left ** 2 / (b_left + 5) + s_right ** 2 / (b_right + 5))


class SymmetricSignificanceCriterion(AbstractClassificationCriterion):
    @staticmethod
    def compute_costs(s_left, s_right, b_left, b_right):
        reg = 10
        result = s_left ** 2 / (b_left + reg) + s_right ** 2 / (b_right + reg)
        result += b_left ** 2 / (s_left + reg) + b_right ** 2 / (s_right + reg)
        return -result


class PoissonSignificanceCriterion(AbstractClassificationCriterion):
    @staticmethod
    def compute_costs(s_left, s_right, b_left, b_right):
        reg = 20
        result = (s_left - b_left) * numpy.log((s_left + reg) / (b_left + reg))
        result += (s_right - b_right) * numpy.log((s_right + reg) / (b_right + reg))
        return -result


def _compute_cumulative_sums(values):
    # for each feature and for each split computes cumulative sums
    left = numpy.cumsum(values, axis=0)
    right = left[[-1], :] - left
    return left[:-1, :], right[:-1, :]


def _compute_cuts_costs_positions(costs, data, orders):
    """
    Supplementary function to compute things that criterion should return.

    :param costs: [n_samples, n_features]
    :param data: [n_samples, n_features] original dataset
    :param orders: [n_samples, n_features] ordering of values for each feature
    """
    optimal_costs = numpy.min(costs, axis=0)
    optimal_sorted_positions = numpy.argmin(costs, axis=0)
    features_index = numpy.arange(costs.shape[1])
    assert (optimal_costs == costs[optimal_sorted_positions, features_index]).all()
    optimal_orders = orders[optimal_sorted_positions, features_index]
    _next_optimal_orders = orders[optimal_sorted_positions + 1, features_index]
    optimal_cuts = (data[optimal_orders, features_index] + data[_next_optimal_orders, features_index]) / 2.
    return optimal_cuts, optimal_costs, optimal_sorted_positions


criterions = {'mse': MseCriterion,
              'fmse': FriedmanMseCriterion,
              'friedman-mse': FriedmanMseCriterion,
              'pvalue': PValueCriterion,
              'significance': SignificanceCriterion,
              'significance2': SymmetricSignificanceCriterion,
              'gini': GiniCriterion,
              'entropy': EntropyCriterion,
              'poisson': PoissonSignificanceCriterion
}


class FastTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 max_depth=5,
                 max_features=None,
                 min_samples_split=40,
                 max_events_used=1000,
                 criterion='mse',
                 random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_events_used = max_events_used
        self.criterion = criterion
        self.random_state = random_state
        # keeps the indices of features and the values at which we split them.
        # dict{node_index -> (feature_index, split_value) or (leaf_value)}
        # Node index is defined as:
        # left: `0`bit, right: `1`bit.
        # Indices of some nodes: root - 1b, left child of root: 10b, it's right child: 101b.
        self.nodes_data = dict()

    def print_tree_stats(self):
        print(len(self.nodes_data), ' nodes in tree')
        leaves = [k for k, v in self.nodes_data.items() if len(v) == 1]
        print(len(leaves), ' leaf nodes in tree')

    def print_tree(self, node_index=1, prefix=''):
        data = self.nodes_data[node_index]
        print(prefix, data)
        if len(data) > 1:
            left, right = self._children(node_index)
            self.print_tree(left, "  " + prefix)
            self.print_tree(right, "  " + prefix)

    @staticmethod
    def _children(node_index):
        left_node = 2 * node_index
        right_node = 2 * node_index + 1
        return left_node, right_node

    def _fit_tree_node(self, X, y, w, node_index, depth, passed_indices):
        """Recursive function to fit tree, rather simple implementation"""
        if len(passed_indices) <= self.min_samples_split or depth >= self.max_depth:
            self.nodes_data[node_index] = (numpy.average(y[passed_indices], weights=w[passed_indices]), )
            return

        selected_events = passed_indices
        if len(passed_indices) > self.max_events_used:
            selected_events = self.random_state.choice(passed_indices, size=self.max_events_used, replace=True)

        selected_features = self.random_state.choice(self.n_features, size=self._n_used_features, replace=False)
        cuts, costs, _ = self._criterion.compute_best_splits(
            X[numpy.ix_(selected_events, selected_features)], y[selected_events], sample_weight=w[selected_events])

        # feature that showed best pre-estimated cost
        feature_index = selected_features[numpy.argmin(costs)]
        split = cuts[numpy.argmin(costs)]
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

    def fast_apply(self, X):
        """Same as previous function, but uses special structures and vectorization """
        assert isinstance(X, numpy.ndarray), 'X should be numpy.array'
        # vertices in tree are enumerated (this time in dense way without binary notation), root is 0.
        ordered_nodes = OrderedDict(self.nodes_data)
        n_nodes = len(ordered_nodes)
        old2new_index = dict(zip(ordered_nodes.keys(), range(n_nodes)))
        child_indices = numpy.zeros(2 * n_nodes, dtype=int)
        features = numpy.zeros(n_nodes, dtype=int)
        splits = numpy.zeros(n_nodes, dtype=float)
        leaf_values = numpy.zeros(n_nodes, dtype=float)

        for i, (node_index, node_data) in enumerate(ordered_nodes.items()):
            if len(node_data) == 1:
                leaf_values[i] = node_data[0]
                child_indices[2 * i] = i
                child_indices[2 * i + 1] = i
            else:
                feature_index, split = node_data
                features[i] = feature_index
                splits[i] = split
                left, right = self._children(node_index)
                child_indices[2 * i] = old2new_index[left]
                child_indices[2 * i + 1] = old2new_index[right]

        rows = numpy.arange(len(X))
        leaf_indices = numpy.zeros(len(X), dtype=int)
        for _ in range(self.max_depth + 1):
            passed = X[rows, features[leaf_indices]] > splits[leaf_indices]
            leaf_indices = child_indices[2 * leaf_indices + passed]

        return leaf_indices, leaf_values[leaf_indices]

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


class FastNeuroTreeRegressor(FastTreeRegressor):
    def __init__(self,
                 max_depth=5,
                 max_features=None,
                 min_samples_split=40,
                 min_samples_leaf=10,
                 n_lincomb=2,
                 n_events_form_lincomb=50,
                 max_events_used=1000,
                 criterion='mse',
                 random_state=None):
        self.min_samples_leaf = min_samples_leaf
        self.n_lincomb = n_lincomb
        self.n_events_form_lincomb = n_events_form_lincomb
        FastTreeRegressor.__init__(self,
                                   max_depth=max_depth,
                                   max_features=max_features,
                                   min_samples_split=min_samples_split,
                                   max_events_used=max_events_used,
                                   criterion=criterion,
                                   random_state=random_state)

    def _fit_tree_node(self, X, y, w, node_index, depth, passed_indices):
        """Recursive function to fit tree, rather simple implementation"""
        if len(passed_indices) <= self.min_samples_split or depth >= self.max_depth:
            self.nodes_data[node_index] = (numpy.average(y[passed_indices], weights=w[passed_indices]), )
            return

        selected_events = passed_indices
        if len(passed_indices) > self.max_events_used:
            selected_events = self.random_state.choice(passed_indices, size=self.max_events_used, replace=True)

        candidate_features = self.random_state.choice(self.n_features, replace=True,
                                                      size=[self._n_used_features, self.n_lincomb])
        formed_data = numpy.zeros([len(selected_events), self._n_used_features])
        candidate_lincomb_coefficients = numpy.zeros_like(candidate_features, dtype=float)
        for i, lincomb_features in enumerate(candidate_features):
            pre_events_used = selected_events[:self.n_events_form_lincomb]
            data = X[numpy.ix_(pre_events_used, lincomb_features)]
            w_used = w[pre_events_used]
            lr = LinearRegression().fit(data * w_used[:, numpy.newaxis], y[pre_events_used] * w_used)
            # normalizing coeffs
            coeffs = lr.coef_
            coeffs /= numpy.abs(coeffs).sum() + 0.01

            candidate_lincomb_coefficients[i, :] = coeffs
            formed_data[:, i] = self._compute_lincomb(X, indices=selected_events,
                                                      lincomb_features=lincomb_features,
                                                      lincomb_coefficients=coeffs)

        cuts, costs, _ = self._criterion.compute_best_splits(
            formed_data, y[selected_events], sample_weight=w[selected_events])

        # feature that showed best pre-estimated cost
        combination_index = numpy.argmin(costs)
        split = cuts[combination_index]
        lincomb_features = candidate_features[combination_index, :]
        lincomb_coefficients = candidate_lincomb_coefficients[combination_index, :]
        # computing information for (possible) children
        lincomb_values = self._compute_lincomb(X, indices=passed_indices,
                                               lincomb_features=lincomb_features,
                                               lincomb_coefficients=lincomb_coefficients)

        passed_left_subtree = passed_indices[lincomb_values <= split]
        passed_right_subtree = passed_indices[lincomb_values > split]
        left, right = self._children(node_index)
        if len(passed_left_subtree) < self.min_samples_leaf or len(passed_right_subtree) < self.min_samples_leaf:
            # this will be leaf
            self.nodes_data[node_index] = (numpy.average(y[passed_indices], weights=w[passed_indices]), )
        else:
            # non-leaf, recurrent calls
            self.nodes_data[node_index] = (lincomb_features, lincomb_coefficients, split)
            self._fit_tree_node(X, y, w, left, depth + 1, passed_left_subtree)
            self._fit_tree_node(X, y, w, right, depth + 1, passed_right_subtree)

    def _compute_lincomb(self, X, indices, lincomb_features, lincomb_coefficients):
        result = numpy.zeros(len(indices))
        for feature, coeff in zip(lincomb_features, lincomb_coefficients):
            result += X[indices, feature] * coeff
        return result

    def _apply_node(self, X, leaf_indices, predictions, node_index, passed_indices):
        """Recursive function to compute the index """
        node_data = self.nodes_data[node_index]
        if len(node_data) == 1:
            # leaf node
            leaf_indices[passed_indices] = node_index
            predictions[passed_indices] = node_data[0]
        else:
            # non-leaf
            lincomb_features, lincomb_coefficients, split = node_data
            lincomb_values = self._compute_lincomb(X, indices=passed_indices,
                                                   lincomb_features=lincomb_features,
                                                   lincomb_coefficients=lincomb_coefficients)
            passed_left_subtree = passed_indices[lincomb_values <= split]
            passed_right_subtree = passed_indices[lincomb_values > split]
            left, right = self._children(node_index)
            self._apply_node(X, leaf_indices, predictions, left, passed_left_subtree)
            self._apply_node(X, leaf_indices, predictions, right, passed_right_subtree)

