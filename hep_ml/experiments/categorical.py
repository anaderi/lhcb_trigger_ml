from __future__ import division, print_function, absolute_import
from sklearn.utils.validation import check_arrays
from hep_ml.commonutils import check_sample_weight
from hep_ml.experiments import fasttree
import numpy
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from collections import OrderedDict
from hep_ml.losses import BinomialDevianceLossFunction
from scipy.special import expit

__author__ = 'Alex Rogozhnikov'


# All the classifiers in this file can only be trained on categorical features
# these are integer features, without senseless ordering
# examples of categorical features:
# particle type, user id, advertisement id.
# isn't intended for physical purposes


class CategoricalTreeRegressor(fasttree.FastTreeRegressor):
    def __init__(self,
                 max_depth=5,
                 max_features=None,
                 min_samples_split=40,
                 criterion='mse',
                 n_categories_power=5,
                 random_state=None):
        fasttree.FastTreeRegressor.__init__(self, max_depth=max_depth,
                                            max_features=max_features,
                                            min_samples_split=min_samples_split,
                                            max_events_used=1,
                                            criterion=criterion,
                                            random_state=random_state)
        self.n_categories_power = n_categories_power

    def _fit_tree_node(self, X, y, w, node_index, depth, passed_indices):
        """Recursive function to fit tree"""
        if len(passed_indices) <= self.min_samples_split or depth >= self.max_depth:
            self.nodes_data[node_index] = (numpy.average(y[passed_indices], weights=w[passed_indices]), )
            return

        multiplier = numpy.random.choice([1, 3, 5, 7, 11, 13, 17, 23])
        n_categories = 2 ** self.n_categories_power
        mask = n_categories - 1
        # modulus = self.random_state.randint(self.min_modulus, self.max_modulus)
        selected_features = self.random_state.choice(self.n_features, size=self._n_used_features, replace=False)
        candidates = OrderedDict()
        costs = []
        for feature_id in selected_features:
            # the last element is 0, new elements will be sent to left subtree
            nominator = numpy.bincount((multiplier * X[passed_indices, feature_id]) & mask,
                                       weights=w[passed_indices] * y[passed_indices],
                                       minlength=n_categories)
            weights = numpy.bincount((multiplier * X[passed_indices, feature_id]) & mask,
                                     weights=w[passed_indices],
                                     minlength=n_categories) + 2
            nominator += 2 * numpy.random.random(len(nominator))
            means = nominator / weights
            order = numpy.argsort(means)
            means = means[order]
            weights = weights[order]
            na = numpy.newaxis
            cut, cost, _ = self._criterion.compute_best_splits(numpy.arange(n_categories)[:, na], means, weights)
            costs.append(cost)
            directions = numpy.argsort(order) > cut
            candidates[feature_id] = (cut, cost, directions)

        # feature that showed best pre-estimated cost
        feature_index = selected_features[numpy.argmin(costs)]
        split, cost, directions = candidates[feature_index]
        passed = numpy.take(directions, (multiplier * X[passed_indices, feature_index]) & mask)
        # computing information for (possible) children
        passed_left_subtree = passed_indices[~passed]
        passed_right_subtree = passed_indices[passed]
        left, right = self._children(node_index)
        if len(passed_left_subtree) == 0 or len(passed_right_subtree) == 0:
            # this will be leaf
            self.nodes_data[node_index] = (numpy.average(y[passed_indices], weights=w[passed_indices]), )
        else:
            # non-leaf, recurrent calls
            self.nodes_data[node_index] = (feature_index, multiplier, mask, directions)
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
            feature_index, multiplier, mask, directions = node_data
            clipped_category = (multiplier * X[passed_indices, feature_index]) & mask
            passed = numpy.take(directions, clipped_category)
            passed_left_subtree = passed_indices[~passed]
            passed_right_subtree = passed_indices[passed]
            left, right = self._children(node_index)
            self._apply_node(X, leaf_indices, predictions, left, passed_left_subtree)
            self._apply_node(X, leaf_indices, predictions, right, passed_right_subtree)


class SimpleCategoricalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_features=1, n_categories_power=7, regularization=1., n_attempts=1, method='pvalue'):
        """
        :param n_attempts: will try several things and work with the best one
        :param method:
        :return:
        """
        self.n_features = n_features
        self.n_categories_power = n_categories_power
        self.regularization = regularization
        self.n_attempts = n_attempts
        self.method = method

    def fit(self, X, y, sample_weight, check_input=False):
        minlength = 2 ** self.n_categories_power
        candidates = []
        mask = generate_slice(len(X), min(1., 100000. / len(X)))
        testX, testY, testW = X[mask], y[mask], sample_weight[mask]
        for _ in range(self.n_attempts):
            feature_ids = numpy.random.choice(X.shape[1], size=self.n_features, replace=False)
            feature_coeffs = numpy.random.choice([-1, 0, 0, 0, 2, 5, 7, 13, 51, 113, 227],
                                                 size=self.n_features, replace=False)
            categories = self._compute_categories_full(testX, feature_ids, feature_coeffs)
            if self.method == 'pvalue':
                quality_nom = numpy.bincount(categories, weights=testY, minlength=minlength) ** 2
                quality_denom = numpy.bincount(categories, minlength=minlength) + 3.
                quality = (quality_nom / quality_denom).sum()
            else:
                assert self.method == 'cv'
                denom = numpy.abs(testY) * (1 - numpy.abs(testY))
                values = 0.1 * numpy.bincount(categories[::2], weights=testY[::2], minlength=minlength)
                values /= numpy.bincount(categories[::2], weights=denom[::2], minlength=minlength) + 1e-6
                quality = testY[1::2] * values[categories[1::2]] - denom[1::2] * values[categories[1::2]] ** 2 / 2.
                quality = quality.sum()

            candidates.append([quality, feature_ids, feature_coeffs])

        # selecting best (with max quality)
        candidates = sorted(candidates, key=lambda x: -x[0])
        quality, self.feature_ids, self.feature_coeffs = candidates[0]

        # computing statistics
        feature_categories = self._compute_categories(X)

        categories_sum = numpy.bincount(feature_categories, weights=y * sample_weight, minlength=minlength)
        y_abs = numpy.abs(y)
        categories_denominator = numpy.bincount(feature_categories, weights=y_abs * (1 - y_abs) * sample_weight,
                                                minlength=minlength)

        self.categories_values = categories_sum / (categories_denominator + self.regularization)
        return self

    def _compute_categories_full(self, X, feature_ids, feature_coeffs):
        result = numpy.zeros(len(X), dtype='uint16')
        for feature, coeff in zip(feature_ids, feature_coeffs):
            result += X[:, feature] * coeff
        result &= 2 ** self.n_categories_power - 1
        return result

    def _compute_categories(self, X):
        return self._compute_categories_full(X, self.feature_ids, self.feature_coeffs)

    def predict(self, X):
        feature_categories = self._compute_categories(X)
        return numpy.take(self.categories_values, feature_categories)


def oblivious_normalize(x, splits, n_features, pfactor):
    factor = pfactor / (1. - pfactor)
    z = x.astype('float32').reshape([splits + 1] * n_features)
    for axis in range(n_features):
        z += z.sum(axis=axis, keepdims=True) * factor
    return z.reshape([-1])


def generate_slice(length, subsample):
    if subsample == 1.0:
        return slice(None, None, None)
    else:
        p = numpy.random.random() * (1. - subsample)
        start = int(p * length)
        stop = int((p + subsample) * length)
        return slice(start, stop, None)


class ObliviousCategoricalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_features=1, n_categories_power=5, splits=3, subsample=0.1, regularization=1., pfactor=None):
        self.n_features = n_features
        self.n_categories_power = n_categories_power
        self.regularization = regularization
        self.subsample = subsample
        self.splits = splits
        self.pfactor = pfactor

    def fit(self, X, y, sample_weight, check_input=False):
        if (self.splits + 1) ** self.n_features > 100000:
            raise ValueError('We need to much memory!')
        if self.pfactor is None:
            self.pfactor = 0.1
        n_categories = 2 ** self.n_categories_power
        self.mask = n_categories - 1
        self.feature_ids = numpy.random.choice(X.shape[1], size=self.n_features, replace=False)
        self.feature_coeffs = numpy.random.choice([1, 5, 7, 13, 51, 113, 191, 227], size=self.n_features)
        self.total_bins = (self.splits + 1) ** self.n_features
        # each category will be
        self.maps = []
        selection = generate_slice(len(X), self.subsample)

        for feature, coeff in zip(self.feature_ids, self.feature_coeffs):
            data = (X[selection, feature] * coeff) & self.mask

            denom = numpy.bincount(data, minlength=n_categories) + 1e-10
            nom = numpy.bincount(data, weights=y[selection], minlength=n_categories)
            mean = nom / denom
            percentiles = numpy.random.choice(mean, p=denom / denom.sum(), size=self.splits, replace=False)
            percentiles = numpy.unique(percentiles)
            percentiles += numpy.random.normal() * 1e-10
            self.maps.append(numpy.searchsorted(percentiles, mean).astype('int8'))

        feature_categories = self._compute_categories(X)

        categories_sum = numpy.bincount(feature_categories, weights=y * sample_weight, minlength=self.total_bins)
        y_abs = numpy.abs(y)
        categories_denominator = numpy.bincount(feature_categories, weights=y_abs * (1 - y_abs) * sample_weight,
                                                minlength=self.total_bins)
        categories_denominator += self.regularization

        categories_sum = oblivious_normalize(categories_sum, self.splits, self.n_features, self.pfactor)
        categories_denominator = oblivious_normalize(categories_denominator, self.splits, self.n_features, self.pfactor)

        self.categories_values = categories_sum / categories_denominator
        return self

    def _compute_categories(self, X):
        result = numpy.zeros(len(X), dtype=int)
        for feature, coeff, mapping in zip(self.feature_ids, self.feature_coeffs, self.maps):
            result *= (self.splits + 1)
            result += numpy.take(mapping, (X[:, feature] * coeff) & self.mask)
        return result

    def predict(self, X):
        feature_categories = self._compute_categories(X)
        return self.categories_values[feature_categories]


class CategoricalLinearClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, l1_reg=0., l2_reg=0., max_categories=128):
        """

        :param learning_rate:
        :param l1_reg:
        :param l2_reg:
        :param max_categories: in each feature number should be in [0, n_categories)
        :return:
        """
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_categories = max_categories

    def fit(self, X, y, sample_weight=None, iterations=100, loss=None):
        X, y = check_arrays(X, y)
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        if loss is None:
            loss = BinomialDevianceLossFunction()
        loss.fit(X, y, sample_weight=sample_weight)
        self.n_features = X.shape[1]
        self.coeffs = numpy.zeros([self.n_features, self.max_categories], dtype='float')
        assert numpy.max(X) < self.max_categories
        assert numpy.min(X) >= 0

        for iteration in range(iterations):
            # this line could be skipped, but we need it to avoid
            # mistakes after too many steps of computations
            y_pred = self.decision_function(X)
            print(iteration, loss(y_pred))

            for feature in range(self.n_features):
                ngradient = loss.negative_gradient(y_pred)
                nominator = numpy.bincount(X[:, feature], weights=ngradient, minlength=self.max_categories)
                nominator -= self.l2_reg * self.coeffs[feature, :] + self.l1_reg * numpy.sign(self.coeffs[feature, :])

                denominator = numpy.abs(ngradient) * (1. - numpy.abs(ngradient))
                denominator = numpy.bincount(X[:, feature], weights=denominator, minlength=self.max_categories)
                denominator += 2 * self.l2_reg + 5

                gradients = nominator / denominator
                right_gradients = gradients
                # those already zeros not to become nonzero
                mask = (self.coeffs[feature, :] == 0) & (numpy.abs(gradients) < self.l1_reg)
                right_gradients[mask] = 0
                # those already not zeros
                old_coeffs = self.coeffs[feature, :]
                new_coeffs = old_coeffs + self.learning_rate * right_gradients
                new_coeffs[new_coeffs * old_coeffs < 0] = 0
                y_pred += numpy.take(new_coeffs - old_coeffs, X[:, feature])
                self.coeffs[feature, :] = new_coeffs

        return self

    def decision_function(self, X):
        X = numpy.array(X)
        assert X.shape[1] == self.n_features
        result = numpy.zeros(len(X), dtype='float')
        for feature in range(self.n_features):
            result += numpy.take(self.coeffs[feature, :], X[:, feature])
        return result

    def predict_proba(self, X):
        result = numpy.zeros([len(X), 2])
        result[:, 1] = expit(self.decision_function(X))
        result[:, 0] = 1 - result[:, 1]
        return result






