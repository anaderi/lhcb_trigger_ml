from __future__ import division, print_function, absolute_import

import numpy
import pandas
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.distributions import norm

__author__ = 'Alex Rogozhnikov'


class SupervisedTransform(BaseEstimator, TransformerMixin):
    def __init__(self, scale=0., like_normal=False):
        """
        This transformation applies nonlinear rescale to each axis,
        only order of events is taken into account.

        If sig and bck events are neighbours in some feature, they will be 'splitted' by 'insertion'
        Scale is how big insertion is
        """
        self.like_normal = like_normal
        self.scale = scale

    def fit(self, X, y):
        X = numpy.array(X)
        self.initial_values = []
        self.transformed_values = []
        for axis in range(X.shape[1]):
            initial_values = X[:, axis] * (1 + 1e-6 * numpy.random.normal(size=len(X)))
            initial_values += 1e-8 * numpy.random.normal(size=len(X))
            indices = numpy.argsort(initial_values)
            initial_values = initial_values[indices]
            self.initial_values.append(initial_values)
            transformed = numpy.arange(len(X), dtype='float')
            # increase the distance between neighs of different classes
            additions = numpy.abs(numpy.diff(y[indices]))
            additions = numpy.cumsum(additions)
            transformed[1:] += additions * self.scale
            transformed /= transformed[-1] / 2.
            transformed -= 1

            if self.like_normal:
                # converting to normal-like distributions
                transformed -= transformed[0]
                transformed /= transformed[-1] / 0.9
                transformed += 0.05
                transformed = norm().ppf(transformed)

            self.transformed_values.append(transformed)

        return self

    def transform(self, X):
        X = numpy.array(X)
        result = []
        for axis, (init_vals, trans_vals) in enumerate(zip(self.initial_values, self.transformed_values)):
            result.append(numpy.interp(X[:, axis], init_vals, trans_vals))
        return numpy.vstack(result).T


def shuffled_indices(n_samples, shuffle_factor, random_state=None):
    random_state = check_random_state(random_state)
    order = numpy.arange(n_samples) + random_state.normal(0, shuffle_factor * n_samples, size=n_samples)
    quantiles = numpy.argsort(order) + random_state.normal(0, 1, size=n_samples)
    return numpy.clip(quantiles / (n_samples - 1.), 0, 1)


class Shuffler(BaseEstimator, TransformerMixin):
    """
    Shuffler
    """
    def __init__(self, shuffle_factor=0.05, not_shuffled_columns=None, random_state=None):
        self.shuffle_factor = shuffle_factor
        self.random_state = check_random_state(random_state)
        if not_shuffled_columns is None:
            self.not_shuffled_columns = []
        else:
            self.not_shuffled_columns = not_shuffled_columns

    def fit(self, X, y=None):
        """
        In principle, for transforming the same dataset many times,
        the process may be much faster. This will require fitting.
        """
        pass

    def transform(self, X):
        assert isinstance(X, pandas.DataFrame) or len(self.not_shuffled_columns) == 0, \
            'to work with columns, one shall pass pandas.DataFrame, not numpy.array'

        if self.shuffle_factor <= 0:
            return X.copy()

        result = pandas.DataFrame(X).copy()
        for column in result.columns:
            if column in self.not_shuffled_columns:
                continue

            order = numpy.argsort(result[column])
            permutation = numpy.arange(len(order))
            permutation += self.random_state.normal(0, len(order) * self.shuffle_factor, size=len(order))

            # wrapping the outliers
            upper = len(order) - 1
            permutation = numpy.abs(permutation)
            permutation = upper - numpy.abs(permutation - upper)
            permutation = numpy.clip(permutation.astype(numpy.int), 0, upper)
            # I love this conjugation construction. Mindblowing a bit.
            result[column] = result[column].values[order][permutation][numpy.argsort(order)]
        return result


# endregion



