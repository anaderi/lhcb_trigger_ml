"""
This module contains procedures to generate toy Monte-Carlo (toyMC)
by using modified SMOTE approach.

Mathematically, toyMC generation is like trying to generate data
from same distribution as you already have.

SMOTE approach relies on assumption that events close in euclidean metrics
in dataset are really close.

The more data -> the better toyMC.
"""
from __future__ import division, print_function

import math
import numpy
import pandas
import pylab
from scipy.stats.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from .commonutils import map_on_cluster, check_sample_weight

__author__ = 'Alex Rogozhnikov'
__all__ = ['generate_toymc_with_special_features']


# TODO test whether we really need to symmetrize, in the other case everything can be simplified


def _count_probabilities(primary_weights, secondary_weights, knn):
    """Computes probabilities of all points to be chosen as the second point in pair
    :type primary_weights: numpy.array, shape = [n_samples],
        the first event is generated according to these weights
    :type secondary_weights: numpy.array, shape = [n_samples],
        the second event is chosen between knn of first according to this weights
    :type knn: dict, {event_id: list of neighbors ids}.
    :rtype: numpy.array, shape = [n_samples], the probabilities
    """
    primary_probabilities = primary_weights / numpy.sum(primary_weights)
    secondary_weights = numpy.array(secondary_weights)
    knn_probabilities = numpy.take(secondary_weights, knn)
    knn_probabilities /= knn_probabilities.sum(axis=1, keepdims=True)

    secondary_probabilities = numpy.bincount(knn.flatten(), weights=knn_probabilities.flatten())
    return primary_probabilities * secondary_probabilities


def generate_toymc(data, size, knn=4, power=2.0,
                   reweighting_iterations=5, sample_weight=None, random_state=numpy.random):
    """Generates toy Monte-Carlo, the dataset with distribution very close to the original one.

    :type data: numpy.array | pandas.DataFrame, the original distribution
    :type size: int, the number of events to generate
    :type knn: int | None, how many neighbours should we consider
    :type power: float, instead of uniform distribution, makes the point to tend to one of initial points
        (the greater, the closer to first point)
    :type reweighting_iterations: int, an iterative algorithm is used, which changes the probabilities
        so that all the points have equal probability to be chosen as neighbour
    :rtype: (pandas.DataFrame, int), returns the generated toymc and the number of events
        that were simply copied from original data set.
    """

    data = pandas.DataFrame(data)
    input_length = len(data)

    sample_weight = check_sample_weight(data, sample_weight=sample_weight)
    sample_weight /= numpy.sum(sample_weight)

    if input_length <= 2:
        # unable to generate new events with only one-two given
        return data, len(data)

    assert knn > 0, "knn should be positive"

    # generating knn
    neighbors_helper = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree', )
    neighbors_helper.fit(data)
    neighbours = neighbors_helper.kneighbors(data, return_distance=False)[:, 1:]

    secondary_weights = numpy.ones(len(neighbours), dtype=float)
    for _ in range(reweighting_iterations):
        probabilities = _count_probabilities(sample_weight, secondary_weights, neighbours)
        secondary_weights *= ((sample_weight / probabilities) ** 0.5)

    # generating indices and weights
    k_1 = random_state.choice(input_length, p=sample_weight, size=size)
    k_2 = numpy.zeros(size, dtype=int)

    # drawing random neighbours
    for i, first_neighbour in enumerate(k_1):
        neighs = neighbours[first_neighbour]
        neigh_weights = numpy.take(secondary_weights, neighs)
        neigh_weights /= numpy.sum(neigh_weights)
        k_2[i] = random_state.choice(neighs, p=neigh_weights)

    numpied_df = data.values
    t_1 = 0.6 * random_state.random(size) ** power
    t_2 = 1. - t_1

    new_data = t_1[:, numpy.newaxis] * numpied_df[k_1, :] + t_2[:, numpy.newaxis] * numpied_df[k_2, :]
    return pandas.DataFrame(new_data, columns=data.columns), 0


def prepare_toymc(group, clustering_features, stayed_features, size_factor):
    """This procedure prepares one block of data,  written specially for parallel execution
    :type group: pandas.grouping = (group_key, group_data), the data used to generate monte-carlo
    :type clustering_features: the features that were used to split data
    :type stayed_features: the names of other features, needed to reconstruct
    :type size_factor: float, the size of generated toymc26 is about size_factor * len(group)
    :rtype: (pandas.DataFrame, int)
    """
    group_values, df = group
    toymc_part, n_copied = generate_toymc(df[stayed_features],  int(len(df) * size_factor), knn=None)
    for i, col in enumerate(clustering_features):
        toymc_part[col] = group_values[i]
    return toymc_part, n_copied


def generate_toymc_with_special_features(data, size, knn=4, clustering_features=None, ipc_profile=None):
    """Generate the toymc.
    :type data: numpy.array | pandas.DataFrame, from which data is generated
    :type size: int, how many events to generate
    :type clustering_features: the events with different values of this feature can not be mixed together.
        For instance: is_signal, number of jets / muons.
    :type ipc_profile: toymc can be generated on the cluster,
        provided there is at least one clustering feature
    :rtype: pandas.DataFrame with result,
        all the columns should be the same as in input
    """
    if clustering_features is None:
        clustering_features = []
    stayed_features = [col for col in data.columns if col not in clustering_features]
    size_factor = float(size) / len(data)
    copied_groups = 0
    if len(clustering_features) == 0:
        result, copied = generate_toymc(data, size=size, knn=knn)
    else:
        grouped = data.groupby(clustering_features)
        print("Generating ...")
        n_groups = len(grouped)
        results = map_on_cluster(ipc_profile, prepare_toymc, grouped, [clustering_features] * n_groups,
                                 [stayed_features] * n_groups, [size_factor] * n_groups)
        toymc_parts, copied_list = zip(*results)
        copied = numpy.sum(copied_list)
        copied_groups = numpy.sum(numpy.array(copied_list) != 0)
        result = pandas.concat(toymc_parts)
    if copied > 0:
        print("Copied %i events in %i groups from original file. Totally generated %i rows " %
              (copied, copied_groups, len(result)))
    return result


def compare_toymc(data, clustering_features=None):
    """Prints comparison of distributions: original one and generated toyMC.
    Should be used in IPython notebook.
    :param pandas.DataFrame data: the dataset we generate toyMC for.
    :type clustering_features: list | None, very close to integer ones, usually have some bool or integer values,
        but events with different values in these columns should not be mixed together
        example: 'isSignal', number of muons
    :type integer_features: list | None, features that have some discrete values, but events can be mixed together
        if they have different values in these columns, the result will be integer
        example: some weight of event, which should be integer due to technical restrictions, isolation vars.
    """
    assert isinstance(data, pandas.DataFrame), 'need a Pandas.DataFrame here'
    from IPython.display import display_html

    def print_title(title):
        display_html("<h3>{}</h3>".format(title), raw=True)

    toy_data = generate_toymc_with_special_features(data, len(data), clustering_features=clustering_features)

    numpy.set_printoptions(precision=4, suppress=True)
    n_cols = 3
    n_rows = (len(data.columns) + n_cols - 1) // n_cols
    pylab.figure(figsize=(18, 5 * n_rows))
    for i, column in enumerate(data.columns, 1):
        pylab.subplot(n_rows, n_cols, i)
        pylab.title(column)
        pylab.hist([data[column].values, toy_data[column].values], histtype='step', bins=20)
    pylab.show()

    print_title("Means and std")
    mean_index = []
    mean_rows = []
    for column in data.columns:
        mean_index.append(column)
        mean1 = numpy.mean(data[column])
        mean2 = numpy.mean(toy_data[column])
        std1 = numpy.std(data[column])
        std2 = numpy.std(toy_data[column])
        mean_rows.append([mean1, mean2, mean2 - mean1, abs((mean1-mean2) * 100. / mean1),
                          std1, std2, std2 - std1, abs((std2 - std1) * 100. / std1)])

    display_html(pandas.DataFrame(mean_rows, index=mean_index,
        columns=['mean orig', 'mean toy', 'difference', 'error, %', 'std orig', 'std toy', 'difference', 'error, %']))

    print_title('Covariance')
    cov_index = []
    cov_rows = []
    for i, first_column in enumerate(data.columns, ):
        for second_column in data.columns[i+1:]:
            cov_index.append((first_column, second_column))
            data_cov = pearsonr(data[first_column], data[second_column])[0]
            toy_cov = pearsonr(toy_data[first_column], toy_data[second_column])[0]
            cov_rows.append([data_cov, toy_cov, toy_cov-data_cov, abs((data_cov - toy_cov) * 100. / data_cov)])
    display_html(pandas.DataFrame(cov_rows, index=cov_index, columns=['original', 'toy', 'difference', 'error, %']))

    for col, first_column in enumerate(data.columns[:4]):
        for second_column in data.columns[col + 1:4]:
            x_min = numpy.min(data[first_column])
            x_max = numpy.max(data[first_column])
            y_min = numpy.min(data[second_column])
            y_max = numpy.max(data[second_column])

            pylab.figure(figsize=(12, 5))
            pylab.subplot(121)
            pylab.plot(data[first_column], data[second_column], '.', alpha=0.1)
            pylab.xlim(x_min, x_max), pylab.ylim(y_min, y_max)
            pylab.title("original data")

            pylab.subplot(122)
            pylab.plot(toy_data[first_column], toy_data[second_column], '.', alpha=0.1)
            pylab.ylim(y_min, y_max), pylab.xlim(x_min, x_max)
            pylab.title("toy MC")

            pylab.suptitle("{} vs {}".format(first_column, second_column))
            pylab.show()
