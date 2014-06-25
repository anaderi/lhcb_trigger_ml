from __future__ import division
from __future__ import print_function

import math
import numpy
import pandas
from numpy.random.mtrand import randint
from sklearn.neighbors import NearestNeighbors

__author__ = 'Alex Rogozhnikov'

# About

# This module contains procedures to generate Toy Monte-Carlo
# by using modified SMOTE approach


def count_probabilities(weights, knn):
    """Computes probabilities of all points to be chosen as the second point in pair
    :type weights: numpy.array, shape = [n_samples],
        the second element is chosen between knn according to this weights
    :type knn: dict, {event_id: list of neighbors ids}.
    :rtype: numpy.array, shape = [n_samples], the probabilities
    """
    size = len(knn)
    weights = numpy.array(weights)
    probabilities = numpy.zeros(size)
    for index, neighbours in knn.iteritems():
        knn_weights = numpy.take(weights, neighbours)
        knn_proba = knn_weights / numpy.sum(knn_weights) / size
        probabilities[neighbours] += knn_proba
    return probabilities


def generate_toymc(data, size, knn=4, symmetrize=True, power=2.0, reweighting_iterations=5):
    """Generates toy Monte-Carlo, the dataset with distribution very close to the original one

    :type data: numpy.array | pandas.DataFrame, the original distribution
    :type size: int, the number of events to generate
    :type knn: int | None, how many neighbours should we consider
    :type symmetrize: bool, if symmetrize==True, knn will be computed in symmetric way: if a in knn of b,
        then b in knn of a, this helps to fight covariance shrinking
    :type power: float, instead of uniform distribution, makes the point to tend to one of initial points
        (the greater, the closer to first point)
    :type reweighting_iterations: int, an iterative algorithm is used, which changes the probabilities
        so that all the points have equal probability to be chosen as neighbour
    :rtype: (pandas.DataFrame, int), returns the generated toymc and the number of events
        that were copied from original data set.
    """

    data = pandas.DataFrame(data)
    input_length = len(data)

    if input_length <= 2:
        # unable to generate new events with only one-two given
        return data, len(data)

    if knn is None:
        knn = int(math.pow(input_length, 0.33) / 2)
        knn = max(knn, 2)
        knn = min(knn, 25)
        knn = min(knn, input_length)

    assert knn > 0, "knn should be positive"

    # generating knn
    neighbors_helper = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree', )
    neighbors_helper.fit(data)
    neighbours = neighbors_helper.kneighbors(data, return_distance=False)

    two_side_neighbours = {}
    for i, neighbours_i in enumerate(neighbours):
        two_side_neighbours[i] = list(neighbours_i[1:])

    if symmetrize:
        # symmetrization goes here
        for i in range(len(neighbours)):
            for n in neighbours[i]:
                two_side_neighbours[n].append(i)

        # removing duplicates in neighbors
        old_neighbours = two_side_neighbours
        two_side_neighbours = {}
        for i, neighbours_i in old_neighbours.iteritems():
            two_side_neighbours[i] = numpy.unique(neighbours_i)

    weights = numpy.ones(len(neighbours), dtype=float)
    for _ in range(reweighting_iterations):
        probabilities = count_probabilities(weights, two_side_neighbours)
        weights *= (probabilities ** -0.5)

    # generating indices and weights
    k_1 = randint(0, input_length, size)
    t_1 = 0.6 * numpy.random.random(size) ** power
    t_2 = 1. - t_1

    k_2 = numpy.zeros(size, dtype=numpy.int)
    for i in range(size):
        neighs = two_side_neighbours[k_1[i]]
        neigh_weights = numpy.take(weights, neighs)
        neigh_weights /= numpy.sum(neigh_weights)
        # selected_neigh = getRandom(neighs, weights)
        k_2[i] = numpy.random.choice(neighs, p=neigh_weights)

    numpied_df = data.values
    first = numpy.multiply(t_1[:, numpy.newaxis], numpied_df[k_1, :])
    second = numpy.multiply(t_2[:, numpy.newaxis], numpied_df[k_2, :])

    return pandas.DataFrame(numpy.add(first, second), columns=data.columns), 0


def prepare_toymc(group, clustering_features, stayed_features, size_factor):
    """This procedure prepares one block of data,  written specially for parallel execution
    :type group: pandas.grouping = (group_key, group_data), the data used to generate monte-carlo
    :type clustering_features: the features that were used to split data
    :type stayed_features: the names of other features, needed to reconstruct
    :type size_factor: float, the size of generated toymc is about size_factor * len(group)
    :rtype: (pandas.DataFrame, int)
    """
    group_values, df = group
    toymc_part, n_copied = generate_toymc(df[stayed_features],  int(len(df) * size_factor), knn=None)
    for i, col in enumerate(clustering_features):
        toymc_part[col] = group_values[i]
    return toymc_part, n_copied


def generate_toymc_with_special_features(data, size, clustering_features=None, integer_features=None,
                                         ipc_profile=None):
    """Generate the toymc.
    :type data: numpy.array | pandas.DataFrame, from which data is generated
    :type size: int, how many events to generate
    :type clustering_features: the events with different values of this feature can not be mixed together.
        For instance: is_signal, number of jets / muons.
    :type integer_features: this features are treated as usual,
        but after toymc is generated, they are rounded to the closest integer value
    :type ipc_profile: toymc can be generated on the cluster,
        provided there is at least one clustering feature
    :rtype: pandas.DataFrame with result,
        all the columns should be the same as in input
    """
    if integer_features is None:
        integer_features = []
    if clustering_features is None:
        clustering_features = []
    stayed_features = [col for col in data.columns if col not in clustering_features]
    size_factor = float(size) / len(data)
    copied_groups = 0
    if len(clustering_features) == 0:
        result, copied = generate_toymc(data, size=size, knn=None)
    else:
        grouped = data.groupby(clustering_features)
        print("Generating ...")
        n_groups = len(grouped)
        if ipc_profile is None:
            results = map(prepare_toymc, grouped, [clustering_features] * n_groups,
                          [stayed_features] * n_groups, [size_factor] * n_groups)
        else:
            from IPython.parallel import Client
            lb_view = Client(profile=ipc_profile).load_balanced_view()
            grouped = list(grouped)
            results = lb_view.map_sync(prepare_toymc, grouped, [clustering_features] * n_groups,
                                       [stayed_features] * n_groups, [size_factor] * n_groups)

        toymc_parts, copied_list = zip(*results)
        copied = numpy.sum(copied_list)
        copied_groups = numpy.sum(numpy.array(copied_list) != 0)
        result = pandas.concat(toymc_parts)
    for col in integer_features:
        result[col] = result[col].astype(numpy.int)
    if copied > 0:
        print("Copied %i events in %i groups from original file. Totally generated %i rows " %
              (copied, copied_groups, len(result)))
    return result


def test_toy_monte_carlo(size=100):
    df = pandas.DataFrame(numpy.random.random((size, 40)))
    res = generate_toymc_with_special_features(df, 5000)
    assert isinstance(res, pandas.DataFrame), "something wrong with MonteCarlo"
    print("toymc is ok")


if __name__ == '__main__':
    import cProfile
    cProfile.run("test_toy_monte_carlo(10000)")
else:
    test_toy_monte_carlo(1000)