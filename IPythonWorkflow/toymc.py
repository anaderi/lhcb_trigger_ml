from __future__ import division
from __future__ import print_function

import math
import numpy
import pandas
from scipy.stats.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from commonutils import map_on_cluster, check_sample_weight

__author__ = 'Alex Rogozhnikov'

# About

# This module contains procedures to generate Toy Monte-Carlo
# by using modified SMOTE approach

# TODO test whether we really need to symmetrize, in the other case everything can be simplified
# TODO use classifier-based measure of quality
# TODO random state


def count_probabilities(primary_weights, secondary_weights, knn):
    """Computes probabilities of all points to be chosen as the second point in pair
    :type primary_weights: numpy.array, shape = [n_samples],
        the first event is generated according to these weights
    :type secondary_weights: numpy.array, shape = [n_samples],
        the second event is chosen between knn of first according to this weights
    :type knn: dict, {event_id: list of neighbors ids}.
    :rtype: numpy.array, shape = [n_samples], the probabilities
    """
    size = len(knn)
    primary_weights /= numpy.sum(primary_weights)
    secondary_weights = numpy.array(secondary_weights)
    probabilities = numpy.zeros(size, dtype=float)
    for index, neighbours in knn.iteritems():
        knn_weights = numpy.take(secondary_weights, neighbours)
        knn_proba = knn_weights / numpy.sum(knn_weights) * primary_weights[index]
        probabilities[neighbours] += knn_proba
    return probabilities


def generate_toymc(data, size, knn=4, symmetrize=True, power=2.0, reweighting_iterations=5, sample_weight=None):
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

    sample_weight = check_sample_weight(data, sample_weight=sample_weight)
    sample_weight /= numpy.sum(sample_weight)

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

    secondary_weights = numpy.ones(len(neighbours), dtype=float)
    for _ in range(reweighting_iterations):
        probabilities = count_probabilities(sample_weight, secondary_weights, two_side_neighbours)
        secondary_weights *= ((sample_weight / probabilities) ** 0.5)

    # generating indices and weights
    k_1 = numpy.random.choice(input_length, p=sample_weight, size=size)
        # randint(0, input_length, size)
    t_1 = 0.6 * numpy.random.random(size) ** power
    t_2 = 1. - t_1

    k_2 = numpy.zeros(size, dtype=int)
    for i in range(size):
        neighs = two_side_neighbours[k_1[i]]
        neigh_weights = numpy.take(secondary_weights, neighs)
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
        results = map_on_cluster(ipc_profile, prepare_toymc, grouped, [clustering_features] * n_groups,
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


# def compare_covariance_3d(data, toy_data, n_features=6):
#     import pylab
#     data_cov = numpy.cov(data.T)[:n_features,:n_features]
#     toy_cov = numpy.cov(toy_data.T)[:n_features,:n_features]
#
#     pylab.figure(figsize=(12, 5))
#     assert data_cov.shape == toy_cov.shape, "different size of matrices"
#     vars1, vars2 = data_cov.shape
#     x, y = range(vars1), range(vars2)
#     X, Y = numpy.meshgrid(x, y)
#     X = X.flatten()
#     Y = Y.flatten()
#     Z_min = numpy.zeros_like(X)
#     Z_max_left = data_cov.flatten()
#     Z_max_right = toy_cov.flatten()
#
#     maximal_cov = max(numpy.max(toy_cov), numpy.max(data_cov))
#
#     pylab.subplot(121, projection='3d')
#     pylab.bar3d(X - 0.5, Y - 0.5, Z_min, 1, 1, Z_max_left,  color='b', zsort='average')
#     pylab.zlim(0, maximal_cov)
#     pylab.title("Original MC")
#     pylab.view_init(35, 225 + 30)
#
#     pylab.subplot(122, projection='3d')
#     pylab.bar3d(X - 0.5, Y - 0.5, Z_min, 1, 1, Z_max_right, color='b', zsort='average')
#     pylab.zlim(0, maximal_cov)
#     pylab.title("Toy MC")
#     pylab.view_init(35, 225 + 30)
#     pylab.show()


def test_on_dataframe(df, excluded_features=None, clustering_features=None, integer_features=None):
    """Prints comparison of distributions: original one and toymc.
    :type excluded_features:  features we absolutely don't take into account
    :type clustering_features: list | None, very close to integer ones, usually have some bool or integer values,
        but events with different values in these columns should not be mixed together
        example: 'isSignal', number of muons
    :type integer_features: list | None, features that have some discrete values, but events can be mixed together
        if they have different values in these columns, the result will be integer
        example: some weight of event, which should be integer due to technical restrictions
    """
    from IPython.display import display_html
    import pylab

    if excluded_features is None:
        excluded_features = []
    else:
        print("\nEXCLUDED columns:\n", list(excluded_features))

    selected_columns = [col for col in df.columns if col not in excluded_features]
    print("\nSTAYED columns:\n", selected_columns)

    if integer_features is None:
        integer_features = []
    else:
        print("\nINTEGER columns:\n", list(integer_features))

    if clustering_features is None:
        clustering_features = []
    else:
        print("\nCLUSTERING columns:\n", list(clustering_features))

    data = df[selected_columns]
    toy_data = generate_toymc_with_special_features(data, len(data), clustering_features=clustering_features,
                                                    integer_features=integer_features)

    numpy.set_printoptions(precision=4, suppress=True)
    n_cols = 3
    n_rows = (len(data.columns) + n_cols - 1) // n_cols
    pylab.figure(figsize=(18, 5 * n_rows))
    for i, column in enumerate(data.columns):
        pylab.subplot(n_rows, n_cols, i + 1)
        pylab.title(column)
        pylab.hist([data[column], toy_data[column]], histtype='step', bins=20)
    pylab.show()

    print("\nMEANS AND STD")
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

    print("\nCOVARIANCES")
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
            pylab.title("original MC")

            pylab.subplot(122)
            pylab.plot(toy_data[first_column], toy_data[second_column], '.', alpha=0.1)
            pylab.ylim(y_min, y_max), pylab.xlim(x_min, x_max)
            pylab.title("toy MC")

            pylab.suptitle(str(first_column) + " vs " + str(second_column))
            pylab.show()


def test_toy_monte_carlo(size=100):
    df = pandas.DataFrame(numpy.random.random((size, 40)))
    res = generate_toymc_with_special_features(df, 5000)
    assert isinstance(res, pandas.DataFrame), "something wrong with MonteCarlo"
    print("toymc is ok")


if __name__ == '__main__':
    test_on_dataframe(pandas.DataFrame(numpy.random.normal(size=(1000, 10))))
    import cProfile
    cProfile.run("test_toy_monte_carlo(10000)")
else:
    test_toy_monte_carlo(1000)