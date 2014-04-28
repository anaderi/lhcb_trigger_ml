import random
import math
import numpy
from numpy.random.mtrand import randint
import pandas
from sklearn.neighbors import NearestNeighbors
import time

__author__ = 'Alex Rogozhnikov'

# This module contains procedures to generate Toy Monte-Carlo
# by using modified SMOTE approach


def getRandom(candidates, all_weights):
    """
    weights - probabilities of all elements
    candidates - (numpy.array), we choose among them according to their weights,
       canidates are indices of weights array
    very simple (but in this case quite efficient) implementation
    """
    # TODO numpy.choice for later versions of numpy
    weights = all_weights[candidates]

    p = numpy.random.random() * sum(weights)
    for i in range(len(candidates)):
        p -= weights[i]
        if p < 0:
            return candidates[i]
    return candidates[-1]


def testGetRandom():
    candidates = [2 * i + 1 for i in range(10)]
    weights = numpy.array(range(30)) ** 2

    random_results = numpy.array([getRandom(candidates, weights) for _ in range(1000)])

    for x in random_results:
        assert x in candidates, "GetRandom gives unexpected result"
    probs = [numpy.sum(random_results == x) for x in candidates]
    probs = numpy.array(probs) * 1.0 / len(random_results)
    w = weights[candidates] * 1.0 / sum(weights[candidates])
    mse = numpy.sum((probs - w) ** 2)
    assert mse < 0.004, "MSE of getRandom is too big"
    print "getRandom is ok"

testGetRandom()


def countProbabilities(weights, knn):
    """
    Computes probabilities of all point to be chosen as the second point in pair
     * knn is array or dictionary which contains indices of nearest neighbors
     * weights is an array of weights used
    """
    size = len(knn)
    sum_weights = numpy.zeros(size)
    for i, neighbours_i in enumerate(knn):
        sum_weights[i] = numpy.sum(weights[neighbours_i])
    inverse_weights = sum_weights ** -1
    probabilities = weights + 0.0
    for i, neighbours_i in enumerate(knn):
        probabilities[i] *= numpy.sum(inverse_weights[neighbours_i])
    return probabilities



def generateToyMonteCarlo(inputDF, size, knn=4, symmetrize=True, power=2.0, reweight_iterations=5):
    """
    Generates set of events with the same distribution as in inputDF
    Referred as SMOTE in the ML papers
    inputDF - numpy.ndarray or pandas.Dataframe
    size - how many new events to generate
    symmetrize - if symmetrize==True, knn will be computed in symmetric way: if a in knn of b,
        then b in knn of a, this helps to fight covariance shrinking
    power - instead of uniform distribution, makes the point to tend to one of initial points
        (the greater, the closer to first point)
    reweightIterations - an iterative algorithm, which changes the probabilities so that all the points have
        equal probability to be chosen as neighbour
    --------
    Returns:
        (result, n_copied_from_original)
        result is generated toy MonteCarlo,
        n_copied_from_original - number of elements that were taken from original dataframe without any modification
    """
    t = time.time()

    inputDF = pandas.DataFrame(inputDF)
    input_length = len(inputDF)


    if input_length <= 2:
        # print "unable to generate new events with only one-two given"
        return inputDF, len(inputDF)

    if knn is None:
        knn = int(math.pow(input_length, 0.33)) / 2
        knn = max(knn, 2)
        knn = min(knn, 25)
        knn = min(knn, input_length)

    assert knn > 0, "knn should be positive"

    numpiedDF = inputDF.values

    neighbours_helper = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree', )
    neighbours_helper.fit(inputDF)
    neighbours = neighbours_helper.kneighbors(inputDF, return_distance=False)
    two_side_neighbours = {}

    for i, neighbours_i in enumerate(neighbours):
        two_side_neighbours[i] = list(neighbours_i[1:])

    if symmetrize:
        for i in range(len(neighbours)):
            for n in neighbours[i]:
                two_side_neighbours[n].append(i)

    weights = numpy.zeros(len(neighbours)) + 1
    for _ in range(reweight_iterations):
        probs = countProbabilities(weights, two_side_neighbours)
        weights *= (probs ** -0.5)

    # generating indices and weights
    k_1 = randint(0, input_length, size)
    t_1 = 0.6 * numpy.random.random(size) ** power
    t_2 = 1. - t_1

    k_2 = numpy.zeros(size, dtype=numpy.int)
    for i in range(size):
        neighs = two_side_neighbours[k_1[i]]
        selected_neigh = getRandom(neighs, weights)
        k_2[i] = selected_neigh

    first = numpy.multiply(t_1[:, numpy.newaxis], numpiedDF[k_1, :])
    second = numpy.multiply(t_2[:, numpy.newaxis], numpiedDF[k_2, :])

    return pandas.DataFrame(numpy.add(first, second), columns=inputDF.columns), 0

    # new_events = []
    #
    # for i in range(size):
    #     k = random.randint(0, input_length - 1)
    #     neighs = two_side_neighbours[k]
    #     selected_neigh = getRandom(neighs, weights)
    #     # from 0 to 1/2
    #     t = 0.5 * random.random() ** power
    #
    #     new_event = (1.0 - t) * inputDF.irow(k) + t * inputDF.irow(selected_neigh)
    #     new_events.append(new_event)
    #
    # result = pandas.DataFrame(new_events)
    # result.set_index([range(len(result))], inplace=True)
    #
    # return result, 0

def prepareToyMC(group, clusterization_features, stayed_features, size_coeff):
    group_vals, df = group
    toyMC_part, n_copied = generateToyMonteCarlo(df[stayed_features],  int(len(df) * size_coeff), knn=None)
    for i, col in enumerate(clusterization_features):
        toyMC_part[col] = group_vals[i]
    return toyMC_part, n_copied



def generateToyMonteCarloWithSpecialFeatures(inputDF, size, clusterization_features=None, integer_features=None,
                                             ipc_profile=None):
    """
    Excluded features - features we absolutely don't take into account
    ClusterizationFeatures - very close to integer ones, usually have some bool or integer values,
        but events with different values in these columns should not be mixed together
        example: 'isSignal', number of muons
    IntegerFeatures - (rarely used) features that have some discrete values, but events can be mixed together
        if they have different values in these columns, the result will be integer
        example: some weight of event, which should be integer due to technical restrictions
    """
    if integer_features is None:
        integer_features = []
    if clusterization_features is None:
        clusterization_features = []
    stayed_features = [col for col in inputDF.columns if col not in clusterization_features]
    size_coeff = float(size) / len(inputDF)
    copied_groups = 0
    if len(clusterization_features) == 0:
        result, copied = generateToyMonteCarlo(inputDF,  int(len(inputDF) * size_coeff), knn=None)
    else:
        grouped = inputDF.groupby(clusterization_features)
        print "Generating ..."
        copied_list = []
        if ipc_profile is None:
            toyMC_parts = []
            for group_vals, df in grouped:
                toyMC_part, n_copied = generateToyMonteCarlo(df[stayed_features],  int(len(df) * size_coeff), knn=None)
                copied_list.append(n_copied)
                for i, col in enumerate(clusterization_features):
                    toyMC_part[col] = group_vals[i]
                toyMC_parts.append(toyMC_part)
        else:
            from IPython.parallel import Client
            client = Client(profile=ipc_profile)
            grouped = list(grouped)
            results = client.load_balanced_view().map_sync(prepareToyMC, grouped,
                  [clusterization_features] * len(grouped), [stayed_features] * len(grouped),
                  [size_coeff] * len(grouped))
            toyMC_parts, copied_list = zip(*results)
            del client
        copied_list = numpy.array(copied_list)
        copied = copied_list.sum()
        copied_groups = numpy.sum(copied_list != 0)
        result = pandas.concat(toyMC_parts)
    for col in integer_features:
        result[col] = result[col].astype(numpy.int)
    if copied > 0:
        print "Copied %i events in %i groups from original file. Totally generated %i rows " %\
              (copied, copied_groups, len(result))
    return result


def testToyMonteCarlo(size=100):
    df = pandas.DataFrame(numpy.random.random((size, 40)))
    res = generateToyMonteCarloWithSpecialFeatures(df, 5000)
    assert isinstance(res, pandas.DataFrame), "something wrong with MonteCarlo"
    print "toyMC is ok"

testToyMonteCarlo(1000)

# import cProfile as profile
# profile.run("testToyMonteCarlo(10000)")

