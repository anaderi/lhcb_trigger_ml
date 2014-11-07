from __future__ import division, print_function, absolute_import
import numpy

from ..commonutils import indices_of_values


__author__ = 'Alex Rogozhnikov'


def generate_max_voter(event_indices):
    """
    This voter is prepared specially for experiments in triggers.
    Voting returns max(svr predictions),
    :param event_indices: array, each element is the index of event
    which current SVR belongs to.

    Result should be used as voter in meanadaboost.
    """
    groups = indices_of_values(event_indices)

    def voter(cumulative_score, knn_scores):
        result = numpy.zeros(len(cumulative_score))
        for key, group in groups:
            result[group] = numpy.max(cumulative_score[group])
        return result

    return voter
