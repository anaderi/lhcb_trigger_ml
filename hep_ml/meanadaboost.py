"""
In this module some very simple modification of AdaBoost is implemented.
In weights boosting it uses mean of neighbours scores
instead of having only score of event itself
"""
from __future__ import division, print_function, absolute_import

import numpy
import sklearn
from sklearn.tree.tree import DecisionTreeClassifier

from .commonutils import computeKnnIndicesOfSameClass, check_uniform_label
from .supplementaryclassifiers import AbstractBoostingClassifier


__author__ = 'Alex Rogozhnikov'

# TODO add multi-class classification
# TODO think on the role of weights


class MeanAdaBoostClassifier(AbstractBoostingClassifier):
    def __init__(self,
                 uniform_variables=None,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=10,
                 learning_rate=0.1,
                 n_neighbours=10,
                 uniform_label=1,
                 train_variables=None,
                 voting='mean'):
        """
        Modification of AdaBoostClassifier, has modified reweighting procedure
        (as described in article 'New Approaches for Boosting to Uniformity').
        In the simplest case (voting='mean') we use the mean of neighbours'
        predictions.
        :param uniform_variables: list of variables along which uniformity is desired (i.e. ['mass'])
        :param base_estimator: any sklearn classifier which support weights (i.e. DecisionTreeClassifier())
        :param n_estimators: number of base classifiers to be trained
        :param learning_rate: float, the 'size of step'
        :param n_neighbours: number of neighbours to use
        :param uniform_label:
        :param train_variables: variables to use in training
            (usually these should not include ones from uniform_variables)
        :param (str|callable) voting: string, describes how we use
            predictions of neighbour classifiers. Possible values are:
            'mean', 'median', 'random-percentile', 'random-mean', 'matrix'
            (in the 'matrix' case one should also provide a matrix to fit method.
            Matrix is generalization of )
        """
        self.uniform_variables = uniform_variables
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_neighbours = n_neighbours
        self.uniform_label = uniform_label
        self.train_variables = train_variables
        self.voting = voting

    def fit(self, X, y, sample_weight=None, A=None):
        if self.voting == 'matrix':
            assert A is not None, 'A matrix should be passed'
            assert A.shape == (len(X), len(X)), 'wrong shape of passed matrix'

        self.uniform_label = check_uniform_label(self.uniform_label)
        X, y, sample_weight = self.check_input(X, y, sample_weight)
        y_signed = 2 * y - 1

        knn_indices = computeKnnIndicesOfSameClass(self.uniform_variables, X, y, self.n_neighbours)

        # for those events with non-uniform label we repeat it's own index several times
        for label in [0, 1]:
            if label not in self.uniform_label:
                knn_indices[y == label, :] = numpy.arange(len(y))[y == label][:, numpy.newaxis]

        X = self.get_train_vars(X)
        cumulative_score = numpy.zeros(len(X))
        self.estimators = []

        for stage in range(self.n_estimators):
            knn_scores = numpy.take(cumulative_score, knn_indices)
            if self.voting == 'mean':
                voted_score = numpy.mean(knn_scores, axis=1)
            elif self.voting == 'median':
                voted_score = numpy.median(knn_scores, axis=1)
            elif self.voting == 'random-percentile':
                voted_score = numpy.percentile(knn_scores, numpy.random.random(), axis=1)
            elif self.voting == 'random-mean':
                n_feats = numpy.random.randint(self.n_neighbours//2, self.n_neighbours)
                voted_score = numpy.mean(knn_scores[:, :n_feats], axis=1)
            elif self.voting == 'matrix':
                voted_score = A.dot(cumulative_score)
            else:  # self.voting is callable
                assert not isinstance(self.voting, str), \
                    'unknown value for voting: {}'.format(self.voting)
                voted_score = self.voting(cumulative_score, knn_scores)

            weight = sample_weight * numpy.exp(- y_signed * voted_score)
            weight = self.normalize_weights(y=y, sample_weight=weight)

            classifier = sklearn.clone(self.base_estimator)
            classifier.fit(X, y, sample_weight=weight)
            cumulative_score += self.learning_rate * self.compute_score(classifier, X=X)
            self.estimators.append(classifier)

        return self

    @staticmethod
    def normalize_weights(y, sample_weight):
        sample_weight[y == 0] /= numpy.sum(sample_weight[y == 0])
        sample_weight[y == 1] /= numpy.sum(sample_weight[y == 1])
        return sample_weight




