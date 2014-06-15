from __future__ import print_function, division

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy
import pandas
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone

__author__ = "Alex Rogozhnikov"


class HidingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, train_variables=None, base_estimator=None):
        """This is a dumb meta-classifier, which uses only subset of variables to train
        base classifier and to estimate result.
        It is intended to wrap classifier sklearn classifiers, because they use all features by default"""
        self.train_variables = train_variables
        self.base_estimator = base_estimator

    def fit(self, X, y):
        assert self.base_estimator is not None, "base estimator was not set"
        self._trained_estimator = clone(self.base_estimator)
        self._trained_estimator.fit(X[self.train_variables], y)

    def predict(self, X):
        return self._trained_estimator.predict(X[self.train_variables])

    def predict_proba(self, X):
        return self._trained_estimator.predict_proba(X[self.train_variables])

    def staged_predict_proba(self, X):
        return self._trained_estimator.staged_predict_proba(X[self.train_variables])


class FeatureSplitter(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_name, base_estimators, final_estimator):
        """ The dataset is supposed to have some special variable, and depending on this variable,
        the event has some set of features. For each pair of values we use common features to train
        additional variables
        :param feature_name: str, the name of key feature
        :param base_estimators: dict, the classifiers used to generate features
        :param final_estimator: BaseEstimator, the classifier used to make final decision
        """
        self.base_estimators = base_estimators
        self.feature_name = feature_name
        self.final_estimator = final_estimator

    def fit(self, X, y, sample_weight=None):
        assert isinstance(self.base_estimators, dict), 'Estimators should be passed in a dictionary'
        assert len(X) == len(y), 'the lengths are different'
        assert sample_weight is None or len(sample_weight) == len(y), 'the lengths are different'
        if sample_weight is None:
            sample_weight = numpy.ones(len(y))
        assert self.feature_name in X.columns, 'there is no feature %s' % self.feature_name
        self.columns_order = X.columns

        column = numpy.array(X[self.feature_name])
        self.column_values = list(set(column))
        self.stayed_columns = dict()        # value -> list of columns
        self.common_features = dict()       # (value_from, value_to) -> list of columns
        self.classifiers = dict()           # (value_from, value_to, classifier_name) -> classifier
        self.final_classifiers = dict()     # (value, classifier_name) -> classifier
        rows_dict = dict()                  # (value) -> boolean list of rows
        self.final_columns_orders = dict()  # (value) -> list of features
        for value in self.column_values:
            rows = numpy.array(X[self.feature_name] == value)
            rows_dict[value] = rows
            x_part = X.loc[rows, :]
            cols = pandas.notnull(x_part).all()
            self.stayed_columns[value] = cols[cols==True].keys()

        for value_to, rows_to in rows_dict.iteritems():
            columns_to = self.stayed_columns[value_to]
            new_features = pandas.DataFrame()
            for value_from, rows_from in rows_dict.iteritems():
                if value_from == value_to:
                    continue
                common_columns = list(set(self.stayed_columns[value_from]).union(set(self.stayed_columns[value_to])))
                common_columns.remove(self.feature_name)
                self.common_features[value_from, value_to] = common_columns
                for name, estimator in self.base_estimators.iteritems():
                    rows_from = rows_dict[value_from]
                    new_classifier = sklearn.clone(estimator)\
                        .fit(X.loc[rows_from, common_columns], y[rows_from], sample_weight=sample_weight[rows_from])

                    self.classifiers[value_from, value_to, name] = new_classifier
                    new_feature = new_classifier.predict_proba(X.loc[rows_to, common_columns])[:, 1]
                    new_features[str(value_from) + "_" + name] = new_feature
            X_to_part = X.loc[rows_to, columns_to]
            new_features = new_features.set_index(X_to_part.index)
            X_to_part = pandas.concat([X_to_part, new_features], axis=1)
            final_classifier = sklearn.clone(self.final_estimator)
            final_classifier.fit(X_to_part, y[rows_to], sample_weight=sample_weight[rows_to])
            self.final_columns_orders[value_to] = X_to_part.columns
            self.final_classifiers[value_to] = final_classifier
        return self

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        assert isinstance(X, pandas.DataFrame), 'should be a dataframe'
        result = numpy.zeros([len(X), 2])
        assert numpy.all( self.columns_order == numpy.array(X.columns) ), \
            "different columns in test and train: \n{0}\n{1} ".format(self.columns_order, X.columns)
        rows_dict = dict()                  # (value) -> boolean list of rows
        for value in self.column_values:
            rows_dict[value] = numpy.array(X[self.feature_name] == value)

        for value_to, rows_to in rows_dict.iteritems():
            columns_to = self.stayed_columns[value_to]
            new_features = pandas.DataFrame()
            for value_from in rows_dict:
                if value_to == value_from:
                    continue
                common_columns = self.common_features[value_from, value_to]
                for name in self.base_estimators:
                    new_classifier = self.classifiers[value_from, value_to, name]
                    new_feature = new_classifier.predict_proba(X.loc[rows_to, common_columns])[:, 1]
                    new_features[str(value_from) + "_" + name] = new_feature
            X_to_part = X.loc[rows_to, columns_to]
            new_features = new_features.set_index(X_to_part.index)
            X_to_part = pandas.concat([X_to_part, new_features], axis=1)
            assert numpy.all(self.final_columns_orders[value_to] == numpy.array(X_to_part.columns)), \
                'the lists of features are different:\n{0}\n{1}'.\
                    format(self.final_columns_orders[value_to], X_to_part.columns)

            result[numpy.array(rows_to), :] = self.final_classifiers[value_to].predict_proba(X_to_part)
        return result

    def stage_predict_proba(self, X):
        # It is difficult to give some meaningful staged prediction
        raise NotImplementedError()


class DumbSplitter(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_name=None, base_estimator=None):
        """
        Splits the dataset by specific column, trains separately on each part
        :param feature_name: the name of the column
        :param base_estimator: the classifier used to classify each part of data
        """
        self.feature_name = feature_name
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weight=None):
        assert len(X) == len(y), 'the lengths are different'
        if sample_weight is None:
            sample_weight = numpy.ones(len(y))
        assert len(y) == len(sample_weight), 'the lengths are different'
        assert self.feature_name in X.columns, 'no such feature in index'
        self.values = set(X.loc[:, self.feature_name])
        self.classifiers = dict()
        for value in self.values:
            rows = numpy.array(X[self.feature_name] == value)
            new_classifier = sklearn.clone(self.base_estimator).fit(X.loc[rows], y[rows], sample_weight[rows])
            self.classifiers[value] = new_classifier
        return self

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        result = numpy.zeros([len(X), 2])
        assert self.feature_name in X.columns, 'no such feature in index'
        for value in self.values:
            rows = X[self.feature_name] == value
            result[numpy.array(rows), :] = self.classifiers[value].predict_proba(X.loc[rows])
        return result


class ChainClassifiers(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators=None):
        # TODO the bad thing is we use the same dataset for train and the for prediction.
        # smearing or toymc can probably solve address issue
        self.base_estimators = base_estimators

    def fit(self, X, y, sample_weight=None):
        assert isinstance(self.base_estimators, OrderedDict)
        X = pandas.DataFrame(X).copy()
        assert len(X) == len(y), 'lengths are different'
        self.trained_estimators = OrderedDict()
        for name, classifier in self.base_estimators.iteritems():
            new_classifier = sklearn.clone(classifier)
            new_classifier.fit(X, y, sample_weight)
            X['new_'+name] = new_classifier.predict_proba(X)[:, 1]
            self.trained_estimators[name] = new_classifier
        return self

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = pandas.DataFrame(X).copy()
        result = None
        for name, classifier in self.trained_estimators.iteritems():
            result = classifier.predict_proba(X)
            X['new_'+name] = result[:, 1]
        return result


def test_feature_splitter(size=2000):
    import commonutils
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.lda import LDA
    from sklearn.qda import QDA

    X, y = commonutils.generateSample(size, 10, distance=0.5)
    X['column0'] = numpy.clip(numpy.array(X['column0']).astype(numpy.int), -2, 2)
    trainX, testX, trainY, testY = commonutils.my_train_test_split(X, y)
    base_estimators = {'rf': RandomForestClassifier()}
    splitter = FeatureSplitter('column0', base_estimators=base_estimators, final_estimator=RandomForestClassifier())
    splitter.fit(trainX, trainY)
    print(splitter.score(testX, testY))
    print(RandomForestClassifier().fit(trainX, trainY).score(testX, testY))
    print(DumbSplitter('column0', base_estimator=RandomForestClassifier()).fit(trainX, trainY).score(testX, testY))
    chain = OrderedDict()
    chain['QDA'] = QDA()
    chain['LDA'] = LDA()
    chain['RF'] = RandomForestClassifier()
    print(ChainClassifiers(chain).fit(trainX, trainY).score(testX, testY))
    print(LDA().fit(trainX, trainY).score(testX, testY))

test_feature_splitter()



