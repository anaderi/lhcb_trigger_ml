from __future__ import division, print_function

import copy
import numbers
import numpy
import pandas
import sklearn
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.ensemble.gradient_boosting import LossFunction
from sklearn.tree.tree import DecisionTreeRegressor, DTYPE
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_arrays, column_or_1d, array2d

from hep_ml.commonutils import check_sample_weight, generate_sample, map_on_cluster, indices_of_values
from hep_ml.losses import AbstractLossFunction
from transformations import enhance_data, Shuffler

real_s = 691.988607712
real_b = 410999.847322


#region Functions for measurements

def get_higgs_data(train_file = '/Users/axelr/ipython/datasets/higgs/training.csv'):
    data = pandas.read_csv(train_file, index_col='EventId')
    answers_bs = numpy.ravel(data.Label)
    weights = numpy.ravel(data.Weight)
    data = data.drop(['Label', 'Weight'], axis=1)
    answers = numpy.zeros(len(answers_bs), dtype=numpy.int)
    answers[answers_bs == 's'] = 1
    return data, answers, weights


def AMS(answers, predictions, sample_weight):
    """ Predictions are classes """
    assert len(answers) == len(predictions) == len(sample_weight)
    predictions = column_or_1d(predictions)
    total_s = numpy.sum(sample_weight[answers > 0.5])
    total_b = numpy.sum(sample_weight[answers < 0.5])
    s = numpy.sum(sample_weight[answers * predictions > 0.5])
    b = numpy.sum(sample_weight[(1 - answers) * predictions > 0.5])
    s *= real_s / total_s
    b *= real_b / total_b
    br = 10.
    radicand = 2 * ( (s+b+br) * numpy.log(1.0 + s/(b+br)) - s)
    if radicand < 0:
        raise ValueError('Radicand is negative')
    else:
        return numpy.sqrt(radicand)


def compute_ams_on_cuts(answers, predictions, sample_weight):
    """ Prediction is probabilities"""
    assert len(answers) == len(predictions) == len(sample_weight)
    answers = column_or_1d(answers)
    predictions = column_or_1d(predictions)
    sample_weight = column_or_1d(sample_weight)
    order = numpy.argsort(predictions)[::-1]
    reordered_answers = answers[order]
    reordered_weights = sample_weight[order]
    s_cumulative = numpy.cumsum(reordered_answers * reordered_weights)
    b_cumulative = numpy.cumsum((1 - reordered_answers) * reordered_weights)
    b_cumulative *= real_b / b_cumulative[-1]
    s_cumulative *= real_s / s_cumulative[-1]
    br = 10.
    s = s_cumulative
    b = b_cumulative
    radicands = 2 * ((s + b + br) * numpy.log(1.0 + s/(b + br)) - s)
    return predictions[order], radicands


def optimal_AMS(answers, predictions, sample_weight):
    """ Prediction is probabilities """
    cuts, radicands = compute_ams_on_cuts(answers, predictions, sample_weight)
    return numpy.sqrt(numpy.max(radicands))


def plot_ams_report(answers, predictions, sample_weight=None):
    import pylab

    cuts, radicands = compute_ams_on_cuts(answers, predictions, sample_weight)
    pylab.figure(figsize=(18, 9))
    pylab.subplot(131)
    pylab.title('On cuts')
    pylab.plot(cuts, numpy.sqrt(numpy.clip(radicands, 0, 100)))
    pylab.subplot(132)
    pylab.title('On signal order')
    order = numpy.argsort(predictions)[::-1]
    pylab.plot( numpy.sqrt(numpy.clip(radicands[answers[order] > 0.5], 0, 100)) )
    pylab.subplot(133)
    pylab.title('On common order')
    pylab.plot( numpy.sqrt(radicands) )


def plot_AMS_on_cuts(answers, predictions, sample_weight):
    """ Prediction is probabilities """
    import pylab
    cuts, radicands = compute_ams_on_cuts(answers, predictions, sample_weight)
    pylab.plot(cuts, numpy.sqrt(numpy.clip(radicands, 0, 100)))


def plot_AMS_on_signal_order(answers, predictions, sample_weight):
    """ Prediction is probabilities """
    import pylab
    cuts, radicands = compute_ams_on_cuts(answers, predictions, sample_weight)
    order = numpy.argsort(predictions)[::-1]
    pylab.plot( numpy.sqrt(numpy.clip(radicands[answers[order] > 0.5], 0, 100)) )

#endregion


#region Losses

class MyLossFunction(BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        pass

    def negative_gradient(self, y, y_pred, sample_weight=None):
        raise NotImplementedError()

    def update_terminal_regions(self, tree, X, y, residual, pred, sample_mask, sample_weight):
        assert y.ndim == 1 and residual.ndim == 1 and \
               pred.ndim == 1 and sample_mask.ndim == 1 and sample_weight.ndim == 1

        # residual is negative gradient
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        for leaf, leaf_indices in indices_of_values(masked_terminal_regions):
            if leaf == -1:
                continue
            self._update_terminal_region(tree, terminal_regions=masked_terminal_regions,
                                         leaf=leaf, X=X, y=y, residual=residual, pred=pred,
                                         sample_weight=sample_weight, leaf_indices=leaf_indices)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight, leaf_indices):
        """This function should select a better values for leaves"""
        pass


class LogitLossFunction(MyLossFunction):
    def __init__(self, shift=0.):
        MyLossFunction.__init__(self)
        self.shift = shift

    def __call__(self, y, y_pred, sample_weight=None):
        y_signed = 2. * y - 1
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        return numpy.sum(sample_weight * numpy.log(1 + numpy.exp(- y_signed * y_pred - self.shift)))

    def negative_gradient(self, y, y_pred, sample_weight=None):
        y_signed = 2. * y - 1
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        return sample_weight * y_signed * expit(-y_signed * y_pred - self.shift)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight, leaf_indices):
        """Making one Newton step"""
        # terminal_region = numpy.where(terminal_regions == leaf)[0]
        terminal_region = leaf_indices
        y = y.take(terminal_region, axis=0)
        y_signed = 2. * y - 1
        pred = pred.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region)
        argument = -y_signed * pred - self.shift
        n_gradient = numpy.sum(sample_weight * y_signed * expit(argument))
        laplacian = numpy.sum(sample_weight / numpy.logaddexp(0., argument) / numpy.logaddexp(0., -argument))
        tree.value[leaf, 0, 0] = n_gradient / laplacian


class AdaLossFunction(MyLossFunction):
    def __init__(self, signal_curvature=1.):
        self.signal_curvature = signal_curvature
        # we need only one variable
        MyLossFunction.__init__(self)

    def fit(self, X, y, sample_weight=None):
        pass

    def _signed_multiplier(self, y):
        result = numpy.ones(len(y), dtype=float)
        result[y > 0.5] = - self.signal_curvature
        return result

    def _weight_multiplier(self, y):
        result = numpy.ones(len(y), dtype=float)
        result[y > 0.5] = 1 / self.signal_curvature
        return result

    def __call__(self, y, y_pred, sample_weight=None):
        signed_multiplier = self._signed_multiplier(y)
        weight_multiplier = self._weight_multiplier(y)
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        return numpy.sum(sample_weight * weight_multiplier * numpy.exp(y_pred * signed_multiplier))

    def negative_gradient(self, y, y_pred, sample_weight=None, **kargs):
        multiplier = self._signed_multiplier(y)
        y_signed = 2. * y - 1
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        return sample_weight * y_signed * numpy.exp(y_pred * multiplier)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight, leaf_indices):
        terminal_region = leaf_indices

        curv = self.signal_curvature
        y = y.take(terminal_region, axis=0)
        pred = pred.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region)
        w_sig = numpy.sum(sample_weight[y > 0.5] * numpy.exp(- curv * pred[y > 0.5]))
        w_bck = numpy.sum(sample_weight[y < 0.5] * numpy.exp(pred[y < 0.5]))
        # minimizing w_sig * exp(-curv * x) / curv + w_bck * exp(x)
        w_sum = w_sig + w_bck
        w_sig += 1e-4 * w_sum
        w_bck += 1e-4 * w_sum
        tree.value[leaf, 0, 0] = 1 / (1. + curv) * numpy.log(w_sig / w_bck)


#endregion


#region Interpolation

def interpolate(vals, step, steps, use_log=False):
    if isinstance(vals, numbers.Number):
        return vals
    t = numpy.clip(step / float(steps), 0, 1)
    assert len(vals) == 2, 'Not two values'
    if use_log:
        return numpy.exp(numpy.interp(t, [0., 1.], numpy.log(vals)))
    else:
        return numpy.interp(t, [0., 1.], vals)

#endregion


#region GradientBoosting

class GradientBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, loss,
                 n_estimators=10,
                 learning_rate=1.,
                 max_depth=15,
                 min_samples_leaf=5,
                 min_samples_split=2,
                 max_features='auto',
                 subsample=1.,
                 criterion='mse',
                 splitter='best',
                 weights_in_loss=True,
                 update_tree=True,
                 update_on='all',
                 smearing=0.0,
                 init_estimator=None,
                 init_smearing=0.05,
                 recount_step=1000,
                 random_state=None):
        """
        Supports only two classes
        :type loss: LossFunction
        :type n_estimators: int, 
        :type learning_rate: float,
        :type max_depth: int | NoneType,
        :type min_samples_leaf: int,
        :type min_samples_split: int,
        :type max_features: int | 'auto',
        :type subsample: float,
        :type splitter: str,
        :type weights_in_loss: bool,
        :type update_on: str, 'all', 'same', 'other', 'random'
        :type smearing: float
        :type init_smearing: float
        :rtype:
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.subsample = subsample
        self.splitter = splitter
        self.criterion = criterion
        self.weights_in_loss = weights_in_loss
        self.random_state = random_state
        self.update_tree = update_tree
        self.update_on = update_on
        self.smearing = smearing
        self.init_estimator = init_estimator
        self.init_smearing = init_smearing
        self.recount_step = recount_step  # at these iterations additionally penalized bg events with decision > 0

    def fit(self, X, y, sample_weight=None):
        shuffler = Shuffler(X, random_state=self.random_state)
        X, y = check_arrays(X, y, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
        y = column_or_1d(y, warn=True)
        n_samples = len(X)
        n_inbag = int(self.subsample * n_samples)
        sample_weight = check_sample_weight(y, sample_weight=sample_weight).copy()
        self.random_state = check_random_state(self.random_state)

        # skipping all checks
        assert self.update_on in ['all', 'same', 'other', 'random']
        y_pred = numpy.zeros(len(y), dtype=float)
        if self.init_estimator is not None:
            self.init_estimator = sklearn.clone(self.init_estimator)
            self.init_estimator.fit(shuffler.generate(self.init_smearing), y,
                                    sample_weight=sample_weight)
            y_pred += self._proba_to_score(self.init_estimator.predict_proba(X))

        self.classifiers = []
        self.learning_rates = []
        self.loss_values = []
        self.loss = copy.copy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)
        iter_X = shuffler.generate(0.)

        prev_smearing = 1
        for iteration in range(self.n_estimators):
            if iteration % self.recount_step == 0:
                if prev_smearing > 0:
                    iter_smearing = interpolate(self.smearing, iteration, self.n_estimators)
                    prev_smearing = iter_smearing
                    iter_X = shuffler.generate(iter_smearing)
                    iter_X, = check_arrays(iter_X, dtype=DTYPE, sparse_format="dense", check_ccontiguous=True)
                    y_pred = numpy.zeros(len(y))
                    y_pred += sum(cl.predict(X) * rate for rate, cl in zip(self.learning_rates, self.classifiers))


            self.loss_values.append(self.loss(y, y_pred, sample_weight=sample_weight))
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=interpolate(self.max_depth, iteration, self.n_estimators),
                min_samples_split=self.min_samples_split,
                min_samples_leaf=interpolate(self.min_samples_leaf, iteration, self.n_estimators, use_log=True),
                max_features=self.max_features,
                random_state=self.random_state)

            sample_mask = _random_sample_mask(n_samples, n_inbag, self.random_state)
            loss_weight = sample_weight if self.weights_in_loss else numpy.ones(len(sample_weight))
            tree_weight = sample_weight if not self.weights_in_loss else numpy.ones(len(sample_weight))
            residual = self.loss.negative_gradient(y, y_pred, sample_weight=loss_weight)

            tree.fit(numpy.array(iter_X)[sample_mask, :],
                     residual[sample_mask],
                     sample_weight=tree_weight[sample_mask], check_input=False)
            # update tree leaves
            if self.update_tree:
                if self.update_on == 'all':
                    update_mask = numpy.ones(len(sample_mask), dtype=bool)
                elif self.update_on == 'same':
                    update_mask = sample_mask
                elif self.update_on == 'other':
                    update_mask = ~sample_mask
                else:  # random
                    update_mask = _random_sample_mask(n_samples, n_inbag, self.random_state)
                self.loss.update_terminal_regions(tree.tree_, X=iter_X, y=y, residual=residual, pred=y_pred,
                                                  sample_mask=update_mask, sample_weight=sample_weight)
            iter_learning_rate = interpolate(self.learning_rate, iteration, self.n_estimators, use_log=True)
            y_pred += iter_learning_rate * tree.predict(X)
            self.classifiers.append(tree)
            self.learning_rates.append(iter_learning_rate)

        return self

    def decision_function(self, X):
        X = array2d(X, dtype=DTYPE)
        result = numpy.zeros(len(X))
        if self.init_estimator is not None:
            result += self._proba_to_score(self.init_estimator.predict_proba(X))
        for rate, estimator in zip(self.learning_rates, self.classifiers):
            result += rate * estimator.predict(X)
        return result

    def staged_decision_function(self, X):
        X = array2d(X, dtype=DTYPE)
        result = numpy.zeros(len(X))
        if self.init_estimator is not None:
            result += self._proba_to_score(self.init_estimator.predict_proba(X))
        yield result
        for rate, classifier in zip(self.learning_rates, self.classifiers):
            result += rate * classifier.predict(X)
            yield result

    @staticmethod
    def _score_to_proba(score):
        result = numpy.zeros([len(score), 2], dtype=float)
        result[:, 1] = expit(score / 100.)
        result[:, 0] = 1. - result[:, 1]
        return result

    def _proba_to_score(self, proba):
        # for init_estimator
        return numpy.clip(logit(proba[:, 1]), -5., 5.)

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return self._score_to_proba(self.decision_function(X))

    def staged_predict_proba(self, X):
        for score in self.staged_decision_function(X):
            yield self._score_to_proba(score)


def test_gradient_boosting(size=100, n_features=10):
    trainX, trainY = generate_sample(size, n_features)
    testX, testY = generate_sample(size, n_features)
    for loss in [AdaLossFunction()]:
        for update in ['all', 'same', 'other', 'random']:
            gb = GradientBoosting(loss=loss, update_on=update, smearing=[0.1, -0.1])
            score = gb.fit(trainX, trainY).score(testX, testY)
            if __name__ == "__main__":
                print(update, score)

test_gradient_boosting()

#endregion


#region Reweighters

def normalize_weight(y, weights, sig_weight=1., pow_sig=1., pow_bg=1.):
    result = numpy.copy(weights)
    assert numpy.all((y == 0) | (y == 1)), 'Supports only two classes'
    result[y == 1] **= pow_sig
    result[y == 0] **= pow_bg

    result[y == 1] /= numpy.mean(result[y == 1]) / sig_weight
    result[y == 0] /= numpy.mean(result[y == 0])
    return result


class ReweightingGB(GradientBoosting):
    def __init__(self, loss,
                 sig_weight=1., pow_sig=1., pow_bg=1.,
                 n_estimators=10, learning_rate=1., max_depth=None, min_samples_leaf=5, min_samples_split=2,
                 max_features='auto', criterion='mse',
                 subsample=1., splitter='best', weights_in_loss=True, update_tree=True,
                 update_on='all', smearing=0.01,
                 init_estimator=None, init_smearing=0.05, recount_step=1000, random_state=None):
        GradientBoosting.__init__(self, loss=loss, n_estimators=n_estimators, learning_rate=learning_rate,
                                  max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split, max_features=max_features, criterion=criterion,
                                  subsample=subsample, splitter=splitter, weights_in_loss=weights_in_loss,
                                  update_on=update_on, update_tree=update_tree,  random_state=random_state,
                                  init_estimator=init_estimator, init_smearing=init_smearing,
                                  recount_step=recount_step,
                                  smearing=smearing)
        # Everything should be set via set_params
        self.sig_weight = sig_weight
        self.pow_bg = pow_bg
        self.pow_sig = pow_sig

    def fit(self, X, y, sample_weight=None):
        sample_weight = normalize_weight(y, sample_weight, sig_weight=self.sig_weight, pow_sig=self.pow_sig,
                                         pow_bg=self.pow_bg)
        return GradientBoosting.fit(self, X, y, sample_weight=sample_weight)







base_gb = ReweightingGB(loss=AdaLossFunction())
base_gb.set_params(loss__signal_curvature=0.7, learning_rate=0.03, min_samples_leaf=125, n_estimators=400,
                   smearing=0.01, max_features=13, update_tree=True, max_depth=16, subsample=0.5,
                   sig_weight=0.1, weights_in_loss=False, update_on='all')

base_gb_short = ReweightingGB(loss=AdaLossFunction())
base_gb_short.set_params(loss__signal_curvature=0.7, learning_rate=0.03, min_samples_leaf=150, n_estimators=500,
                   smearing=0.0, max_features=16, update_tree=True, max_depth=14, subsample=0.4,
                   sig_weight=0.1, weights_in_loss=False, update_on='all')


base_gb_no_shuffle = ReweightingGB(loss=AdaLossFunction())
base_gb_no_shuffle.set_params(loss__signal_curvature=0.7, learning_rate=0.03, min_samples_leaf=125, n_estimators=250,
                   smearing=0., max_features=13, update_tree=True, max_depth=16, subsample=0.5,
                   sig_weight=0.1, weights_in_loss=False, update_on='all')


base_gb_test = ReweightingGB(loss=AdaLossFunction())
base_gb_test.set_params(loss__signal_curvature=0.7, learning_rate=0.03, min_samples_leaf=125, n_estimators=1,
                        smearing=0.01, max_features=15, update_tree=True, max_depth=16, subsample=0.5,
                        sig_weight=0.1, weights_in_loss=False, update_on='all')



#endregion



"""
import gradient_boosting as gb
data, y, w = gb.get_higgs_data()
voter = gb.base_gb
voter.set_params(n_estimators=10)
voter.fit(gb.enhance_data(data), y, w)

"""
