from __future__ import division, print_function, absolute_import

import numpy
from hep_ml import losses
from hep_ml.commonutils import generate_sample

__author__ = 'Alex Rogozhnikov'


def test_loss_functions(size=50, epsilon=1e-3):
    """
    Testing that hessians and gradients of loss functions coincide with numerical
    """
    X, y = generate_sample(size, n_features=10)
    sample_weight = numpy.random.exponential(size=size)
    tested_losses = [
        losses.BinomialDevianceLossFunction(),
        losses.AdaLossFunction(),
        losses.SimpleKnnLossFunction(X.columns[:1], knn=5),
        losses.CompositeLossFunction()
    ]
    pred = numpy.random.normal(size=size)

    for loss in tested_losses:
        loss.fit(X, y, sample_weight=sample_weight)
        # testing sign of gradient
        val = loss(pred)
        gradient = loss.negative_gradient(pred)
        hessian = loss.hessian(pred)
        assert (gradient * (2 * y - 1) >= 0).all()

        numer_gradient = numpy.zeros(len(pred))
        numer_hessian = numpy.zeros(len(pred))
        for i in range(size):
            pred_plus = pred.copy()
            pred_plus[i] += epsilon
            val_plus = loss(pred_plus)

            pred_minus = pred.copy()
            pred_minus[i] -= epsilon
            val_minus = loss(pred_minus)

            numer_gradient[i] = - (val_plus - val_minus) / 2. / epsilon
            numer_hessian[i] = (val_plus + val_minus - 2 * val) / epsilon ** 2

        assert numpy.allclose(gradient, numer_gradient), 'wrong computation of gradient'
        assert numpy.allclose(hessian, numer_hessian), 'wrong computation of hessian'