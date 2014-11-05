from __future__ import division, print_function, absolute_import

import numpy
import pandas
from matplotlib.cbook import Null
from hep_ml import toymc

__author__ = 'Alex Rogozhnikov'


def test_toy_monte_carlo(size=1000):
    df = pandas.DataFrame(numpy.random.random((size, 40)))
    res = toymc.generate_toymc_with_special_features(df, 5000)
    assert isinstance(res, pandas.DataFrame), "something wrong with MonteCarlo"
    print("toymc is ok")


def test_compare_toymc():
    if __name__ != '__main__':
        toymc.pylab = Null()
    toymc.compare_toymc(pandas.DataFrame(numpy.random.normal(size=(1000, 10))))
