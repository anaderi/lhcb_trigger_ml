import numpy
from collections import OrderedDict
from hep_ml.grid_search import SimpleParameterOptimizer, AbstractParameterGenerator, GridOptimalSearchCV
from hep_ml.commonutils import generate_sample

__author__ = 'Alex Rogozhnikov'

numpy.random.seed(42)


class FunctionOptimizer(object):
    """Class was created to test different optimizing algorithms on functions,
    it gets any function of several variables and just optimizes it"""

    def __init__(self, function, param_grid, n_evaluations=100, parameter_generator_type=None):
        """
        :type function: some function, we are looking for its maximal value.
        :type parameter_generator_type: (param_grid, n_evaluations) -> AbstractParameterGenerator
        """
        self.function = function
        if parameter_generator_type is None:
            parameter_generator_type = SimpleParameterOptimizer
        self.generator = parameter_generator_type(param_grid, n_evaluations)

    def optimize(self):
        assert isinstance(self.generator, AbstractParameterGenerator), 'the generator should be an instance of ' \
                                                                       'abstract parameter generator'
        for _ in range(self.generator.n_evaluations):
            next_indices, next_params = self.generator.generate_next_point()
            value = self.function(**next_params)
            self.generator.add_result(state_indices=next_indices, value=value)

    def print_results(self, reorder=True):
        self.generator.print_results(reorder=reorder)


def test_simple_optimizer(n_evaluations=60):
    optimizer = FunctionOptimizer(lambda x, y, z, w: x * y * z * w,
                                  param_grid={'x': list(range(11)),
                                              'y': list(range(11)),
                                              'z': list(range(11)),
                                              'w': list(range(11))
                                             },
                                  n_evaluations=n_evaluations)
    optimizer.optimize()
    assert len(optimizer.generator.grid_scores_) == n_evaluations
    assert len(optimizer.generator.queued_tasks_) == n_evaluations
    assert set(optimizer.generator.grid_scores_.keys()) == optimizer.generator.queued_tasks_
    optimizer.print_results()


def test_grid_search():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

    grid = {'base_estimator': [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4),
                               ExtraTreeClassifier(max_depth=4)],
            'learning_rate': [0.01, 0.1, 0.5, 1.],
            'n_estimators': [5, 10, 15, 20, 30, 40, 50, 75, 100, 125],
            'algorithm': ['SAMME', 'SAMME.R']}
    grid = OrderedDict(grid)

    trainX, trainY = generate_sample(2000, 10, distance=0.5)
    grid_cv = GridOptimalSearchCV(AdaBoostClassifier(), grid, n_evaluations=10, refit=True, log_name='test')
    grid_cv.fit(trainX, trainY)
    grid_cv.predict_proba(trainX)
    grid_cv.predict(trainX)
    grid_cv.print_param_stats([0.1, 0.3, 0.5, 0.7])


