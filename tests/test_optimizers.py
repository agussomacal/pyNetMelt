import os
import numpy as np
import src.networks as networks
import src.integrators as integrators
import src.evaluators as evaluators
from src.optimizers import Optimizer
from hyperopt import hp, STATUS_OK
import unittest


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.max_evals = 100
        self.max_val = 0.2
        self.space = {"x": hp.uniform("x", 0, 1)}

    def test_optimizer(self):
        def obj_func(sp):
            result_dict = dict()
            result_dict['status'] = STATUS_OK
            result_dict['loss'] = -(sp["x"] - self.max_val) ** 2
            result_dict["loss_variance"] = 0
            result_dict["true_loss"] = -(sp["x"] - self.max_val) ** 2
            result_dict["true_loss_variance"] = 0
            return result_dict

        optimizer = Optimizer(optimization_name="test",
                              space=self.space,
                              objective_function=obj_func,
                              max_evals=self.max_evals,
                              append_results=False,
                              path2files=None,  # os.path.dirname(os.path.realpath(__file__))
                              maximize=True)
        tpe_results, best = optimizer.optimize()

        print(best)
        assert np.allclose(best['x'], self.max_val, atol=1/self.max_evals)








