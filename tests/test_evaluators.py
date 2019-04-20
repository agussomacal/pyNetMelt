from lib.evaluators import *
import numpy as np
import unittest


class TestEvaluators(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_AUROClinkage(self):
        np.random.seed(1)
        laplacian_exponent = -0.5  # exponent of laplacian method
        alpha = 0.2  # alpha of the propagation

        max_iter = 100

        p1 = 0.8
        n_targets = 2
        N = 2000
        max_evals = 1
        max_fpr = (1-p1)/2

        # --------------------------
        x = np.roll(np.eye(N), 1, axis=0)
        network = networks.Adjacency(x)

        # --------------------------
        l_seeds_dict = [{seed: 1} for seed in range(N)]#np.eye(N)
        l_targets = [np.roll(np.arange(N), -n)[1:(n_targets + 1)] for n in range(N)]

        p = np.repeat((1 - p1) / (n_targets - 1), n_targets - 1)
        p = np.insert(p, 0, p1)
        l_true_targets = [[int(np.random.choice(targets, size=1, p=p))] for targets in l_targets]


        # --------------------------
        evalauc = AUROClinkage(node_names=network.node_names,
                               l_seeds_dict=l_seeds_dict,
                               l_targets=l_targets,
                               l_true_targets=l_true_targets,
                               alpha=alpha,
                               tol=1e-08,
                               max_iter=max_iter,
                               max_fpr=max_fpr,
                               laplacian_exponent=laplacian_exponent,
                               auroc_normalized=False)
        auc = evalauc.evaluate(network)
        print(auc)
        assert np.allclose(auc["train"], 0.01841, atol=0.01)
        assert np.allclose(auc["train std"], 0.00335, atol=0.01)
        assert np.allclose(auc["test"], 0.01939, atol=0.01)
        assert np.allclose(auc["test std"], 0.00335, atol=0.01)
        assert np.allclose(auc["test"], p1*(1-p1)/8, atol=0.01)







