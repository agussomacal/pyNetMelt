import numpy as np
import src.networks as networks
import src.integrators as integrators
import src.evaluators as evaluators
from src.optimizers import Optimizer
import unittest


class TestEvaluators:

    def tearDown(self):
        np.random.seed(1)

    def test_optimizer(self):

        np.random.seed(1)

        laplacian_exponent = -0.5  # exponent of laplacian method
        alpha = 0.2  # alpha of the propagation

        max_iter = 100

        p1 = 0.9
        n_targets = 2
        N = 100
        max_evals = 20
        max_fpr = (1 - p1) / 2

        # --------------------------
        # x = np.random.uniform(size=(N, N))
        # network = algorithms.Adjacency(1*((x + x.T) >= 1))
        x = np.roll(np.eye(N), 1, axis=0)
        network_1 = networks.Adjacency(x + x.T)
        network_2 = networks.Adjacency(x)
        x = np.random.uniform(size=(N, N))
        network_3 = networks.Adjacency(1 * ((x + x.T) > 1))
        # print(network)

        # ---------------------------
        d_networks = {"Net1": network_1, "Net2": network_2}  # , "Net3": network_3}
        integrator = integrators.SimpleAdditive(d_networks)
        integrator = integrators.LaplacianAdditive(d_networks, -0.5)

        # --------------------------
        l_seeds_dict = [{seed: 1} for seed in range(N)]  # np.eye(N)
        l_targets = [np.roll(np.arange(N), -n)[1:(n_targets + 1)] for n in range(N)]

        p = np.repeat((1 - p1) / (n_targets - 1), n_targets - 1)
        p = np.insert(p, 0, p1)
        l_true_targets = [[int(np.random.choice(targets, size=1, p=p))] for targets in l_targets]

        # --------------------------
        evaluator = evaluators.AUROClinkage(node_names=list(range(N)),
                                            l_seeds_dict=l_seeds_dict,
                                            l_targets=l_targets,
                                            l_true_targets=l_true_targets,
                                            alpha=alpha,
                                            tol=1e-08,
                                            max_iter=max_iter,
                                            max_fpr=max_fpr,
                                            laplacian_exponent=laplacian_exponent,
                                            auroc_normalized=False,
                                            k_fold=3)

        optimizer = Optimizer(optimization_name=evaluator.metric_name + "_" + integrator.__name__,
                              path2files="/home/crux/Downloads",
                              space=Optimizer.get_integrator_space(integrator=integrator),
                              objective_function=lambda sp: Optimizer.gamma_objective_function(sp,
                                                                                               evaluator=evaluator,
                                                                                               integrator=integrator),
                              max_evals=max_evals,
                              maximize=True)
        tpe_results, best = optimizer.optimize()

        assert best == {'Net1': 0.6158253096757701, 'Net2': 0.17713374846173457}







