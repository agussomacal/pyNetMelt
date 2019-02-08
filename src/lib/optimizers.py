
import numpy as np
import pandas as pd
# from bayes_opt import BayesianOptimization
from hyperopt import hp, tpe, fmin
from hyperopt import Trials
import os
import pickle



"""
hyperopt
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
"""


########################################################################################################################
class Optimizer:
    """
    TODO: optimization with constraints in gamma. Now there is no sum=1; so normalization should be performed afterwards
    """

    def __init__(self, optimization_name, path2files, space, objective_function, max_evals, maximize=True):
        self.__name__ = optimization_name
        self.path2files = path2files
        self.filename = self.path2files + "/" + self.__name__ + ".json"

        self.space = space
        self.objective_function = objective_function
        self.max_evals = max_evals
        self.maximize = maximize

    @staticmethod
    def get_integrator_space(integrator):
        return {network_name: hp.uniform(network_name, 0, 1) for network_name in integrator.network_names}

    @staticmethod
    def gamma_objective_function(space, evaluator, integrator):
        gamma = Optimizer.get_gamma_from_values([space[network_name] for network_name in integrator.network_names])
        return evaluator.evaluate(integrator.integrate(gamma))

    @staticmethod
    def get_gamma_from_values(values):
        gamma = np.array(values)
        gamma = gamma / gamma.sum()
        return gamma

    @staticmethod
    def normalize_network_gamma_coeficients(tpe_results, network_names):
        tpe_results.loc[:, network_names] = \
            tpe_results.loc[:, network_names].divide(tpe_results.loc[:, network_names].sum(axis=1), axis=0)
        # for ix, row in tpe_results.iterrows():
        #     tpe_results.loc[ix, network_names] = Optimizer.get_gamma_from_values(row[network_names])
        return tpe_results

    def get_trials(self):
        if os.path.exists(self.filename):
            trials = pickle.load(open(self.filename, "rb"))
        else:
            trials = Trials()
        return trials

    def save_trials(self, trials):
        pickle.dump(trials, open(self.filename, "wb"))

    def optimize(self):
        if self.maximize:
            sign = -1
        else:
            sign = 1

        trials = self.get_trials()
        self.max_evals += len(trials)  # actualize the number of trials to perform
        best = fmin(fn=lambda sp: sign*self.objective_function(sp),
                    space=self.space,
                    trials=trials,
                    algo=tpe.suggest,
                    max_evals=self.max_evals)
        self.save_trials(trials)

        tpe_results = {param_name: param_value for param_name, param_value in trials.idxs_vals[1].items()}
        tpe_results[self.__name__] = [sign * x['loss'] for x in trials.results]
        tpe_results = pd.DataFrame(tpe_results)
        return tpe_results, best

    # def randomly(self, integrator, max_evals):
    #
    #     def get_gamma():
    #         gamma = []
    #         probability_rest = 1
    #         for _ in integrator.network_names[:-1]:  # do except the last one
    #             gamma.append(np.random.uniform(0, 1) * probability_rest)
    #             probability_rest = probability_rest * (1 - gamma[-1])
    #         gamma.append(probability_rest)  # the remaining probability goes to te last
    #
    #         return np.array(gamma)
    #
    #     results_cols = integrator.network_names + [self.evaluator.metric_name]
    #     tpe_results = pd.DataFrame([], columns=results_cols)
    #     for i in range(max_evals):
    #         print("\r{}%".format((100 * i) // max_evals), end="\r")
    #         gamma = get_gamma()
    #         temp_result = pd.Series(np.nan, index=results_cols)
    #         temp_result[integrator.network_names] = gamma
    #         temp_result[self.evaluator.metric_name] = self.evaluator.evaluate(integrator.integrate(gamma))
    #
    #         tpe_results = tpe_results.append(temp_result, ignore_index=True)
    #
    #     print("Finished")
    #     best = tpe_results.loc[tpe_results[self.evaluator.metric_name].idxmax(), :]
    #     return tpe_results, best


########################################################################################################################
if __name__=="__main__":
    import numpy as np
    import networks
    import integrators
    import evaluators

    np.random.seed(1)

    laplacian_exponent = -0.5  # exponent of laplacian method
    alpha = 0.2  # alpha of the propagation

    max_iter = 100

    p1 = 0.9
    n_targets = 2
    N = 100
    max_evals = 20
    max_fpr = (1 - p1) / 2

    auroc_normalized = False

    # --------------------------
    # x = np.random.uniform(size=(N, N))
    # network = algorithms.Adjacency(1*((x + x.T) >= 1))
    x = np.roll(np.eye(N), 1, axis=0)
    network_1 = networks.Adjacency(x + x.T)
    network_2 = networks.Adjacency(x)
    x = np.random.uniform(size=(N, N))
    network_3 = networks.Adjacency(1*((x + x.T) > 1))
    # print(network)

    # ---------------------------
    d_networks = {"Net1": network_1, "Net2": network_2}#, "Net3": network_3}
    integrator = integrators.SimpleAdditive(d_networks)
    integrator = integrators.LaplacianAdditive(d_networks, -0.5)

    # --------------------------
    l_seeds = [[seed] for seed in range(N)]  # np.eye(N)
    l_targets = [np.roll(np.arange(N), -n)[1:(n_targets + 1)] for n in range(N)]

    p = np.repeat((1 - p1) / (n_targets - 1), n_targets - 1)
    p = np.insert(p, 0, p1)
    print(p)
    l_true_targets = [[int(np.random.choice(targets, size=1, p=p))] for targets in l_targets]

    print("Target list:")
    print(l_targets)
    print("True targets")
    print(l_true_targets)

    # --------------------------
    evaluator = evaluators.AUROClinkage(node_names=list(range(N)),
                           l_seeds=l_seeds,
                           l_targets=l_targets,
                           l_true_targets=l_true_targets,
                           alpha=alpha,
                           l_seeds_weight=None,
                           tol=1e-08,
                           max_iter=max_iter,
                           max_fpr=max_fpr,
                           laplacian_exponent=laplacian_exponent,
                           auroc_normalized=False)
    print(evaluator.metric_name)



    optimizer = Optimizer(optimization_name=evaluator.metric_name+"_"+integrator.__name__,
                         path2files="/home/crux/Downloads",
                         space=Optimizer.get_integrator_space(integrator=integrator),
                         objective_function=lambda sp: Optimizer.gamma_objective_function(sp,
                                                                                         evaluator=evaluator,
                                                                                         integrator=integrator),
                         max_evals=max_evals,
                         maximize=True)
    tpe_results, best = optimizer.optimize()
    print(tpe_results)
    print(best)

    # print((5*p1-4*p1**2)/8)
    print(p1*(1-p1)/8)



