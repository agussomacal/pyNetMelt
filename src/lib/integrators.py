
import numpy as np
import pandas as pd
# from bayes_opt import BayesianOptimization
from hyperopt import hp, tpe, fmin
from hyperopt import Trials


import algorithms

"""
hyperopt
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
"""


########################################################################################################################
class Integrator:
    def __init__(self, dict_nets_to_integrate, transformation_function, operation_function):
        self.l_nets_to_integrate = list(dict_nets_to_integrate.values())
        self.network_names = list(dict_nets_to_integrate.keys())
        self.transformation_function = transformation_function
        self.operation_function = operation_function

    def integrate(self, gamma):
        w = self.transformation_function(self.l_nets_to_integrate[0]) * gamma[0]
        for g, net in zip(gamma[1:], self.l_nets_to_integrate[1:]):
            w = self.operation_function(w, self.transformation_function(net) * g)

        # w.network_mode = self.l_nets_to_integrate[0].network_mode # same network_mode, laplacian or adjacency as the ones in the combination
        return w


class AdditiveMethods(Integrator):
    @staticmethod
    def operation_function(w, net):
        return w + net

    def __init__(self, l_nets_to_integrate, transformation_function):
        Integrator.__init__(self, l_nets_to_integrate, transformation_function, self.operation_function)


class MultiplicativeMethods(Integrator):
    @staticmethod
    def operation_function(w, net):
        return w * net

    def __init__(self, l_nets_to_integrate, transformation_function):
        Integrator.__init__(self, l_nets_to_integrate, transformation_function, self.operation_function)


class LaplacianMethods(Integrator):
    """
    This class incorporates the method to recover the possible adjacency matrix that could generate the laplacian.
    """
    def __init__(self, laplacian_exponent):
        self.laplacian_exponent = laplacian_exponent

    def transformation_function(self, net):
        net.to_laplacian(self.laplacian_exponent)
        return net

    def find_adjacency(self, w, max_iter=100):
        if w.strength is None:
            w.strength = np.ones(w.matrix.shape[0]) # initialize all strength in one.

        for i in range(max_iter):
            new_strength = algorithms.normalize_by_strength(w.matrix, -self.laplacian_exponent, w.strength, exponent_sign=1).sum(axis=0)
            new_strength = new_strength/new_strength.sum()*w.strength.sum()
            if np.allclose(new_strength, w.strength):
                break
            w.strength = new_strength

        w.to_adjacency()
        return w


class SimpleMethods(Integrator):
    """
    This class incorporates the method to recover the possible adjacency matrix that could generate the laplacian.

    TODO: code correctly the find adjacency method
    """

    @staticmethod
    def transformation_function(net):
        net.to_adjacency()
        return net


# ------------------------------------------------------
# integration methods upper classes

class SimpleMultiplicative(MultiplicativeMethods, SimpleMethods):
    __name__ = "SimpleMultilicative"

    def __init__(self, l_nets_to_integrate):
        MultiplicativeMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


class SimpleAdditive(AdditiveMethods, SimpleMethods):
    __name__ = "SimpleAdditive"

    def __init__(self, l_nets_to_integrate):
        AdditiveMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


class LaplacianMultiplicative(MultiplicativeMethods, LaplacianMethods):
    __name__ = "LaplacianMultilicative"

    def __init__(self, l_nets_to_integrate, exponent):
        LaplacianMethods.__init__(self, exponent)
        MultiplicativeMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


class LaplacianAdditive(AdditiveMethods, LaplacianMethods):
    __name__ = "LaplacianAdditive"

    def __init__(self, l_nets_to_integrate, exponent):
        LaplacianMethods.__init__(self, exponent)
        AdditiveMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


# ########################################################################################################################
# class OptimizeIntegrator:
#     """
#     TODO: optimization with constraints in gamma. Now there is no sum=1; so normalization should be performed afterwards
#     """
#
#     def __init__(self, evaluator):
#         self.evaluator = evaluator
#
#     def randomly(self, integrator, max_evals):
#
#         def get_gamma():
#             gamma = []
#             probability_rest = 1
#             for _ in integrator.network_names[:-1]:  # do except the last one
#                 gamma.append(np.random.uniform(0, 1)*probability_rest)
#                 probability_rest = probability_rest*(1-gamma[-1])
#             gamma.append(probability_rest)  # the remaining probability goes to te last
#
#             return np.array(gamma)
#
#         results_cols = integrator.network_names+[self.evaluator.metric_name]
#         tpe_results = pd.DataFrame([], columns=results_cols)
#         for i in range(max_evals):
#             print("\r{}%".format((100*i)//max_evals), end="\r")
#             gamma = get_gamma()
#             temp_result = pd.Series(np.nan, index=results_cols)
#             temp_result[integrator.network_names] = gamma
#             temp_result[self.evaluator.metric_name] = self.evaluator.evaluate(integrator.integrate(gamma))
#
#             tpe_results = tpe_results.append(temp_result, ignore_index=True)
#
#         print("Finished")
#         best = tpe_results.loc[tpe_results[self.evaluator.metric_name].idxmax(), :]
#         return tpe_results, best
#
#     def optimize(self, integrator, max_evals, maximize=True):
#         """
#
#         :param integrator:
#         :param max_evals:
#         :param gamma_bounds:
#         :param maximize:
#         :return:
#         """
#         if maximize:
#             sign = -1
#         else:
#             sign = 1
#
#         # def get_gamma_from_values(values):
#         #     gamma = []
#         #     probability_rest = 1
#         #     for val in values[:-1]:  # do except the last one
#         #         gamma.append(val*probability_rest)
#         #         probability_rest = probability_rest*(1-val)
#         #     gamma.append(probability_rest)  # the remaining probability goes to te last
#         #
#         #     return np.array(gamma)
#         def get_gamma_from_values(values):
#             gamma = np.array(values)
#             gamma = gamma/gamma.sum()
#             return gamma
#
#
#         trials = pickle.load(open("myfile.p", "rb"))
#         pickle.dump(tpe_trials, open(".p", "wb"))
#
#         def optim_func(space):
#             gamma = get_gamma_from_values(list(space.values()))
#             return self.evaluator.evaluate(integrator.integrate(gamma))
#
#         space = {network_name: hp.uniform(network_name, 0, 1) for network_name in integrator.network_names}
#
#         # ojo que hay que maximizar entonces va el -
#         tpe_trials = Trials()
#         best = fmin(fn=lambda sp: sign*optim_func(sp),
#                     space=space,
#                     trials=tpe_trials,
#                     algo=tpe.suggest,
#                     max_evals=max_evals)
#
#         tpe_results = {network_name: gamma_values for network_name, gamma_values in tpe_trials.idxs_vals[1].items()}
#         tpe_results[self.evaluator.metric_name] = [sign*x['loss'] for x in tpe_trials.results]
#         tpe_results = pd.DataFrame(tpe_results)
#         # Warning, normalization so it sums up to 1.
#         for ix, row in tpe_results.iterrows():
#             tpe_results.loc[ix, list(space.keys())] = get_gamma_from_values(row[list(space.keys())])
#         # tpe_results[integrator.network_names] = tpe_results[integrator.network_names].div(tpe_results[integrator.network_names].sum(axis=1), axis=0)
#
#         best = pd.Series(best)
#         best[self.evaluator.metric_name] = tpe_results[self.evaluator.metric_name].max()
#         best[list(space.keys())] = get_gamma_from_values(best[list(space.keys())])
#         return tpe_results, best


########################################################################################################################
if __name__=="__main__":
    import numpy as np
    from evaluators import AUROClinkage

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
    x = np.roll(np.eye(N), 1, axis=0)
    network_1 = algorithms.Network(x + x.T)
    network_2 = algorithms.Network(x)
    x = np.random.uniform(size=(N, N))
    network_3 = algorithms.Network(1*((x + x.T) > 1))

    d_networks = {"Net1": network_1, "Net2": network_2}#, "Net3": network_3}
    print(d_networks)

    # ---------------------------
    integrator = SimpleAdditive(d_networks)
    w = integrator.integrate([0.5, 0.5])
    print(w)
    w = integrator.integrate([0.4, 0.6])
    print(w)

    integrator = LaplacianAdditive(d_networks, -0.5)
    w = integrator.integrate([0.5, 0.5])
    print(w)
    w = integrator.integrate([0.4, 0.6])
    print(w)


