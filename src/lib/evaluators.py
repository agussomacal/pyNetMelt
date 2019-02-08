"""
Bayesian optimization
https://github.com/fmfn/BayesianOptimization
"""

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np

import algorithms
import networks

"""
"""


class Evaluator:
    def __init__(self, evaluator_function):
        self.evaluator_function = evaluator_function

    def evaluate(self, network):
        return self.evaluator_function(network)


class SeedTargetRanking(Evaluator):
    """
    TODO: kfold
    # from sklearn.model_selection import StratifiedKFold
#         skf = StratifiedKFold(n_splits=self.kfold)
#         train_seed_set = []
#         test_seed_set = []
#         for i, (train_ix, test_ix) in enumerate(skf.split(seeds, groups)):
    TODO: define clusters of seeds and use only one of them to prioritize when looking for disease gene.
    """

    def ranking_function(self, laplacian, seeds_matrix, l_targets_ix):
        """

        :param laplacian: laplacian to use
        :param seeds_matrix: numpy matrix where columns are real numbers denoting the degree of seed for each test/train
        set. 0 no seed, > 0 some degree of seed. -1 anti-seed?? If there are two classes maybe.
        :param l_targets_ix: list of tests; each tests is another list with the names of the target nodes.
        :return: ranking list of target nodes
        """
        score_matrix = algorithms.Propagator.label_propagator(laplacian,
                                                              seeds_matrix,
                                                              self.alpha,
                                                              tol=self.tol,
                                                              max_iter=self.max_iter)
        score_matrix = score_matrix * (seeds_matrix == 0)  # drop from ranking all seeds

        ranking = []
        for i, targets in enumerate(l_targets_ix):
            ranking.append((score_matrix.shape[0] - score_matrix[targets, i].argsort().argsort()).tolist())

        # score_matrix = np.array([score_matrix[targets, i] for i, targets in enumerate(np.array(l_targets_ix))]).T # target scores
        # ranking = score_matrix.shape[0]-score_matrix.argsort(axis=0).argsort(axis=0)
        return ranking

    @staticmethod
    def find_index_from_names(networks_node_names, specific_node_names):
        indexes = []
        for gen in specific_node_names:
            lix = np.where(gen == np.array(networks_node_names))[0]
            indexes.append(lix[0])  # appends only the first that appears.
        return indexes  # return indexes of specific nodes in the set of all nodes.

    @staticmethod
    def create_start_vector(networks_node_names, l_seeds_ix, l_seeds_weight=None):
        if l_seeds_weight is None:
            l_seeds_weight = np.ones(len(l_seeds_ix))

        y0 = np.zeros((len(networks_node_names), len(l_seeds_ix)))
        for i, (seeds, weight) in enumerate(zip(l_seeds_ix, l_seeds_weight)):
            y0[seeds, i] = weight
        return y0

    @staticmethod
    def get_seed_matrix(networks_node_names, l_seeds, l_seeds_weight):
        seeds_global_ix = [SeedTargetRanking.find_index_from_names(networks_node_names, seeds) for seeds in l_seeds]
        return SeedTargetRanking.create_start_vector(networks_node_names, seeds_global_ix, l_seeds_weight)

    @staticmethod
    def get_target_indexes(networks_node_names, target_nodes):
        l_targets_ix = []
        for targets in target_nodes:
            l_targets_ix.append(SeedTargetRanking.find_index_from_names(networks_node_names, targets))
            # l_true_targets_ix.append(SeedTargetRanking.find_index_from_names(networks_node_names, true_targets))

        return l_targets_ix

    def __init__(self, ranking_to_value_function, node_names, l_seeds, l_targets, l_true_targets, alpha, l_seeds_weight=None,
                 tol=1e-08, max_iter=100, laplacian_exponent=-0.5):
        """

        :param ranking_to_value_function:
        :param seeds_matrix:
        :param targets_list: list of tests; each tests is another list with the names of the target nodes.
        :param true_targets_list: list of tests; each tests is another list with the names of the true target nodes.
        :param alpha:
        :param tol:
        :param max_iter:
        :param laplacian_exponent:
        """
        self.node_names = node_names

        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.laplacian_exponent = laplacian_exponent

        # true_targets is a mask.
        self.y_true = []
        for targets, true_targets in zip(l_targets, l_true_targets):
            self.y_true += [1 if target in true_targets else 0 for target in targets]

        seeds_matrix = SeedTargetRanking.get_seed_matrix(self.node_names, l_seeds, l_seeds_weight)
        l_targets_ix = SeedTargetRanking.get_target_indexes(self.node_names, l_targets)

        def evaluator_function(network):
            if type(network) == networks.Adjacency:
                laplacian = network.get_laplacian(self.laplacian_exponent)
            elif type(network) == networks.Laplacian:
                if network.laplacian_exponent != self.laplacian_exponent:
                    laplacian = network.get_adjacency(infering_technik="iterative").get_laplacian(self.laplacian_exponent)
                else:
                    laplacian = network
            else:
                raise Exception("network is not either an adjacency nor a laplacian.")
            return ranking_to_value_function(self.ranking_function(laplacian, seeds_matrix, l_targets_ix))

        Evaluator.__init__(self, evaluator_function)


class AUROClinkage(SeedTargetRanking):
    def ranking_to_value(self, target_rankings_list):
        """
        :param target_rankings:
        :param true_targets: true_targets is a mask, should be the same shape as target_rankings
        :return:
        """

        y_score = [-ranking for target_rankings in target_rankings_list for ranking in target_rankings]
        fpr, tpr, thresholds = metrics.roc_curve(y_true=self.y_true,
                                                 y_score=y_score)

        # fpr, tpr, thresholds = metrics.roc_curve(y_true=self.y_true,
        #                                          y_score=-target_rankings_list.T.ravel())




        # warning! -terget_rankings because roc_auc goes from minus to plus
        # fpr, tpr, thresholds = metrics.roc_curve(y_true=[element for tt in true_targets for element in tt],
        #                                          y_score=[-element for tr in target_rankings for element in tr])
        auc_unnormalized = np.trapz(x=fpr[fpr < self.max_fpr], y=tpr[fpr < self.max_fpr])
        ix_last_fp = np.sum(fpr < self.max_fpr)-1
        y_last = tpr[ix_last_fp]+(tpr[ix_last_fp+1]-tpr[ix_last_fp]) \
                            *(self.max_fpr-fpr[ix_last_fp])/(fpr[ix_last_fp+1]-fpr[ix_last_fp])
        auc_unnormalized += np.trapz(x=[fpr[ix_last_fp], self.max_fpr],
                                     y=[tpr[ix_last_fp], y_last])

        if not self.auroc_normalized:
            return auc_unnormalized
        else:
            max_auc = self.max_fpr
            min_auc = max_auc ** 2 / 2
            # auc ajustado: https://www.rdocumentation.org/packages/pROC/versions/1.12.1/topics/auc
            # return metrics.roc_auc_score(y_true=true_targets.ravel(), y_score=-target_rankings.ravel(), max_fpr=self.max_fpr)
            return (1 + (auc_unnormalized - min_auc) / (max_auc - min_auc)) / 2

    def __init__(self, node_names, l_seeds, l_targets, l_true_targets, alpha, l_seeds_weight=None,
                 tol=1e-08, max_iter=100, laplacian_exponent=-0.5, max_fpr=1, auroc_normalized=True):

        """

        :param seeds_matrix: matriz with weights on seeds and 0 otherwise to multiply and perform propagation.
        :param targets_list: which are the targets to look for and test.
        :param true_targets: true_targets is a mask, should be the same shape as targets_list
        :param alpha: propagation algorithm parameter.
        :param tol: tolerance to stop propagation when both vectors, t and t-1 are similar.
        :param max_iter: of label propagator.
        :param laplacian_exponent: exponent of the propagation laplacian.
        :param max_fpr: max false positive rate to see performance.
        """
        self.metric_name = "AUROC_{}".format(max_fpr)
        self.max_fpr = max_fpr
        self.auroc_normalized = auroc_normalized

        def ranking_to_value_function(target_rankings_list):
            return self.ranking_to_value(target_rankings_list)

        SeedTargetRanking.__init__(self, ranking_to_value_function,
                                   node_names=node_names,
                                   l_seeds=l_seeds,
                                   l_targets=l_targets,
                                   l_true_targets=l_true_targets,
                                   alpha=alpha,
                                   l_seeds_weight=l_seeds_weight,
                                   tol=tol,
                                   max_iter=max_iter,
                                   laplacian_exponent=laplacian_exponent)


if __name__ == "__main__":
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
    # x = np.random.uniform(size=(N, N))
    # network = networks.Adjacency(1*((x + x.T) >= 1))
    x = np.roll(np.eye(N), 1, axis=0)
    network = networks.Adjacency(x)
    print(network)

    # --------------------------
    l_seeds = [[seed] for seed in range(N)]#np.eye(N)
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
    evalauc = AUROClinkage(node_names=network.node_names,
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
    auc = evalauc.evaluate(network)

    print("AUC: {:.5f}".format(auc))
    print("AUC should be: p1*(1-p1)/8 = ", p1*(1-p1)/8, "if using auroc unnormalized")


