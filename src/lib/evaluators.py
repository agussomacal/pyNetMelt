"""
Bayesian optimization
https://github.com/fmfn/BayesianOptimization
"""

from sklearn import metrics
import numpy as np

import algorithms

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
    TODO: define clusters of seeds and use only one them to prioritize when looking for disease gene.
    """

    def ranking_function(self, network, seeds_matrix, targets_list):
        """

        :param network: network to use
        :param seeds_matrix: numpy matrix where columns are real numbers denoting the degree of seed for each test/train
        set. 0 no seed, > 0 some degree of seed. -1 anti-seed?? If there are two classes maybe.
        :param targets_list: list of tests; each tests is another list with the names of the target nodes.
        :return: ranking list of target nodes
        """
        score_matrix = algorithms.Propagator.label_propagator(network, seeds_matrix, self.alpha, tol=self.tol,
                                                              max_iter=self.max_iter, exponent=self.laplacian_exponent)
        score_matrix = score_matrix * (seeds_matrix == 0)  # drop from ranking all seeds

        # ranking = []
        # for i, targets in enumerate(targets_list):
        #     ranking.append((score_matrix.shape[0] - score_matrix[targets, i].argsort().argsort()).tolist())

        score_matrix = np.array([score_matrix[targets, i] for i, targets in enumerate(np.array(targets_list))]).T # target scores
        ranking = score_matrix.shape[0]-score_matrix.argsort(axis=0).argsort(axis=0)
        return ranking

    def __init__(self, ranking_to_value_function, seeds_matrix, targets_list, true_targets_list, alpha, tol=1e-08, max_iter=100, exponent=-0.5):
        """

        :param ranking_to_value_function:
        :param seeds_matrix:
        :param targets_list: list of tests; each tests is another list with the names of the target nodes.
        :param true_targets_list: list of tests; each tests is another list with the names of the true target nodes.
        :param alpha:
        :param tol:
        :param max_iter:
        :param exponent:
        """
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.laplacian_exponent = exponent

        # true_targets is a mask, should be the same shape
        self.y_true = []
        for targets, true_targets in zip(targets_list, true_targets_list):
            self.y_true += [1 if target in true_targets else 0 for target in targets]

        def evaluator_function(network):
            return ranking_to_value_function(self.ranking_function(network, seeds_matrix, targets_list))

        Evaluator.__init__(self, evaluator_function)


class AUROClinkage(SeedTargetRanking):
    def ranking_to_value(self, target_rankings_list):
        """
        TODO: warning! roc_auc_score is not clear what does if max_fpr not 1
        :param target_rankings:
        :param true_targets: true_targets is a mask, should be the same shape as target_rankings
        :return:
        """

        # y_score = [-ranking for target_rankings in target_rankings_list for ranking in target_rankings]
        # fpr, tpr, thresholds = metrics.roc_curve(y_true=self.y_true,
        #                                          y_score=y_score)

        fpr, tpr, thresholds = metrics.roc_curve(y_true=self.y_true,
                                                 y_score=-target_rankings_list.T.ravel())




        # warning! -terget_rankings because roc_auc goes from minus to plus
        # fpr, tpr, thresholds = metrics.roc_curve(y_true=[element for tt in true_targets for element in tt],
        #                                          y_score=[-element for tr in target_rankings for element in tr])
        auc_unnormalized = np.trapz(x=fpr[fpr <= self.max_fpr], y=tpr[fpr <= self.max_fpr])
        max_auc = self.max_fpr
        min_auc = max_auc ** 2 / 2
        # auc ajustado: https://www.rdocumentation.org/packages/pROC/versions/1.12.1/topics/auc
        return (1 + (auc_unnormalized - min_auc) / (max_auc - min_auc)) / 2
        # return metrics.roc_auc_score(y_true=true_targets.ravel(), y_score=-target_rankings.ravel(), max_fpr=self.max_fpr)

    def __init__(self, seeds_matrix, targets_list, true_targets_list, alpha, tol=1e-08, max_iter=100,
                 laplacian_exponent=-0.5, max_fpr=1):
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

        def ranking_to_value_function(target_rankings_list):
            return self.ranking_to_value(target_rankings_list)

        SeedTargetRanking.__init__(self, ranking_to_value_function, seeds_matrix, targets_list, true_targets_list, alpha, tol, max_iter,
                                   laplacian_exponent)


if __name__ == "__main__":
    np.random.seed(1)
    n_targets = 10
    N = 50  # number of nodes

    laplacian_exponent = -0.5  # exponent of laplacian method
    alpha = 0.2  # alpha of the propagation

    max_fpr = 1
    max_iter = 100

    # --------------------------
    # x = np.random.uniform(size=(N, N))
    # network = algorithms.Network(1*((x + x.T) >= 1))
    x = np.roll(np.eye(N), 1, axis=0)
    network = algorithms.Network(x + x.T)
    network = algorithms.Network(x)
    print(network)

    # --------------------------
    seeds_matrix = np.eye(N)
    targets_list = [np.roll(np.arange(N), -n)[1:(n_targets + 1)] for n in range(N)]

    p1 = 0.8
    p = np.repeat((1 - p1) / (n_targets - 1), n_targets - 1)
    p = np.insert(p, 0, p1)
    print(p)
    true_targets = [[int(np.random.choice(targets, size=1, p=p))] for targets in targets_list]

    print("Target list:")
    print(targets_list)
    print("True targets")
    print(true_targets)

    # --------------------------
    evalauc = AUROClinkage(seeds_matrix,
                                      targets_list,
                                      true_targets,
                                      alpha=alpha,
                                      tol=1e-08,
                                      max_iter=max_iter,
                                      max_fpr=max_fpr,
                                      laplacian_exponent=laplacian_exponent)
    auc = evalauc.evaluate(network)

    print("AUC: {:.5f}".format(auc))

