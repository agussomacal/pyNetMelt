"""
Bayesian optimization
https://github.com/fmfn/BayesianOptimization
"""

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np

import src.algorithms as algorithms
import src.networks as networks


class Evaluator:
    def __init__(self, evaluator_function):
        self.evaluator_function = evaluator_function

    def evaluate(self, network):
        return self.evaluator_function(network)


class SeedTargetRanking(Evaluator):
    """
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
        assert score_matrix.shape == seeds_matrix.shape

        ranking = []
        for i, targets in enumerate(l_targets_ix):
            # rankings between 1 and the number of targets used.
            final_rank = np.ones(len(targets))*len(targets)
            # nodes that are in the network are ranked first.
            ix = []
            targets_with_ix = []
            for j, target in enumerate(targets):
                if target is not None:
                    ix.append(j)
                    targets_with_ix.append(target)
            final_rank[ix] = (len(targets_with_ix) - score_matrix[targets_with_ix, i].argsort().argsort()).tolist()
            ranking.append(final_rank)

        return ranking

    @staticmethod
    def find_index_from_names(networks_node_names, specific_node_names):
        assert isinstance(specific_node_names, (list, tuple, np.ndarray)), "should be a list or np.array of nodes"
        indexes = []
        for gen in specific_node_names:
            lix = list(np.where(gen == np.array(networks_node_names))[0])
            if len(lix) >= 1:
                assert len(lix) == 1, "There is a duplicated gen in the network"
                indexes += lix
            else:
                indexes += [None]  # is a nan because there is no index in the network for that query node.
        return indexes   # return indexes of specific nodes in the set of all nodes.

    @staticmethod
    def find_index_weight_from_names(networks_node_names, specific_node_names):
        assert isinstance(specific_node_names, dict), "should be a dictionary of nodes: weight"
        indexes2weight = dict()
        for gen in specific_node_names.keys():
            lix = list(np.where(gen == np.array(networks_node_names))[0])
            assert len(lix) == 1, "There is a duplicated gen in the network"
            for ix in lix:
                indexes2weight[ix] = specific_node_names[gen]
        return indexes2weight  # return indexes of specific nodes in the set of all nodes.

    @staticmethod
    def create_start_vector(networks_node_names, l_seeds_ix):
        y0 = np.zeros((len(networks_node_names), len(l_seeds_ix)))
        for i, seeds_dict in enumerate(l_seeds_ix):
            y0[list(seeds_dict.keys()), i] = list(seeds_dict.values())
        return y0

    @staticmethod
    def get_seed_matrix(networks_node_names, l_seeds_dict):
        seeds_global_ix = [SeedTargetRanking.find_index_weight_from_names(networks_node_names, seeds_dict)\
                           for seeds_dict in l_seeds_dict]
        return SeedTargetRanking.create_start_vector(networks_node_names, seeds_global_ix)

    @staticmethod
    def get_target_indexes(networks_node_names, target_nodes):
        l_targets_ix = []
        for targets in target_nodes:
            indexes = SeedTargetRanking.find_index_from_names(networks_node_names, targets)
            l_targets_ix.append(indexes)

        return l_targets_ix

    def __init__(self, ranking_to_value_function, node_names, l_seeds_dict, l_targets, l_true_targets, alpha,
                 tol=1e-08, max_iter=100, laplacian_exponent=-0.5, k_fold=10):
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

        self.k_fold = k_fold
        # skf = StratifiedKFold(n_splits=self.k_fold)

        # true_targets is a mask.
        y_true = []
        for targets, true_targets in zip(l_targets, l_true_targets):
            y_true.append([1 if target in true_targets else 0 for target in targets])
        y_true = np.array(y_true)

        seeds_matrix = SeedTargetRanking.get_seed_matrix(self.node_names, l_seeds_dict)
        l_targets_ix = SeedTargetRanking.get_target_indexes(self.node_names, l_targets)

        def evaluator_function(network):
            if type(network) == networks.Adjacency:
                laplacian = network.get_laplacian(self.laplacian_exponent)
            elif type(network) == networks.Laplacian:
                if network.laplacian_exponent != self.laplacian_exponent:
                    laplacian = network.get_adjacency(infering_technik="iterative").get_laplacian(
                        self.laplacian_exponent)
                else:
                    laplacian = network
            else:
                raise Exception("network is not either an adjacency nor a laplacian.")

            y_score = np.array(self.ranking_function(laplacian, seeds_matrix, l_targets_ix))
            # y_score = [[-ranking for ranking in target_rankings] for target_rankings in target_rankings_list]

            # values = np.repeat(np.nan, self.k_fold)
            indexes = np.arange(len(y_true))
            train_values = np.repeat(np.nan, self.k_fold)
            test_values = np.repeat(np.nan, self.k_fold)
            for i in range(self.k_fold):
                np.random.shuffle(indexes)
                splited_shuffled_data_ixes = np.array_split(indexes, self.k_fold)
                values = np.repeat(np.nan, self.k_fold)
                for j, ixes in enumerate(splited_shuffled_data_ixes):
                    values[j] = ranking_to_value_function(y_score[ixes], y_true[ixes])

                train_values[i] = np.mean(values)
                test_values[i] = train_values[i]
                # values[i] = ranking_to_value_function(y_score[test_index], y_true[test_index])

            return {"train": np.nanmean(train_values), "train std": np.nanstd(train_values),
                    "test": np.nanmean(test_values), "test std": np.nanstd(train_values)}

            # y_score = np.array(self.ranking_function(laplacian, seeds_matrix, l_targets_ix))
            # # y_score = [[-ranking for ranking in target_rankings] for target_rankings in target_rankings_list]
            #
            # # values = np.repeat(np.nan, self.k_fold)
            # train_values = np.repeat(np.nan, self.k_fold)
            # test_values = np.repeat(np.nan, self.k_fold)
            # for i,(train_index, test_index) in enumerate(skf.split(X=np.zeros(len(y_true)), y=np.zeros(len(y_true)))):
            #     train_values[i] = ranking_to_value_function(y_score[train_index], y_true[train_index])
            #     test_values[i] = ranking_to_value_function(y_score[test_index], y_true[test_index])
            #     # values[i] = ranking_to_value_function(y_score[test_index], y_true[test_index])
            #
            # return {"train": np.nanmean(train_values), "train std": np.nanstd(train_values),
            #         "test": np.nanmean(test_values), "test std": np.nanstd(train_values)}
            # # return {"train": np.nanmean(values[1:]), "train std": np.nanstd(values[1:]),
            # #         "test": np.nanmean(values[0]), "test std": np.nanstd(values[1:])}

        Evaluator.__init__(self, evaluator_function)


class AUROClinkage(SeedTargetRanking):
    def ranking_to_value(self, y_score, y_true):
        """
        :param target_rankings:
        :param true_targets: true_targets is a mask, should be the same shape as target_rankings
        :return:
        """
        # warning! -terget_rankings because roc_auc goes from minus to plus
        fpr, tpr, thresholds = metrics.roc_curve(y_true=np.concatenate(y_true),
                                                 y_score=-np.concatenate(y_score))

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

    def __init__(self, node_names, l_seeds_dict, l_targets, l_true_targets, alpha,
                 tol=1e-08, max_iter=100, laplacian_exponent=-0.5, max_fpr=1, auroc_normalized=True, k_fold=10):

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

        SeedTargetRanking.__init__(self,
                                   ranking_to_value_function=self.ranking_to_value,
                                   node_names=node_names,
                                   l_seeds_dict=l_seeds_dict,
                                   l_targets=l_targets,
                                   l_true_targets=l_true_targets,
                                   alpha=alpha,
                                   tol=tol,
                                   max_iter=max_iter,
                                   laplacian_exponent=laplacian_exponent,
                                   k_fold=k_fold)


if __name__ == "__main__":
    np.random.seed(1)

    laplacian_exponent = -0.5  # exponent of laplacian method
    alpha = 0.2  # alpha of the propagation

    max_iter = 100
    lno = 1
    n_targets = lno + 1

    p1 = 0.8
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
    l_seeds_dict = [{seed: 1} for seed in range(N)]#np.eye(N)
    l_targets = [np.roll(np.arange(N), -n)[1:(n_targets + 1)] for n in range(N)]

    p = np.repeat((1 - p1) / (n_targets - 1), n_targets - 1)
    p = np.insert(p, 0, p1)
    print(p)
    l_true_targets = [np.random.choice(targets, size=lno, p=p).tolist() for targets in l_targets]

    print("Target list:")
    print(l_targets)
    print("True targets")
    print(l_true_targets)

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

    print("AUC train: {:.5f} +- {:.5f} \nAUC test: {:.5f} +- {:.5f}".format(*auc.values()))
    print("AUC should be: p1*(1-p1)/8 = ", p1*(1-p1)/8, "if using auroc unnormalized")


