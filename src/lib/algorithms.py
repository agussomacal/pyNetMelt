import numpy as np


def normalize_by_strength(matrix, exponent, strength, exponent_sign=None):
    """
    if exponent is positive it will convert from a laplacian to the original adjacency matrix
        D^exponent * matrix * D^(1-exponent)
    Otherwise
        D^exponent * matrix * D^sign(exponent)*(1-abs(exponent))

    TODO: make it work only calculating the non zero strength elements.
    :param adj: 
    :param exponent: 
    :param strength:
    :param exponent_sign: This makes it possible to know when going from laplacian to adj or viceversa
     if exponent is 0 because there is no sign defined and wont understand how to comeback otherwise.
    :return: 
    """

    if exponent_sign is None:
        exponent_sign = np.sign(exponent)

    not_zero_strength_ix = (np.where(strength != 0)[0])
    # s1 = np.power(strength[not_zero_strength_ix], exponent)
    # s2 = np.power(strength[not_zero_strength_ix], np.sign(exponent)*(1 - np.abs(exponent)))
    # matrix[not_zero_strength_ix, :][:, not_zero_strength_ix] = s1 * np.transpose(
    #     s2 * matrix[not_zero_strength_ix, :][:, not_zero_strength_ix])

    s1 = np.zeros(len(strength))
    s2 = np.zeros(len(strength))
    s1[not_zero_strength_ix] = np.power(strength[not_zero_strength_ix], exponent)
    s2[not_zero_strength_ix] = np.power(strength[not_zero_strength_ix], exponent_sign*(1 - np.abs(exponent)))

    return np.transpose(s1 * np.transpose(s2 * matrix))


########################################################################################################################
class Network:
    def __init__(self, matrix=np.array([]), node_names=None):
        """

        :param adj_matrix: adjacency matrix of the graph. ndarray

        TODO: laplacian with D-A
        TODO: make laplacians conservative
        """
        assert type(matrix) == np.ndarray
        assert matrix.shape[0] == matrix.shape[1]
        self.matrix = matrix

        self.strength = None

        if node_names is None:
            self.node_names = np.arange(self.matrix.shape[0])
        else:
            assert len(node_names) == self.matrix.shape[0]
            self.node_names = node_names

        self.kind = "adjacency"
        self.exponent = None

        """
        :param kind:
            "adjacency" if it is the adjacency matrix of the graph
            "lambda-laplacian" if it is the laplacian of the graph by multiplying strength matrix to some exponent
            "laplacian" if it is the laplacian: D-A
        :param exponent:
            laplacian exponent:
                0 for the random walk laplacian.
                0.5 for the symmetric laplacian.
                1 for the heat laplacian.
        """

    def __add__(self, other):
        if type(other) == type(self):
            returned_matrix = self.matrix + other.matrix
        else:
            returned_matrix = self.matrix + other
        return Network(returned_matrix)

    def __mul__(self, other):
        if type(other) == type(self):
            returned_matrix = self.matrix * other.matrix
        else:
            returned_matrix = self.matrix * other
        return Network(returned_matrix)

    def __rmul__(self, other):
        # it is conmutative
        return self.__mul__(other)

    def __str__(self):
        return "mode: {} \n value: \n {}".format(self.kind, self.matrix)

    def get_strength(self):
        # if it has not yet calculated the strength, it does now.
        if self.strength is None:
            self.strength = np.array(np.squeeze(self.matrix.sum(axis=1)),
                                     dtype=float)  # degree or strength (in weighted networks) of the network
        return self.strength

    def set_strength(self, strength):
        """
        useful when using laplacian method of integration.
        :param strength:
        :return:
        """
        self.strength = strength

    def to_laplacian(self, exponent=0.5):
        transition_exponent = exponent
        if "laplacian" in self.kind:
            if self.exponent is not None and exponent != self.exponent:
                transition_exponent = exponent-self.exponent
            else:
                print("Network already in laplacian form. \n Aborting conversion.")
                return None

        self.matrix = normalize_by_strength(self.matrix, transition_exponent, self.get_strength(), exponent_sign=-1)
        self.exponent = exponent
        self.kind = "lambda-laplacian"

    def to_adjacency(self):
        if self.kind == "adjacency":
            print("Network already in adjacency form. \n Aborting conversion.")
        else:
            self.matrix = normalize_by_strength(self.matrix, -self.exponent, self.get_strength(), exponent_sign=1)
            self.exponent = None
            self.kind = "adjacency"

    def number_of_nodes(self):
        return len(self.node_names)

    def set_nodes(self, new_set_of_nodes):
        """
        TODO: warning, not memory efficient. creating zeros matrix unnecessarily.
        """

        common_nodes_ix = [i for i, node in enumerate(self.node_names) if node in new_set_of_nodes]
        number_of_common_nodes = len(common_nodes_ix)
        temp_matrix = np.zeros((len(new_set_of_nodes), len(new_set_of_nodes)))
        temp_matrix[:number_of_common_nodes,:][:,:number_of_common_nodes] = \
            self.matrix[:, common_nodes_ix][common_nodes_ix, :]
        self.matrix = temp_matrix
        self.node_names = list(np.append(np.array(self.node_names)[common_nodes_ix],
                                         np.array(list(set(new_set_of_nodes).difference(self.node_names)))))


########################################################################################################################
class Propagator:

    @staticmethod
    def label_propagator(network, seeds_matrix, alpha, tol=1e-08, max_iter=100, exponent=-0.5):
        """
        TODO: make the np.allclose comparison over each column. If one set has already converged, actualize only the rest

        :param network: network where propagation takes place.
        :param seeds_matrix: numpy matrix where each column is a train/test set of size equal to the number of network
        nodes and is full of zeros except for the row indexes associated with the seed of that train/test set. There
        should be a 1 or a -1 or a score of the importance of that seed to propagate.
        The algorithm performs many multiplications of that column to get the flux stread. But if it is a matrix it will
        gain speed using matrix multilication capabilities of numpy to fast the many sets to try.
        :param alpha: importance of spreading over memory of the initial seeds.
        :param tol: tolerance to find the stabel solution.
        :param max_iter: maximum number of iterations.
        :return: matrix of flux scores in probability format. each column is normalize to sum 1.
        """

        network.to_laplacian(exponent=exponent)
        y = seeds_matrix  # initial vector of field

        # ------propagacion-----------
        for i in range(max_iter):
            y_new = alpha * np.matmul(network.matrix, y) + (1 - alpha) * seeds_matrix  # propagation of flux + restart
            if np.allclose(y, y_new, atol=tol):
                break
            y = y_new

        # return probabilities, that's why the normalization over columns sum.
        return y/np.sum(y, axis=0)


if __name__ == "__main__":

    exponent = -0.5
    matrix = np.array([[1,2],[1,2]])
    strength = matrix.sum(axis=0)
    print(strength.shape)
    lap = normalize_by_strength(matrix, exponent, strength)
    print(lap)
    print(np.sign(0-1e-10))

    n = Network(np.array([[0, 1, 1, 0],
                          [1, 0, 1, 0],
                          [1, 1, 0, 1],
                          [0, 0, 1, 0]]))
    seed_matrix = np.eye(4)

    score_matrix = Propagator.label_propagator(n, seed_matrix, alpha=0.8, max_iter=1, exponent=exponent)
    print(score_matrix)

    # -----------------------------------
    print("Test time")
    N = 1000
    x = np.random.uniform(size=(N, N))
    n = Network((x+x.T) >= 1)
    seed_matrix = np.eye(N)

    from time import time

    t0 = time()
    score_matrix = Propagator.label_propagator(n, seed_matrix, alpha=0.8, max_iter=1, exponent=exponent)
    print(time() - t0)






