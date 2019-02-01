import numpy as np

######################################################
#           Constants
DEFAULT_LAPLACIAN_EXPONENT = -0.5


######################################################
def normalize_by_strength(matrix, strength, exponent1, exponent2=None):
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
    if exponent2 is None:
        exponent2 = np.sign(exponent1) * (1-np.abs(exponent1))

    not_zero_strength_ix = (np.where(strength != 0)[0])
    # s1 = np.power(strength[not_zero_strength_ix], exponent)
    # s2 = np.power(strength[not_zero_strength_ix], np.sign(exponent)*(1 - np.abs(exponent)))
    # matrix[not_zero_strength_ix, :][:, not_zero_strength_ix] = s1 * np.transpose(
    #     s2 * matrix[not_zero_strength_ix, :][:, not_zero_strength_ix])

    s1 = np.zeros(len(strength))
    s2 = np.zeros(len(strength))
    s1[not_zero_strength_ix] = np.power(strength[not_zero_strength_ix], exponent1)
    s2[not_zero_strength_ix] = np.power(strength[not_zero_strength_ix], exponent2)

    return np.transpose(s1 * np.transpose(s2 * matrix))


########################################################################################################################
class Network:
    def __init__(self, matrix=np.array([]), node_names=None, network_mode="adjacency",
                 laplacian_exponent=DEFAULT_LAPLACIAN_EXPONENT, strength=None):
        """

        :param adj_matrix: adjacency matrix of the graph. ndarray

        TODO: laplacian with D-A
        TODO: make laplacians conservative
        """
        assert type(matrix) == np.ndarray
        assert matrix.shape[0] == matrix.shape[1]
        self.matrix = matrix

        self.strength = strength

        if node_names is None:
            self.node_names = np.arange(self.matrix.shape[0])
        else:
            assert len(node_names) == self.matrix.shape[0]
            self.node_names = node_names

        self.network_mode = network_mode
        self.laplacian_exponent = laplacian_exponent

        """
        :param network_mode:
            "adjacency" if it is the adjacency matrix of the graph
            "lambda-laplacian" if it is the laplacian of the graph by multiplying strength matrix to some exponent
            "laplacian" if it is the laplacian: D-A
        :param laplacian_exponent:
            laplacian laplacian_exponent:
                0 for the random walk laplacian.
                0.5 for the symmetric laplacian.
                1 for the heat laplacian.
        """

    def __add__(self, other):
        if type(other) == type(self):
            assert self.network_mode == other.network_mode
            assert np.all(self.node_names == other.node_names)
            returned_matrix = self.matrix + other.matrix
        else:
            returned_matrix = self.matrix + other
        return Network(returned_matrix, node_names=self.node_names, network_mode=self.network_mode, laplacian_exponent=self.laplacian_exponent,
                       strength=self.strength)

    def __mul__(self, other):
        if type(other) == type(self):
            assert self.network_mode == other.network_mode
            assert np.all(self.node_names == other.node_names)
            returned_matrix = self.matrix * other.matrix
        else:
            returned_matrix = self.matrix * other
        return Network(returned_matrix, node_names=self.node_names, network_mode=self.network_mode, laplacian_exponent=self.laplacian_exponent,
                       strength=self.strength)

    def __rmul__(self, other):
        # it is conmutative
        return self.__mul__(other)

    def __str__(self):
        return "mode: {} \n value: \n {}".format(self.network_mode, self.matrix)

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

    def to_laplacian(self, laplacian_exponent=DEFAULT_LAPLACIAN_EXPONENT):

        if "laplacian" in self.network_mode:
            assert self.laplacian_exponent is not None
            if laplacian_exponent != self.laplacian_exponent:
                transition_exponent = laplacian_exponent-self.laplacian_exponent
                self.matrix = normalize_by_strength(self.matrix, self.get_strength(),
                                                    exponent1=transition_exponent,
                                                    exponent2=-transition_exponent)
        else:
            self.matrix = normalize_by_strength(self.matrix, self.get_strength(), laplacian_exponent)

        self.laplacian_exponent = laplacian_exponent
        self.network_mode = "lambda-laplacian"

    def to_adjacency(self):
        if self.network_mode != "adjacency":
            self.matrix = normalize_by_strength(self.matrix, self.get_strength(),
                                                exponent1=-self.laplacian_exponent)
            self.network_mode = "adjacency"

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
    def label_propagator(network, seeds_matrix, alpha, tol=1e-08, max_iter=100, exponent=DEFAULT_LAPLACIAN_EXPONENT):
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

        network.to_laplacian(laplacian_exponent=exponent)
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
    N = 5
    x = np.roll(np.eye(N), 1, axis=0)
    network_1 = Network(x + x.T, laplacian_exponent=-0.5)
    print(network_1)
    network_1.to_laplacian(-0.4)
    print(network_1)
    network_1.to_adjacency()

    x=np.array([[0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0]])

    network_2 = Network(x, laplacian_exponent=-0.4)
    print(network_2)

    network_2.to_laplacian(-0.5)
    print(network_2)
    assert np.allclose(network_2.matrix[0], np.array([ 0, 1/2, 1/np.sqrt(6), 0]))
    assert np.allclose(network_2.matrix[1][2:], np.array([1/np.sqrt(6), 0]))
    assert np.allclose(network_2.matrix[2][-1], np.array([1/np.sqrt(3)]))

    network_2.to_laplacian(-0.25)
    print(network_2)
    assert np.allclose(network_2.matrix[0], np.array([0, 1 / 2, 1 / (np.power(2, 1 / 4) * np.power(3, 3 / 4)), 0]))
    assert np.allclose(network_2.matrix[1][2:], np.array([1 / (np.power(2, 1 / 4) * np.power(3, 3 / 4)), 0]))
    assert np.allclose(network_2.matrix[2][-1], np.array([1 / np.power(3, 1 / 4)]))

    network_2.to_adjacency()
    print(network_2)
    ibv
    # n = Network(np.array([[0, 1, 1, 0],
    #                       [1, 0, 1, 0],
    #                       [1, 1, 0, 1],
    #                       [0, 0, 1, 0]]))
    #
    # m = Network(np.array([[1, 0, 1, 0],
    #                       [0, 0, 1, 1],
    #                       [1, 1, 0, 1],
    #                       [0, 1, 1, 0]]))
    #
    # # print(n.matrix)
    # # a = n*n
    # #
    # # print(a.get_strength())
    # # print(a)
    # # a.to_laplacian(1)
    # # print(a)
    # # a.to_adjacency()
    # # print(a)
    #
    # # -------------------
    # gamma = [0.5, 0.5]
    # exponent = 1
    #
    # n.to_laplacian(exponent)
    # print(n)
    #
    # add = SimpleAdditive([n, m])
    # w_add = add.integrate(gamma, return_laplacian=False)
    # add_lap = LaplacianAdditive([n, m], exponent=exponent)
    # w_addlap = add_lap.integrate(gamma, return_laplacian=False)
    #
    # mul = SimpleMultiplicative([n, m])
    # w_mul = mul.integrate(return_laplacian=False)
    # mul_lap = LaplacianMultiplicative([n, m], exponent=exponent)
    # w_mullap = mul_lap.integrate(return_laplacian=False)
    #
    # # print(w_add)
    # print(w_addlap)
    # n.to_laplacian(exponent)
    # print(n)
    # m.to_laplacian(exponent)
    # print(m)
    #
    # # print(w_mul)
    # # print(w_mullap)

    exponent = -0.5

    matrix = np.array([[1, 2], [1, 2]])
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






