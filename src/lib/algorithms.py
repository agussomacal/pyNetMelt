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
"""

:param adj_matrix: adjacency matrix of the graph. ndarray

TODO: laplacian with D-A
TODO: make laplacians conservative

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


class NetworksModes:
    def __init__(self, matrix, node_names=None):
        assert type(matrix) == np.ndarray
        assert matrix.shape[0] == matrix.shape[1]
        self.matrix = matrix

        if node_names is None:
            self.node_names = np.arange(self.matrix.shape[0])
        else:
            assert len(node_names) == self.matrix.shape[0]
            self.node_names = node_names

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        # it is conmutative
        return self.__mul__(other)

    def __str__(self):
        return "Number of nodes: {}\nMatrix: \n {}".format(self.number_of_nodes(), self.matrix)

    def number_of_nodes(self):
        return len(self.node_names)

    def set_nodes(self, new_set_of_nodes):
        """
        TODO: warning, not memory efficient. creating zeros matrix unnecessarily.
        """
        # self.matrix = pd.DataFrame(self.matrix, columns=self.node_names, index=self.node_names)
        # self.matrix = self.matrix.loc[new_set_of_nodes, new_set_of_nodes].fillna(0).values

        common_nodes_ix = [i for i, node in enumerate(self.node_names) if node in new_set_of_nodes]
        number_of_common_nodes = len(common_nodes_ix)
        temp_matrix = np.zeros((len(new_set_of_nodes), len(new_set_of_nodes)))
        temp_matrix[:number_of_common_nodes,:][:,:number_of_common_nodes] = \
            self.matrix[:, common_nodes_ix][common_nodes_ix, :]
        self.matrix = temp_matrix
        self.node_names = list(np.append(np.array(self.node_names)[common_nodes_ix],
                                         np.array(list(set(new_set_of_nodes).difference(self.node_names)))))


class Adjacency(NetworksModes):
    def __init__(self, matrix, node_names=None):
        self.__name__ = "adjacency"
        NetworksModes.__init__(self, matrix, node_names)

        # degree or strength (in weighted networks) of the network
        self.strength = np.array(np.squeeze(self.matrix.sum(axis=1)), dtype=float)

    def __add__(self, other):
        if type(other) == type(self):
            assert np.all(self.node_names == other.node_names)
            returned_matrix = self.matrix + other.matrix
        else:
            returned_matrix = self.matrix + other
        return Adjacency(returned_matrix, node_names=self.node_names)

    def __mul__(self, other):
        if type(other) == type(self):
            assert np.all(self.node_names == other.node_names)
            returned_matrix = self.matrix * other.matrix
        else:
            returned_matrix = self.matrix * other
        return Adjacency(returned_matrix, node_names=self.node_names)

    def get_laplacian(self, laplacian_exponent=DEFAULT_LAPLACIAN_EXPONENT):
        return Laplacian(matrix=normalize_by_strength(self.matrix, self.strength, exponent1=laplacian_exponent),
                         strength=self.strength,
                         laplacian_exponent=laplacian_exponent,
                         node_names=self.node_names)


class Laplacian(NetworksModes):
    def __init__(self, matrix, strength, laplacian_exponent, node_names=None):
        self.__name__ = "laplacian_{}".format(laplacian_exponent)
        NetworksModes.__init__(self, matrix, node_names)

        # degree or strength (in weighted networks) of the network
        self.strength = strength
        self.laplacian_exponent = laplacian_exponent

    def __add__(self, other):
        if type(other) == type(self):
            assert np.all(self.node_names == other.node_names)  # both should in the same order and have the same names.
            assert self.laplacian_exponent == other.laplacian_exponent  # both should have the same laplacian exponent.
            returned_matrix = self.matrix + other.matrix
            strength = None  # we can't tell what will be the strength of the sum of two laplacians... a speciall method
            # is needed.
        else:
            returned_matrix = self.matrix + other
            strength = self.strength
        return Laplacian(returned_matrix,
                         strength=strength,
                         laplacian_exponent=self.laplacian_exponent,
                         node_names=self.node_names)

    def __mul__(self, other):
        if type(other) == type(self):
            assert np.all(self.node_names == other.node_names)  # both should in the same order and have the same names.
            assert self.laplacian_exponent == other.laplacian_exponent  # both should have the same laplacian exponent.
            returned_matrix = self.matrix * other.matrix
            strength = None  # we can't tell what will be the strength of the sum of two laplacians... a speciall method
            # is needed.
        else:
            returned_matrix = self.matrix * other
            strength = self.strength
        return Laplacian(returned_matrix,
                         strength=strength,
                         laplacian_exponent=self.laplacian_exponent,
                         node_names=self.node_names)

    def infer_strength(self, infering_technik="ones", max_iter=100):
        """

        :param infering_technik:
        :return:
        """
        strength = np.ones(self.number_of_nodes())
        if infering_technik is "iterative":
            for i in range(max_iter):
                new_strength = normalize_by_strength(matrix=self.matrix,
                                                     exponent1=-self.laplacian_exponent,
                                                     strength=strength).sum(axis=0)
                new_strength = new_strength / new_strength.sum() * strength.sum()
                if np.allclose(new_strength, strength):
                    break
                strength = new_strength

        return strength

    def get_adjacency(self, infering_technik="iterative", max_iter=100):
        if self.strength is None:
            self.strength = self.infer_strength(infering_technik, max_iter)
        return Adjacency(matrix=normalize_by_strength(matrix=self.matrix,
                                                      strength=self.strength,
                                                      exponent1=-self.laplacian_exponent),
                         node_names=self.node_names)


########################################################################################################################
class Propagator:

    @staticmethod
    def label_propagator(laplacian, seeds_matrix, alpha, tol=1e-08, max_iter=100):
        """
        TODO: make the np.allclose comparison over each column. If one set has already converged, actualize only the rest

        :param laplacian: laplacian where propagation takes place.
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

        y = seeds_matrix  # initial vector of field

        # ------propagacion-----------
        for i in range(max_iter):
            y_new = alpha * np.matmul(laplacian.matrix, y) + (1 - alpha) * seeds_matrix  # propagation of flux + restart
            if np.allclose(y, y_new, atol=tol):
                break
            y = y_new

        # return probabilities, that's why the normalization over columns sum.
        return y/np.sum(y, axis=0)


if __name__ == "__main__":
    N = 5
    x = np.roll(np.eye(N), 1, axis=0)
    network_1 = Adjacency(x + x.T)
    print(network_1)
    print(network_1.get_laplacian(-0.4))

    x=np.array([[0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0]])

    network_2 = Adjacency(x)
    print(network_2)

    laplacian2 = network_2.get_laplacian(-0.5)
    print(laplacian2)
    assert np.allclose(laplacian2.matrix[0], np.array([ 0, 1/2, 1/np.sqrt(6), 0]))
    assert np.allclose(laplacian2.matrix[1][2:], np.array([1/np.sqrt(6), 0]))
    assert np.allclose(laplacian2.matrix[2][-1], np.array([1/np.sqrt(3)]))

    laplacian2 = network_2.get_laplacian(-0.25)
    print(laplacian2)
    assert np.allclose(laplacian2.matrix[0], np.array([0, 1 / 2, 1 / (np.power(2, 1 / 4) * np.power(3, 3 / 4)), 0]))
    assert np.allclose(laplacian2.matrix[1][2:], np.array([1 / (np.power(2, 1 / 4) * np.power(3, 3 / 4)), 0]))
    assert np.allclose(laplacian2.matrix[2][-1], np.array([1 / np.power(3, 1 / 4)]))

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






