import numpy as np


######################################################
def DAD(matrix, diagonal1, exponent1=None, diagonal2=None, exponent2=None):
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
    if exponent1 is None:
        exponent1 = 1
    if exponent2 is None:
        exponent2 = np.sign(exponent1) * (1-np.abs(exponent1))
    if diagonal2 is None:
        diagonal2 = diagonal1

    not_zero_diagonal1_ix = np.where(diagonal1 != 0)[0]
    not_zero_diagonal2_ix = np.where(diagonal2 != 0)[0]

    matrix_non_zero_ixes = np.ix_(not_zero_diagonal1_ix, not_zero_diagonal2_ix)

    d1 = np.power(diagonal1[not_zero_diagonal1_ix], exponent1)
    d2 = np.power(diagonal2[not_zero_diagonal2_ix], exponent2)

    return_matrix = np.zeros(matrix.shape)
    return_matrix[matrix_non_zero_ixes] = np.transpose(d1 * np.transpose(d2 * matrix[matrix_non_zero_ixes]))

    return return_matrix


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

    matrix = np.array([[1, 2, 0], [2, 0, 0], [0, 0, 0]])
    diagonal1 = np.array([2, 3, 0])
    assert (DAD(matrix, diagonal1=diagonal1, exponent1=1, exponent2=1) == np.array([[ 4, 12,  0],
                                                                                   [12,  0,  0],
                                                                                   [ 0,  0,  0]])).all()
    diagonal2 = np.array([2, 3, 0])
    diagonal1 = np.array([2, 3])
    matrix = np.array([[1, 2, 0], [2, 0, 0]])
    print(DAD(matrix, diagonal1=diagonal1, diagonal2=diagonal2, exponent1=1, exponent2=1))
    assert (DAD(matrix, diagonal1=diagonal1, diagonal2=diagonal2, exponent1=1, exponent2=1) == np.array([[4, 12, 0],
                                                                                    [12, 0, 0]])).all()

    exponent = -0.5

    matrix = np.array([[1, 2], [1, 2]])
    strength = matrix.sum(axis=0)
    print(strength.shape)
    lap = DAD(matrix, diagonal1=strength, exponent1=exponent)
    print(lap)
    print(np.sign(0-1e-10))

    # -------test propagator -------

    import networks
    n = networks.Adjacency(np.array([[0, 1, 1, 0],
                          [1, 0, 1, 0],
                          [1, 1, 0, 1],
                          [0, 0, 1, 0]]))
    seed_matrix = np.eye(4)

    score_matrix = Propagator.label_propagator(n.get_laplacian(exponent), seed_matrix, alpha=0.8, tol=1e-08, max_iter=100)
    print(score_matrix)

    # -----------------------------------
    print("Test time")
    N = 1000
    x = np.random.uniform(size=(N, N))
    n = networks.Adjacency((x+x.T) >= 1)
    seed_matrix = np.eye(N)

    from time import time

    t0 = time()
    score_matrix = Propagator.label_propagator(n.get_laplacian(exponent), seed_matrix, alpha=0.8, tol=1e-08,
                                               max_iter=100)
    print(time() - t0)






