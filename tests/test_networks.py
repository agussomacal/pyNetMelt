from src.networks import *

import numpy as np
import unittest


class TestBipartite(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.edgelist_1 = pd.DataFrame([["A", "B", 0.2],
                                        ["A", "C", 0.5],
                                        ["A", "B", 0.2],
                                        ["D", "B", 0.8]],
                                       columns=["source", "target", "score"])
        self.edgelist_2 = pd.DataFrame([["A", "1", 1],
                                        ["A", "2", 1],
                                        ["B", "2", 1],
                                        ["B", "3", 1]],
                                       columns=["source", "target", "score"])

    def test_bipartite(self):
        bi_net = Bipartite(self.edgelist_1)
        assert set(bi_net.get_neighborhood("A", "source")) == {'B', 'C'}
        assert set(bi_net.get_nodes_ids("source")) == {'A', 'D'}
        assert set(bi_net.get_nodes_ids("target")) == {'B', 'C'}

    def test_bipartite_filter_by_score(self):
        bi_net = Bipartite(self.edgelist_1)
        bi_net.filter_edges_by_score(0.2)
        assert set(bi_net.get_neighborhood("A", "source")) == {'B', 'C'}
        assert set(bi_net.get_nodes_ids("source")) == {'A', 'D'}
        assert set(bi_net.get_nodes_ids("target")) == {'B', 'C'}

    def test_bipartite_filter_by_degree(self):
        bi_net = Bipartite(self.edgelist_1)
        bi_net.filter_nodes_by_degree("source", degree_lower_threshold=2)
        assert set(bi_net.get_neighborhood("A", "source")) == {'B', 'C'}
        assert set(bi_net.get_nodes_ids("source")) == {'A'}
        assert set(bi_net.get_nodes_ids("target")) == {'B', 'C'}

    def test_bipartite_filter_by_intersection(self):
        bi_net = Bipartite(self.edgelist_1)
        bi_net.filter_nodes_by_intersection("source", ["A", "B", "C"])
        assert set(bi_net.get_neighborhood("A", "source")) == {'B', 'C'}
        assert set(bi_net.get_nodes_ids("source")) == {'A'}
        assert set(bi_net.get_nodes_ids("target")) == {'B', 'C'}

    def test_bipartite_one_mode_proyection(self):
        bi_net = Bipartite(self.edgelist_2)
        assert (bi_net.get_proyection("source", "one_mode_proyection").values == np.array([[2, 1], [1, 2]])).all()

    def test_bipartite_laplacian_proyection(self):
        bi_net = Bipartite(self.edgelist_2)
        assert (bi_net.get_proyection("source", "laplacian_1", -1, False).values == np.array(
            [[3.0 / 4, 1 / 4], [1 / 4, 3 / 4]])).all()

    def test_bipartite_get_similar_nodes(self):
        bi_net = Bipartite(self.edgelist_2)
        proyection = bi_net.get_proyection("source", "laplacian_1", -1, False)
        assert bi_net.get_similar_nodes(proyection, 0.5) == {'A': {'A': 0.75}, 'B': {'B': 0.75}}


class TestAdjacency(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        # ------ Test Adjacency ------
        N = 5
        x1 = np.roll(np.eye(N), 1, axis=0)
        x2 = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.network_1 = Adjacency(x1 + x1.T)
        self.network_2 = Adjacency(x2)

    def test_get_laplacian(self):
        laplacian2 = self.network_2.get_laplacian(-0.5)
        assert np.allclose(laplacian2.matrix[0], np.array([0, 1 / 2, 1 / np.sqrt(6), 0]))
        assert np.allclose(laplacian2.matrix[1][2:], np.array([1 / np.sqrt(6), 0]))
        assert np.allclose(laplacian2.matrix[2][-1], np.array([1 / np.sqrt(3)]))

        laplacian2 = self.network_2.get_laplacian(-0.25)
        assert np.allclose(laplacian2.matrix[0], np.array([0, 1 / 2, 1 / (np.power(2, 1 / 4) * np.power(3, 3 / 4)), 0]))
        assert np.allclose(laplacian2.matrix[1][2:], np.array([1 / (np.power(2, 1 / 4) * np.power(3, 3 / 4)), 0]))
        assert np.allclose(laplacian2.matrix[2][-1], np.array([1 / np.power(3, 1 / 4)]))








