import math

import pytest

import networkx as nx

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


class TestEigenvectorCentrality:
    def test_K5(self):
        """Eigenvector centrality: K5"""
        G = nx.complete_graph(5)
        b = nx.eigenvector_centrality(G)
        v = math.sqrt(1 / 5.0)
        b_answer = dict.fromkeys(G, v)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)
        nstart = {n: 1 for n in G}
        b = nx.eigenvector_centrality(G, nstart=nstart)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

        b = nx.eigenvector_centrality_numpy(G)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-3)

    def test_P3(self):
        """Eigenvector centrality: P3"""
        G = nx.path_graph(3)
        b_answer = {0: 0.5, 1: 0.7071, 2: 0.5}
        b = nx.eigenvector_centrality_numpy(G)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-4)
        b = nx.eigenvector_centrality(G)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-4)

    def test_P3_unweighted(self):
        """Eigenvector centrality: P3"""
        G = nx.path_graph(3)
        b_answer = {0: 0.5, 1: 0.7071, 2: 0.5}
        b = nx.eigenvector_centrality_numpy(G, weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-4)

    def test_maxiter(self):
        with pytest.raises(nx.PowerIterationFailedConvergence):
            G = nx.path_graph(3)
            nx.eigenvector_centrality(G, max_iter=0)


class TestEigenvectorCentralityDirected:
    @classmethod
    def setup_class(cls):
        G = nx.DiGraph()

        edges = [
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 2),
            (3, 5),
            (4, 2),
            (4, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 8),
            (7, 1),
            (7, 5),
            (7, 8),
            (8, 6),
            (8, 7),
        ]

        G.add_edges_from(edges, weight=2.0)
        cls.G = G.reverse()
        cls.G.evc = [
            0.25368793,
            0.19576478,
            0.32817092,
            0.40430835,
            0.48199885,
            0.15724483,
            0.51346196,
            0.32475403,
        ]

        H = nx.DiGraph()

        edges = [
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 2),
            (3, 5),
            (4, 2),
            (4, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 8),
            (7, 1),
            (7, 5),
            (7, 8),
            (8, 6),
            (8, 7),
        ]

        G.add_edges_from(edges)
        cls.H = G.reverse()
        cls.H.evc = [
            0.25368793,
            0.19576478,
            0.32817092,
            0.40430835,
            0.48199885,
            0.15724483,
            0.51346196,
            0.32475403,
        ]

    def test_eigenvector_centrality_weighted(self):
        G = self.G
        p = nx.eigenvector_centrality(G)
        for a, b in zip(list(p.values()), self.G.evc):
            assert a == pytest.approx(b, abs=1e-4)

    def test_eigenvector_centrality_weighted_numpy(self):
        G = self.G
        p = nx.eigenvector_centrality_numpy(G)
        for a, b in zip(list(p.values()), self.G.evc):
            assert a == pytest.approx(b, abs=1e-7)

    def test_eigenvector_centrality_unweighted(self):
        G = self.H
        p = nx.eigenvector_centrality(G)
        for a, b in zip(list(p.values()), self.G.evc):
            assert a == pytest.approx(b, abs=1e-4)

    def test_eigenvector_centrality_unweighted_numpy(self):
        G = self.H
        p = nx.eigenvector_centrality_numpy(G)
        for a, b in zip(list(p.values()), self.G.evc):
            assert a == pytest.approx(b, abs=1e-7)


class TestEigenvectorCentralityExceptions:
    def test_multigraph_empty(self):
        """Empty multigraph should raise NetworkXPointlessConcept"""
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.eigenvector_centrality(nx.MultiGraph())

    def test_multigraph_numpy_empty(self):
        """Empty multigraph should raise NetworkXPointlessConcept"""
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.eigenvector_centrality_numpy(nx.MultiGraph())

    def test_null(self):
        with pytest.raises(nx.NetworkXException):
            nx.eigenvector_centrality(nx.Graph())

    def test_null_numpy(self):
        with pytest.raises(nx.NetworkXException):
            nx.eigenvector_centrality_numpy(nx.Graph())

    @pytest.mark.parametrize(
        "G",
        [
            nx.empty_graph(3),
            nx.DiGraph([(0, 1), (1, 2)]),
        ],
    )
    def test_disconnected_numpy(self, G):
        msg = "does not give consistent results for disconnected"
        with pytest.raises(nx.AmbiguousSolution, match=msg):
            nx.eigenvector_centrality_numpy(G)

    def test_zero_nstart(self):
        G = nx.Graph([(1, 2), (1, 3), (2, 3)])
        with pytest.raises(
            nx.NetworkXException, match="initial vector cannot have all zero values"
        ):
            nx.eigenvector_centrality(G, nstart={v: 0 for v in G})


class TestEigenvectorCentralityMultigraph:
    def test_multigraph_basic(self):
        """Eigenvector centrality: basic multigraph"""
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)  # parallel edge
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        centrality = nx.eigenvector_centrality(G)
        assert len(centrality) == 3
        # Node 1 should have highest centrality (most edge "weight" via parallel edges)
        assert centrality[1] >= centrality[2]

    def test_multigraph_weighted(self):
        """Eigenvector centrality: weighted multigraph"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(0, 1, weight=3.0)  # total = 5.0
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 0, weight=1.0)

        centrality = nx.eigenvector_centrality(G, weight="weight")
        assert len(centrality) == 3

    def test_multigraph_weight_aggregation_sum(self):
        """Verify sum aggregation for parallel edges"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(0, 1, weight=3.0)  # total = 6.0
        G.add_edge(1, 2, weight=6.0)
        G.add_edge(2, 0, weight=6.0)

        # With sum aggregation, this should be like a simple triangle
        # with all edges having weight 6
        H = nx.Graph()
        H.add_edge(0, 1, weight=6.0)
        H.add_edge(1, 2, weight=6.0)
        H.add_edge(2, 0, weight=6.0)

        c_multi = nx.eigenvector_centrality(G, weight="weight", multigraph_weight=sum)
        c_simple = nx.eigenvector_centrality(H, weight="weight")

        for node in G.nodes():
            assert c_multi[node] == pytest.approx(c_simple[node], abs=1e-5)

    def test_multigraph_weight_aggregation_max(self):
        """Verify max aggregation for parallel edges"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(0, 1, weight=5.0)
        G.add_edge(0, 1, weight=3.0)  # max = 5.0
        G.add_edge(1, 2, weight=5.0)
        G.add_edge(2, 0, weight=5.0)

        H = nx.Graph()
        H.add_edge(0, 1, weight=5.0)
        H.add_edge(1, 2, weight=5.0)
        H.add_edge(2, 0, weight=5.0)

        c_multi = nx.eigenvector_centrality(G, weight="weight", multigraph_weight=max)
        c_simple = nx.eigenvector_centrality(H, weight="weight")

        for node in G.nodes():
            assert c_multi[node] == pytest.approx(c_simple[node], abs=1e-5)

    def test_multigraph_unweighted_counts_edges(self):
        """For unweighted multigraphs, parallel edges count as weight"""
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)
        G.add_edge(0, 1)  # 3 edges
        G.add_edge(1, 2)  # 1 edge
        G.add_edge(2, 0)  # 1 edge

        # Equivalent simple graph with edge counts as weights
        H = nx.Graph()
        H.add_edge(0, 1, weight=3)
        H.add_edge(1, 2, weight=1)
        H.add_edge(2, 0, weight=1)

        c_multi = nx.eigenvector_centrality(G)  # unweighted
        c_simple = nx.eigenvector_centrality(H, weight="weight")

        for node in G.nodes():
            assert c_multi[node] == pytest.approx(c_simple[node], abs=1e-5)

    def test_multidigraph(self):
        """Eigenvector centrality: MultiDiGraph"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        centrality = nx.eigenvector_centrality(G.reverse())
        assert len(centrality) == 3
