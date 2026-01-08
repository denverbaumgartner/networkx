"""Tests for SALSA algorithm."""

import pytest

import networkx as nx
from networkx.algorithms.link_analysis.salsa_alg import (
    _salsa_python,
    _salsa_scipy,
)

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


class TestSALSA:
    """Main test class for SALSA algorithm."""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        # Simple directed graph
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 1)])
        cls.G = G

        # Expected normalized values (hand-calculated)
        cls.G.expected_authorities = {
            1: 1.0 / 3.0,
            2: 1.0 / 6.0,
            3: 0.5,
        }
        cls.G.expected_hubs = {
            1: 0.5,
            2: 1.0 / 6.0,
            3: 1.0 / 3.0,
        }

        # Weighted graph
        cls.G_weighted = nx.DiGraph()
        cls.G_weighted.add_edge(1, 2, weight=2.0)
        cls.G_weighted.add_edge(1, 3, weight=1.0)
        cls.G_weighted.add_edge(2, 3, weight=1.0)

    @pytest.mark.parametrize("salsa_alg", (nx.salsa, _salsa_python, _salsa_scipy))
    def test_salsa_basic(self, salsa_alg):
        """Test basic SALSA computation."""
        G = self.G
        h, a = salsa_alg(G)

        for n in G:
            assert h[n] == pytest.approx(G.expected_hubs[n], abs=1e-4)
            assert a[n] == pytest.approx(G.expected_authorities[n], abs=1e-4)

    def test_empty(self):
        """Test SALSA on empty graph."""
        G = nx.DiGraph()
        assert nx.salsa(G) == ({}, {})
        assert _salsa_python(G) == ({}, {})
        assert _salsa_scipy(G) == ({}, {})

    @pytest.mark.parametrize("salsa_alg", (nx.salsa, _salsa_python, _salsa_scipy))
    def test_normalization(self, salsa_alg):
        """Test normalized scores sum to 1."""
        h, a = salsa_alg(self.G, normalized=True)
        assert sum(h.values()) == pytest.approx(1.0, abs=1e-10)
        assert sum(a.values()) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("salsa_alg", (nx.salsa, _salsa_python, _salsa_scipy))
    def test_unnormalized(self, salsa_alg):
        """Test unnormalized scores."""
        G = self.G
        h, a = salsa_alg(G, normalized=False)
        # Unnormalized expected values (before dividing by 3)
        expected_h = {1: 1.5, 2: 0.5, 3: 1.0}
        expected_a = {1: 1.0, 2: 0.5, 3: 1.5}
        for n in G:
            assert h[n] == pytest.approx(expected_h[n], abs=1e-10)
            assert a[n] == pytest.approx(expected_a[n], abs=1e-10)

    @pytest.mark.parametrize("salsa_alg", (nx.salsa, _salsa_python, _salsa_scipy))
    def test_weights(self, salsa_alg):
        """Test SALSA with edge weights."""
        h, a = salsa_alg(self.G_weighted, weight="weight")
        # Verify weighted results differ from unweighted (checking authorities)
        h_uw, a_uw = salsa_alg(self.G_weighted, weight=None)
        # Authority scores should differ because they depend on out-degree weights
        assert a != a_uw

    @pytest.mark.parametrize("salsa_alg", (nx.salsa, _salsa_python, _salsa_scipy))
    def test_weight_none(self, salsa_alg):
        """Test that weight=None ignores edge weights."""
        G = self.G_weighted
        h_weighted, a_weighted = salsa_alg(G, weight="weight")
        h_unweighted, a_unweighted = salsa_alg(G, weight=None)

        # Authority results should differ when weights are not uniform
        assert a_weighted != a_unweighted

    def test_implementation_consistency(self):
        """Test all implementations produce same results."""
        h_main, a_main = nx.salsa(self.G)
        h_scipy, a_scipy = _salsa_scipy(self.G)
        h_python, a_python = _salsa_python(self.G)

        for n in self.G:
            assert h_main[n] == pytest.approx(h_scipy[n], abs=1e-10)
            assert h_main[n] == pytest.approx(h_python[n], abs=1e-10)
            assert a_main[n] == pytest.approx(a_scipy[n], abs=1e-10)
            assert a_main[n] == pytest.approx(a_python[n], abs=1e-10)

    def test_return_types(self):
        """Test that return values are Python floats, not numpy types."""
        G = self.G
        h, a = nx.salsa(G)
        for node in G:
            assert type(h[node]) is float, f"Hub score for {node} is not float"
            assert type(a[node]) is float, f"Auth score for {node} is not float"

    def test_undirected_graph(self):
        """Test SALSA on undirected graph (treated as bidirectional)."""
        G = nx.Graph([(1, 2), (2, 3)])
        h, a = nx.salsa(G)
        # Should run without error
        assert isinstance(h, dict) and isinstance(a, dict)

    def test_self_loop(self):
        """Test SALSA handles self-loops correctly."""
        G = nx.DiGraph([(1, 1), (1, 2), (2, 1)])
        h, a = nx.salsa(G)
        # Should run without error
        assert isinstance(h, dict) and isinstance(a, dict)
        assert len(h) == 2 and len(a) == 2


class TestSALSAEdgeCases:
    """Edge case tests for SALSA algorithm."""

    def test_single_node_no_edges(self):
        """Test single node with no edges."""
        G = nx.DiGraph()
        G.add_node(1)
        h, a = nx.salsa(G)
        assert h == {1: 0.0}
        assert a == {1: 0.0}

    def test_isolated_nodes(self):
        """Test graph with isolated nodes."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2)])
        G.add_node(3)
        h, a = nx.salsa(G)
        assert h[3] == 0.0
        assert a[3] == 0.0

    def test_star_graph(self):
        """Test star graph structure."""
        G = nx.DiGraph()
        for i in range(1, 5):
            G.add_edge(0, i)
        h, a = nx.salsa(G)
        assert h[0] == 1.0
        for i in range(1, 5):
            assert h[i] == 0.0
            assert a[i] == pytest.approx(0.25, abs=1e-10)

    def test_cycle_graph(self):
        """Test cycle graph structure."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        h, a = nx.salsa(G)
        for n in G:
            assert h[n] == pytest.approx(1.0 / 3.0, abs=1e-4)
            assert a[n] == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_undirected_cycle(self):
        """Test undirected cycle graph."""
        G = nx.cycle_graph(4)
        h, a = nx.salsa(G)
        for n in G:
            assert h[n] == pytest.approx(0.25, abs=1e-4)
            assert a[n] == pytest.approx(0.25, abs=1e-4)

    def test_single_edge(self):
        """Test graph with single edge."""
        G = nx.DiGraph([(1, 2)])
        h, a = nx.salsa(G, normalized=False)
        # Node 1 has out_degree=1, in_degree=0
        # Node 2 has out_degree=0, in_degree=1
        # auth(1) = 0 (no incoming edges)
        # auth(2) = 1/1 = 1 (edge 1->2, out_deg(1)=1)
        # hub(1) = 1/1 = 1 (edge 1->2, in_deg(2)=1)
        # hub(2) = 0 (no outgoing edges)
        assert a[1] == 0.0
        assert a[2] == pytest.approx(1.0, abs=1e-10)
        assert h[1] == pytest.approx(1.0, abs=1e-10)
        assert h[2] == 0.0

    def test_complete_graph(self):
        """Test on complete directed graph."""
        G = nx.complete_graph(5, create_using=nx.DiGraph)
        h, a = nx.salsa(G)

        # All nodes should have equal scores
        for n in G:
            assert h[n] == pytest.approx(0.2, abs=1e-4)
            assert a[n] == pytest.approx(0.2, abs=1e-4)

    def test_path_graph(self):
        """Test on directed path graph."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        h, a = nx.salsa(G)

        # First node: hub only
        assert h[1] > 0
        assert a[1] == 0.0

        # Last node: authority only
        assert h[4] == 0.0
        assert a[4] > 0

    def test_bipartite_graph(self):
        """Test on bipartite directed graph."""
        G = nx.DiGraph()
        # Hubs: 1, 2
        # Authorities: 3, 4
        G.add_edges_from([(1, 3), (1, 4), (2, 3), (2, 4)])
        h, a = nx.salsa(G)

        # Hubs have zero authority
        assert a[1] == 0.0
        assert a[2] == 0.0

        # Authorities have zero hub
        assert h[3] == 0.0
        assert h[4] == 0.0

    def test_large_graph_performance(self):
        """Test SALSA completes in reasonable time on larger graph."""
        # This is a smoke test, not a strict performance benchmark
        G = nx.gnm_random_graph(1000, 5000, directed=True, seed=42)
        h, a = nx.salsa(G)  # Should complete without timeout

        assert len(h) == 1000
        assert len(a) == 1000
