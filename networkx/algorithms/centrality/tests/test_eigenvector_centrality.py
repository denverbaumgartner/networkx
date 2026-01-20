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


class TestEigenvectorCentralityCollapsePlan:
    """Tests for the collapse_plan parameter."""

    def test_multidigraph_collapses_correctly(self):
        """Verify MultiDiGraph collapse: drops weights, direction, parallel edges"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, weight=10)  # These weights should be dropped
        G.add_edge(1, 0, weight=20)  # Opposite direction - becomes same undirected edge
        G.add_edge(0, 1, weight=5)   # Parallel edge - should collapse
        G.add_edge(1, 2, weight=1)

        # With unweighted_undirected collapse
        centrality = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")

        # Should be equivalent to simple path graph 0-1-2
        H = nx.Graph()
        H.add_edge(0, 1)  # No weight
        H.add_edge(1, 2)  # No weight
        expected = nx.eigenvector_centrality(H)

        for node in G.nodes():
            assert centrality[node] == pytest.approx(expected[node], abs=1e-7)

    def test_self_loops_kept(self):
        """Self-loops should be preserved"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 0)  # Self-loop
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        centrality = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")

        # Should have 3 nodes, all with positive centrality
        assert len(centrality) == 3
        assert all(v > 0 for v in centrality.values())

        # Compare with simple graph that has self-loop
        H = nx.Graph()
        H.add_edge(0, 0)  # Self-loop
        H.add_edge(0, 1)
        H.add_edge(1, 2)
        H.add_edge(2, 0)
        expected = nx.eigenvector_centrality(H)

        for node in G.nodes():
            assert centrality[node] == pytest.approx(expected[node], abs=1e-7)

    def test_simple_graph_unchanged(self):
        """Simple Graph should pass through without modification"""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        c1 = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")
        c2 = nx.eigenvector_centrality(G)

        for node in G.nodes():
            assert c1[node] == pytest.approx(c2[node], abs=1e-7)

    def test_digraph_drops_direction(self):
        """DiGraph direction should be dropped"""
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 0)  # Opposite direction
        G.add_edge(1, 2)

        centrality = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")

        # Should be equivalent to undirected path
        H = nx.Graph()
        H.add_edge(0, 1)
        H.add_edge(1, 2)
        expected = nx.eigenvector_centrality(H)

        for node in G.nodes():
            assert centrality[node] == pytest.approx(expected[node], abs=1e-7)

    def test_multigraph_collapses_parallel_edges(self):
        """MultiGraph parallel edges should collapse"""
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)  # Parallel
        G.add_edge(0, 1)  # Parallel
        G.add_edge(1, 2)

        centrality = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")

        # Should be equivalent to simple path
        H = nx.Graph()
        H.add_edge(0, 1)
        H.add_edge(1, 2)
        expected = nx.eigenvector_centrality(H)

        for node in G.nodes():
            assert centrality[node] == pytest.approx(expected[node], abs=1e-7)

    def test_collapse_plan_none_uses_existing_behavior(self):
        """collapse_plan=None should use existing multigraph_weight behavior"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=2)
        G.add_edge(0, 1, weight=3)  # Total weight = 5 with sum
        G.add_edge(1, 2, weight=5)

        # With collapse_plan=None (default), should use multigraph_weight=sum
        c1 = nx.eigenvector_centrality(G, weight="weight", multigraph_weight=sum)

        # Equivalent simple graph with summed weights
        H = nx.Graph()
        H.add_edge(0, 1, weight=5)
        H.add_edge(1, 2, weight=5)
        c2 = nx.eigenvector_centrality(H, weight="weight")

        for node in G.nodes():
            assert c1[node] == pytest.approx(c2[node], abs=1e-7)

    def test_numpy_version_collapses_correctly(self):
        """Verify eigenvector_centrality_numpy also supports collapse_plan"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, weight=10)
        G.add_edge(1, 0, weight=20)
        G.add_edge(0, 1, weight=5)
        G.add_edge(1, 2, weight=1)

        # Both versions should produce equivalent results with collapse
        c_power = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")
        c_numpy = nx.eigenvector_centrality_numpy(G, collapse_plan="unweighted_undirected")

        for node in G.nodes():
            assert c_power[node] == pytest.approx(c_numpy[node], abs=1e-5)

    def test_warning_when_weight_specified_with_collapse_plan(self):
        """Should warn when weight is specified alongside collapse_plan"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, weight=10)
        G.add_edge(1, 2, weight=5)

        with pytest.warns(UserWarning, match="'weight' parameter is ignored"):
            nx.eigenvector_centrality(
                G,
                collapse_plan="unweighted_undirected",
                weight="weight",  # This should trigger warning
            )

    def test_warning_when_multigraph_weight_specified_with_collapse_plan(self):
        """Should warn when multigraph_weight is changed from default alongside collapse_plan"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, weight=10)
        G.add_edge(1, 2, weight=5)

        with pytest.warns(UserWarning, match="'multigraph_weight' parameter is ignored"):
            nx.eigenvector_centrality(
                G,
                collapse_plan="unweighted_undirected",
                multigraph_weight=max,  # Non-default, should trigger warning
            )

    def test_no_warning_with_default_multigraph_weight(self):
        """Should NOT warn when multigraph_weight is left at default"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, weight=10)
        G.add_edge(1, 2, weight=5)

        # No warning should be raised - multigraph_weight is at default (sum)
        import warnings as warn_module

        with warn_module.catch_warnings():
            warn_module.simplefilter("error")  # Turn warnings into errors
            nx.eigenvector_centrality(
                G,
                collapse_plan="unweighted_undirected",
                # multigraph_weight not specified, uses default sum
            )

    def test_collapse_plan_overrides_weight_parameter(self):
        """Verify that weight parameter is truly ignored when collapse_plan is set"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, weight=100)
        G.add_edge(1, 2, weight=1)

        # With collapse_plan, weight should be ignored - results should match unweighted
        with pytest.warns(UserWarning):
            c_with_weight = nx.eigenvector_centrality(
                G,
                collapse_plan="unweighted_undirected",
                weight="weight",
            )

        c_without_weight = nx.eigenvector_centrality(
            G,
            collapse_plan="unweighted_undirected",
        )

        # Results should be identical - weight was ignored
        for node in G.nodes():
            assert c_with_weight[node] == pytest.approx(c_without_weight[node], abs=1e-7)

    def test_numpy_warning_when_weight_specified_with_collapse_plan(self):
        """Should warn when weight is specified alongside collapse_plan in numpy version"""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, weight=10)
        G.add_edge(1, 2, weight=5)

        with pytest.warns(UserWarning, match="'weight' parameter is ignored"):
            nx.eigenvector_centrality_numpy(
                G,
                collapse_plan="unweighted_undirected",
                weight="weight",  # This should trigger warning
            )

    def test_unknown_collapse_plan_raises_error(self):
        """Unknown collapse_plan value should raise NetworkXError"""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        with pytest.raises(nx.NetworkXError, match="Unknown collapse_plan"):
            nx.eigenvector_centrality(G, collapse_plan="invalid_plan")

    def test_numpy_unknown_collapse_plan_raises_error(self):
        """Unknown collapse_plan value should raise NetworkXError in numpy version"""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        with pytest.raises(nx.NetworkXError, match="Unknown collapse_plan"):
            nx.eigenvector_centrality_numpy(G, collapse_plan="invalid_plan")

    def test_isolated_nodes_preserved(self):
        """Isolated nodes should be preserved during collapse - tested with numpy version"""
        G = nx.MultiDiGraph()
        # Create a connected graph with an additional self-loop node for testing
        G.add_edge(0, 1)
        G.add_edge(1, 0)
        G.add_edge(1, 2)
        G.add_edge(2, 1)

        centrality = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")

        # All nodes preserved
        assert len(centrality) == 3
        # All nodes should have positive centrality in connected graph
        assert all(v > 0 for v in centrality.values())

    def test_weighted_graph_weights_dropped(self):
        """Weighted simple graph should have weights dropped"""
        # Use a triangle which is more stable for eigenvector centrality
        G = nx.Graph()
        G.add_edge(0, 1, weight=10)   # Heavy weight
        G.add_edge(1, 2, weight=1)    # Light weight
        G.add_edge(2, 0, weight=1)    # Light weight

        # Without collapse, weights matter
        c_weighted = nx.eigenvector_centrality(G, weight="weight")

        # With collapse, weights are dropped
        c_collapsed = nx.eigenvector_centrality(G, collapse_plan="unweighted_undirected")

        # Unweighted equivalent
        H = nx.Graph()
        H.add_edge(0, 1)
        H.add_edge(1, 2)
        H.add_edge(2, 0)
        c_unweighted = nx.eigenvector_centrality(H)

        # In an unweighted triangle, all nodes have equal centrality
        # Collapsed results should match the unweighted equivalent
        for node in G.nodes():
            assert c_collapsed[node] == pytest.approx(c_unweighted[node], abs=1e-7)
