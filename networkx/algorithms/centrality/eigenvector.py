"""Functions for computing eigenvector centrality."""

import math
import warnings

import networkx as nx

__all__ = ["eigenvector_centrality", "eigenvector_centrality_numpy"]


def _get_edge_weight(G, u, v, weight, multigraph_weight=sum):
    """Get aggregated edge weight between nodes u and v.

    For simple graphs, returns the edge weight attribute (or 1 if unweighted).
    For multigraphs, aggregates weights of all parallel edges using the
    specified aggregation function.

    Parameters
    ----------
    G : NetworkX graph
        The graph containing the edge.
    u, v : nodes
        The endpoints of the edge.
    weight : string or None
        The edge attribute name for weights. If None, each edge has weight 1.
    multigraph_weight : callable, optional (default=sum)
        Function to aggregate weights of parallel edges in multigraphs.
        Common options: sum, max, min, statistics.mean.
        For unweighted multigraphs, aggregates edge counts.

    Returns
    -------
    float
        The (aggregated) edge weight.

    Notes
    -----
    For multigraphs without a weight attribute specified (weight=None),
    this returns the count of parallel edges (len(edges)), which represents
    the "strength" of the connection through multiplicity.
    """
    if G.is_multigraph():
        edges = G[u][v]  # dict of edge_key -> edge_attrs
        if weight is not None:
            weights = [attrs.get(weight, 1) for attrs in edges.values()]
            return multigraph_weight(weights)
        else:
            # For unweighted multigraphs, return count of parallel edges
            return len(edges)
    else:
        # Simple graph
        if weight is not None:
            return G[u][v].get(weight, 1)
        else:
            return 1


def _collapse_to_simple_graph(G):
    """Collapse any graph to a simple unweighted undirected graph.

    This function:
    1. Drops all edge weights
    2. Converts to undirected (drops direction)
    3. Collapses parallel edges (binary adjacency: edge exists or not)
    4. Keeps self-loops

    Parameters
    ----------
    G : NetworkX graph
        Input graph (Graph, DiGraph, MultiGraph, or MultiDiGraph)

    Returns
    -------
    H : nx.Graph
        Simple undirected unweighted graph with the same nodes as G.
        Edges are unweighted (no weight attribute).
        Self-loops from G are preserved.

    Notes
    -----
    This is intentionally the simplest possible collapse. For weighted
    collapse with aggregation (sum, max, avg), use the existing
    `multigraph_weight` parameter in `eigenvector_centrality()`.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # Add edges - no weights, just binary adjacency
    # Using a set to deduplicate (handles parallel edges and bidirectional)
    edges_seen = set()
    for u, v in G.edges():
        # Normalize to undirected form (smaller node first, except self-loops)
        edge = (min(u, v), max(u, v)) if u != v else (u, v)
        if edge not in edges_seen:
            edges_seen.add(edge)
            H.add_edge(u, v)  # No weight attribute

    return H


@nx._dispatchable(edge_attrs="weight")
def eigenvector_centrality(
    G, max_iter=100, tol=1.0e-6, nstart=None, weight=None, multigraph_weight=sum, collapse_plan=None
):
    r"""Compute the eigenvector centrality for the graph G.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node $i$ is the
    $i$-th element of a left eigenvector associated with the eigenvalue $\lambda$
    of maximum modulus that is positive. Such an eigenvector $x$ is
    defined up to a multiplicative constant by the equation

    .. math::

         \lambda x^T = x^T A,

    where $A$ is the adjacency matrix of the graph G. By definition of
    row-column product, the equation above is equivalent to

    .. math::

        \lambda x_i = \sum_{j\to i}x_j.

    That is, adding the eigenvector centralities of the predecessors of
    $i$ one obtains the eigenvector centrality of $i$ multiplied by
    $\lambda$. In the case of undirected graphs, $x$ also solves the familiar
    right-eigenvector equation $Ax = \lambda x$.

    By virtue of the Perron–Frobenius theorem [1]_, if G is strongly
    connected there is a unique eigenvector $x$, and all its entries
    are strictly positive.

    If G is not strongly connected there might be several left
    eigenvectors associated with $\lambda$, and some of their elements
    might be zero.

    Parameters
    ----------
    G : graph
      A networkx graph.

    max_iter : integer, optional (default=100)
      Maximum number of power iterations.

    tol : float, optional (default=1.0e-6)
      Error tolerance (in Euclidean norm) used to check convergence in
      power iteration.

    nstart : dictionary, optional (default=None)
      Starting value of power iteration for each node. Must have a nonzero
      projection on the desired eigenvector for the power method to converge.
      If None, this implementation uses an all-ones vector, which is a safe
      choice.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal. Otherwise holds the
      name of the edge attribute used as weight. In this measure the
      weight is interpreted as the connection strength.

    multigraph_weight : callable, optional (default=sum)
        For multigraphs, a function to aggregate weights of parallel edges.
        The function should accept a sequence of weights and return a single
        value. Common options: sum, max, min, statistics.mean.
        For unweighted multigraphs (weight=None), this aggregates edge counts.
        Ignored for simple graphs.

    collapse_plan : str or None, optional (default=None)
        If specified, collapse the graph before computing centrality.
        This is a master switch that overrides `weight` and `multigraph_weight`.
        Options:
        - None: No collapse. Use graph as-is with `multigraph_weight` for
          parallel edge aggregation (existing behavior).
        - "unweighted_undirected": Collapse to simple unweighted undirected graph.
          Drops all weights, drops direction, keeps self-loops.
          Ignores `weight` and `multigraph_weight` parameters when set.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value. The
       associated vector has unit Euclidean norm and the values are
       nonegative.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality(G)
    >>> sorted((v, f"{c:0.2f}") for v, c in centrality.items())
    [(0, '0.37'), (1, '0.60'), (2, '0.60'), (3, '0.37')]

    Raises
    ------
    NetworkXPointlessConcept
        If the graph G is the null graph.

    NetworkXError
        If each value in `nstart` is zero, or if `collapse_plan` is set to
        an unknown value.

    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    See Also
    --------
    eigenvector_centrality_numpy
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Eigenvector centrality was introduced by Landau [2]_ for chess
    tournaments. It was later rediscovered by Wei [3]_ and then
    popularized by Kendall [4]_ in the context of sport ranking. Berge
    introduced a general definition for graphs based on social connections
    [5]_. Bonacich [6]_ reintroduced again eigenvector centrality and made
    it popular in link analysis.

    This function computes the left dominant eigenvector, which corresponds
    to adding the centrality of predecessors: this is the usual approach.
    To add the centrality of successors first reverse the graph with
    ``G.reverse()``.

    The implementation uses power iteration [7]_ to compute a dominant
    eigenvector starting from the provided vector `nstart`. Convergence is
    guaranteed as long as `nstart` has a nonzero projection on a dominant
    eigenvector, which certainly happens using the default value.

    The method stops when the change in the computed vector between two
    iterations is smaller than an error tolerance of ``G.number_of_nodes()
    * tol`` or after ``max_iter`` iterations, but in the second case it
    raises an exception.

    This implementation uses $(A + I)$ rather than the adjacency matrix
    $A$ because the change preserves eigenvectors, but it shifts the
    spectrum, thus guaranteeing convergence even for networks with
    negative eigenvalues of maximum modulus.

    For multigraphs, parallel edges between the same pair of nodes are
    aggregated using the `multigraph_weight` function (default: sum).
    This means that multiple edges strengthen the connection between nodes.
    If `weight=None` for a multigraph, the number of parallel edges is used
    as the effective weight.

    When `collapse_plan="unweighted_undirected"` is specified, the graph is
    first collapsed to a simple unweighted undirected graph before computing
    centrality. This means:
    - All edge weights are discarded (adjacency is binary: 0 or 1)
    - All edge directions are removed (A->B and B->A become single A-B)
    - All parallel edges collapse to single edges
    - Self-loops are preserved

    The `collapse_plan` parameter overrides `weight` and `multigraph_weight`
    when set. If conflicting parameters are specified, a UserWarning is emitted.

    References
    ----------
    .. [1] Abraham Berman and Robert J. Plemmons.
       "Nonnegative Matrices in the Mathematical Sciences."
       Classics in Applied Mathematics. SIAM, 1994.

    .. [2] Edmund Landau.
       "Zur relativen Wertbemessung der Turnierresultate."
       Deutsches Wochenschach, 11:366–369, 1895.

    .. [3] Teh-Hsing Wei.
       "The Algebraic Foundations of Ranking Theory."
       PhD thesis, University of Cambridge, 1952.

    .. [4] Maurice G. Kendall.
       "Further contributions to the theory of paired comparisons."
       Biometrics, 11(1):43–62, 1955.
       https://www.jstor.org/stable/3001479

    .. [5] Claude Berge
       "Théorie des graphes et ses applications."
       Dunod, Paris, France, 1958.

    .. [6] Phillip Bonacich.
       "Technique for analyzing overlapping memberships."
       Sociological Methodology, 4:176–185, 1972.
       https://www.jstor.org/stable/270732

    .. [7] Power iteration:: https://en.wikipedia.org/wiki/Power_iteration

    """
    # ===== OVERRIDE LOGIC: collapse_plan is the master switch =====
    if collapse_plan is not None:
        # Warn if user specified parameters that will be ignored
        if weight is not None:
            warnings.warn(
                f"collapse_plan='{collapse_plan}' is set; 'weight' parameter is ignored. "
                "To use weighted centrality, set collapse_plan=None.",
                UserWarning,
                stacklevel=2,
            )
        if multigraph_weight is not sum:
            warnings.warn(
                f"collapse_plan='{collapse_plan}' is set; 'multigraph_weight' parameter is ignored. "
                "To use custom edge aggregation, set collapse_plan=None.",
                UserWarning,
                stacklevel=2,
            )

        # Apply the collapse plan
        if collapse_plan == "unweighted_undirected":
            G = _collapse_to_simple_graph(G)
            weight = None  # OVERRIDE: force unweighted
            # multigraph_weight is now irrelevant (G is simple graph)
        else:
            raise nx.NetworkXError(f"Unknown collapse_plan: '{collapse_plan}'")
    # ===== END OVERRIDE LOGIC =====

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    # If no initial vector is provided, start with the all-ones vector.
    if nstart is None:
        nstart = {v: 1 for v in G}
    if all(v == 0 for v in nstart.values()):
        raise nx.NetworkXError("initial vector cannot have all zero values")
    # Normalize the initial vector so that each entry is in [0, 1]. This is
    # guaranteed to never have a divide-by-zero error by the previous line.
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    nnodes = G.number_of_nodes()
    # make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = xlast.copy()  # Start with xlast times I to iterate with (A+I)
        # do the multiplication y^T = x^T A (left eigenvector)
        for n in x:
            for nbr in G[n]:
                w = _get_edge_weight(G, n, nbr, weight, multigraph_weight)
                x[nbr] += xlast[n] * w
        # Normalize the vector. The normalization denominator `norm`
        # should never be zero by the Perron--Frobenius
        # theorem. However, in case it is due to numerical error, we
        # assume the norm to be one instead.
        norm = math.hypot(*x.values()) or 1
        x = {k: v / norm for k, v in x.items()}
        # Check for convergence (in the L_1 norm).
        if sum(abs(x[n] - xlast[n]) for n in x) < nnodes * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


@nx._dispatchable(edge_attrs="weight")
def eigenvector_centrality_numpy(G, weight=None, max_iter=50, tol=0, collapse_plan=None):
    r"""Compute the eigenvector centrality for the graph `G`.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node $i$ is the
    $i$-th element of a left eigenvector associated with the eigenvalue $\lambda$
    of maximum modulus that is positive. Such an eigenvector $x$ is
    defined up to a multiplicative constant by the equation

    .. math::

         \lambda x^T = x^T A,

    where $A$ is the adjacency matrix of the graph `G`. By definition of
    row-column product, the equation above is equivalent to

    .. math::

        \lambda x_i = \sum_{j\to i}x_j.

    That is, adding the eigenvector centralities of the predecessors of
    $i$ one obtains the eigenvector centrality of $i$ multiplied by
    $\lambda$. In the case of undirected graphs, $x$ also solves the familiar
    right-eigenvector equation $Ax = \lambda x$.

    By virtue of the Perron--Frobenius theorem [1]_, if `G` is (strongly)
    connected, there is a unique eigenvector $x$, and all its entries
    are strictly positive.

    However, if `G` is not (strongly) connected, there might be several left
    eigenvectors associated with $\lambda$, and some of their elements
    might be zero.
    Depending on the method used to choose eigenvectors, round-off error can affect
    which of the infinitely many eigenvectors is reported.
    This can lead to inconsistent results for the same graph,
    which the underlying implementation is not robust to.
    For this reason, only (strongly) connected graphs are accepted.

    Parameters
    ----------
    G : graph
        A connected NetworkX graph.

    weight : None or string, optional (default=None)
        If ``None``, all edge weights are considered equal. Otherwise holds the
        name of the edge attribute used as weight. In this measure the
        weight is interpreted as the connection strength.

    max_iter : integer, optional (default=50)
        Maximum number of Arnoldi update iterations allowed.

    tol : float, optional (default=0)
        Relative accuracy for eigenvalues (stopping criterion).
        The default value of 0 implies machine precision.

    collapse_plan : str or None, optional (default=None)
        If specified, collapse the graph before computing centrality.
        This is a master switch that overrides the `weight` parameter.
        Options:
        - None: No collapse. Use graph as-is (existing behavior).
        - "unweighted_undirected": Collapse to simple unweighted undirected graph.
          Drops all weights, drops direction, keeps self-loops.
          Ignores `weight` parameter when set.

    Returns
    -------
    nodes : dict of nodes
        Dictionary of nodes with eigenvector centrality as the value. The
        associated vector has unit Euclidean norm and the values are
        nonnegative.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality_numpy(G)
    >>> print([f"{node} {centrality[node]:0.2f}" for node in centrality])
    ['0 0.37', '1 0.60', '2 0.60', '3 0.37']

    Raises
    ------
    NetworkXPointlessConcept
        If the graph `G` is the null graph.

    ArpackNoConvergence
        When the requested convergence is not obtained. The currently
        converged eigenvalues and eigenvectors can be found as
        eigenvalues and eigenvectors attributes of the exception object.

    AmbiguousSolution
        If `G` is not connected.

    NetworkXError
        If `collapse_plan` is set to an unknown value.

    See Also
    --------
    :func:`scipy.sparse.linalg.eigs`
    eigenvector_centrality
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Eigenvector centrality was introduced by Landau [2]_ for chess
    tournaments. It was later rediscovered by Wei [3]_ and then
    popularized by Kendall [4]_ in the context of sport ranking. Berge
    introduced a general definition for graphs based on social connections
    [5]_. Bonacich [6]_ reintroduced again eigenvector centrality and made
    it popular in link analysis.

    This function computes the left dominant eigenvector, which corresponds
    to adding the centrality of predecessors: this is the usual approach.
    To add the centrality of successors first reverse the graph with
    ``G.reverse()``.

    This implementation uses the
    :func:`SciPy sparse eigenvalue solver<scipy.sparse.linalg.eigs>` (ARPACK)
    to find the largest eigenvalue/eigenvector pair using Arnoldi iterations
    [7]_.

    When `collapse_plan="unweighted_undirected"` is specified, the graph is
    first collapsed to a simple unweighted undirected graph before computing
    centrality. The `collapse_plan` parameter overrides the `weight` parameter
    when set. If a conflicting `weight` parameter is specified, a UserWarning
    is emitted.

    References
    ----------
    .. [1] Abraham Berman and Robert J. Plemmons.
       "Nonnegative Matrices in the Mathematical Sciences".
       Classics in Applied Mathematics. SIAM, 1994.

    .. [2] Edmund Landau.
       "Zur relativen Wertbemessung der Turnierresultate".
       Deutsches Wochenschach, 11:366--369, 1895.

    .. [3] Teh-Hsing Wei.
       "The Algebraic Foundations of Ranking Theory".
       PhD thesis, University of Cambridge, 1952.

    .. [4] Maurice G. Kendall.
       "Further contributions to the theory of paired comparisons".
       Biometrics, 11(1):43--62, 1955.
       https://www.jstor.org/stable/3001479

    .. [5] Claude Berge.
       "Théorie des graphes et ses applications".
       Dunod, Paris, France, 1958.

    .. [6] Phillip Bonacich.
       "Technique for analyzing overlapping memberships".
       Sociological Methodology, 4:176--185, 1972.
       https://www.jstor.org/stable/270732

    .. [7] Arnoldi, W. E. (1951).
       "The principle of minimized iterations in the solution of the matrix eigenvalue problem".
       Quarterly of Applied Mathematics. 9 (1): 17--29.
       https://doi.org/10.1090/qam/42792
    """
    import numpy as np
    import scipy as sp

    # ===== OVERRIDE LOGIC: collapse_plan is the master switch =====
    if collapse_plan is not None:
        # Warn if user specified parameters that will be ignored
        if weight is not None:
            warnings.warn(
                f"collapse_plan='{collapse_plan}' is set; 'weight' parameter is ignored. "
                "To use weighted centrality, set collapse_plan=None.",
                UserWarning,
                stacklevel=2,
            )

        # Apply the collapse plan
        if collapse_plan == "unweighted_undirected":
            G = _collapse_to_simple_graph(G)
            weight = None  # OVERRIDE: force unweighted
        else:
            raise nx.NetworkXError(f"Unknown collapse_plan: '{collapse_plan}'")
    # ===== END OVERRIDE LOGIC =====

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    connected = nx.is_strongly_connected(G) if G.is_directed() else nx.is_connected(G)
    if not connected:  # See gh-6888.
        raise nx.AmbiguousSolution(
            "`eigenvector_centrality_numpy` does not give consistent results for disconnected graphs"
        )
    M = nx.to_scipy_sparse_array(G, nodelist=list(G), weight=weight, dtype=float)
    _, eigenvector = sp.sparse.linalg.eigs(
        M.T, k=1, which="LR", maxiter=max_iter, tol=tol
    )
    largest = eigenvector.flatten().real
    norm = np.sign(largest.sum()) * sp.linalg.norm(largest)
    return dict(zip(G, (largest / norm).tolist()))
