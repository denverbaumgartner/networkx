"""SALSA (Stochastic Approach for Link-Structure Analysis) of graph structure."""

import networkx as nx

__all__ = ["salsa"]


@nx._dispatchable(edge_attrs="weight")
def salsa(G, weight="weight", normalized=True):
    """Returns SALSA hubs and authorities values for nodes.

    The SALSA algorithm computes two numbers for each node using a stochastic
    approach based on random walks. Authorities estimate the node value based
    on incoming links weighted by the source's out-degree. Hubs estimate the
    node value based on outgoing links weighted by the target's in-degree.

    Unlike HITS which uses iterative mutual reinforcement, SALSA computes
    scores directly in a single pass, making it more efficient and always
    convergent.

    Parameters
    ----------
    G : graph
        A NetworkX graph. Directed graphs are processed as-is. Undirected
        graphs are treated as bidirectional (each edge counts in both
        directions).

    weight : string or None, optional (default="weight")
        Edge data key to use as weight. If None, all edge weights are
        set to 1.

    normalized : bool, optional (default=True)
        Normalize results so that hub values sum to 1 and authority
        values sum to 1.

    Returns
    -------
    (hubs, authorities) : tuple of two dictionaries
        Two dictionaries keyed by node containing the hub and authority
        values as floats.

    See Also
    --------
    hits : HITS algorithm for hub and authority computation
    pagerank : PageRank algorithm for node importance

    Notes
    -----
    The SALSA algorithm was designed for directed graphs but this
    implementation does not check if the input graph is directed and will
    execute on undirected graphs by treating each edge as bidirectional.

    For a node with no outgoing edges, its hub score will be 0.
    For a node with no incoming edges, its authority score will be 0.

    The algorithm runs in O(m) time where m is the number of edges,
    compared to HITS which requires O(m * iterations).

    References
    ----------
    .. [1] Lempel, R. and Moran, S.,
       "SALSA: The Stochastic Approach for Link-Structure Analysis"
       ACM Transactions on Information Systems, 19(2): 131-160, 2001.
       https://doi.org/10.1145/382979.383041
    .. [2] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 1)])
    >>> hubs, authorities = nx.salsa(G)

    With edge weights:

    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2, weight=0.5)
    >>> G.add_edge(1, 3, weight=1.5)
    >>> G.add_edge(2, 3, weight=1.0)
    >>> hubs, authorities = nx.salsa(G, weight="weight")

    Without normalization:

    >>> hubs, authorities = nx.salsa(G, normalized=False)
    """
    return _salsa_scipy(G, weight=weight, normalized=normalized)


def _salsa_scipy(G, weight="weight", normalized=True):
    """SciPy sparse matrix implementation of SALSA.

    Uses sparse matrix operations for efficient computation on large graphs.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    weight : string or None, optional (default="weight")
        Edge data key to use as weight. If None, all edge weights are set to 1.

    normalized : bool, optional (default=True)
        Normalize results so that values sum to 1.

    Returns
    -------
    (hubs, authorities) : tuple of two dictionaries
        Two dictionaries keyed by node containing the hub and authority values.

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.salsa_alg import _salsa_scipy
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
    >>> h, a = _salsa_scipy(G)
    """
    import numpy as np
    import scipy as sp

    if len(G) == 0:
        return {}, {}

    nodelist = list(G)
    n = len(nodelist)

    # Build sparse adjacency matrix with weights
    # A[i,j] = weight of edge from nodelist[i] to nodelist[j]
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)

    # Compute weighted out-degrees (sum of each row)
    out_deg = np.asarray(A.sum(axis=1)).flatten()

    # Compute weighted in-degrees (sum of each column)
    in_deg = np.asarray(A.sum(axis=0)).flatten()

    # Compute authority scores
    # authority[j] = sum over i of: A[i,j] / out_deg[i]
    # This is: (D_out^{-1} @ A).sum(axis=0) where D_out is diagonal out-degree matrix
    # Avoid division by zero by setting 1/0 = 0
    out_deg_inv = np.zeros(n)
    nonzero_out = out_deg > 0
    out_deg_inv[nonzero_out] = 1.0 / out_deg[nonzero_out]

    # Create sparse diagonal matrix for out-degree inverse
    D_out_inv = sp.sparse.diags(out_deg_inv, format="csr")

    # Authority scores: sum columns of D_out_inv @ A
    authority_scores = np.asarray((D_out_inv @ A).sum(axis=0)).flatten()

    # Compute hub scores
    # hub[i] = sum over j of: A[i,j] / in_deg[j]
    # This is: (A @ D_in^{-1}).sum(axis=1) where D_in is diagonal in-degree matrix
    in_deg_inv = np.zeros(n)
    nonzero_in = in_deg > 0
    in_deg_inv[nonzero_in] = 1.0 / in_deg[nonzero_in]

    # Create sparse diagonal matrix for in-degree inverse
    D_in_inv = sp.sparse.diags(in_deg_inv, format="csr")

    # Hub scores: sum rows of A @ D_in_inv
    hub_scores = np.asarray((A @ D_in_inv).sum(axis=1)).flatten()

    # Normalize if requested
    if normalized:
        hub_sum = hub_scores.sum()
        if hub_sum > 0:
            hub_scores /= hub_sum

        authority_sum = authority_scores.sum()
        if authority_sum > 0:
            authority_scores /= authority_sum

    # Convert to dictionaries
    hubs = dict(zip(nodelist, map(float, hub_scores)))
    authorities = dict(zip(nodelist, map(float, authority_scores)))

    return hubs, authorities


def _salsa_python(G, weight="weight", normalized=True):
    """Pure Python implementation of SALSA.

    Uses dictionary-based computation without NumPy/SciPy dependencies.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    weight : string or None, optional (default="weight")
        Edge data key to use as weight. If None, all edge weights are set to 1.

    normalized : bool, optional (default=True)
        Normalize results so that values sum to 1.

    Returns
    -------
    (hubs, authorities) : tuple of two dictionaries
        Two dictionaries keyed by node containing the hub and authority values.

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.salsa_alg import _salsa_python
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
    >>> h, a = _salsa_python(G)
    """
    if len(G) == 0:
        return {}, {}

    # Initialize scores to zero for all nodes
    hubs = dict.fromkeys(G, 0.0)
    authorities = dict.fromkeys(G, 0.0)

    # Compute weighted out-degrees
    out_deg = {}
    for n in G:
        out_deg[n] = sum(
            data.get(weight, 1) if weight is not None else 1
            for _, _, data in G.out_edges(n, data=True)
        )

    # Compute weighted in-degrees
    in_deg = {}
    for n in G:
        in_deg[n] = sum(
            data.get(weight, 1) if weight is not None else 1
            for _, _, data in G.in_edges(n, data=True)
        )

    # Compute scores by iterating over edges
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1) if weight is not None else 1

        # Authority score contribution
        if out_deg[u] > 0:
            authorities[v] += w / out_deg[u]

        # Hub score contribution
        if in_deg[v] > 0:
            hubs[u] += w / in_deg[v]

    # Normalize if requested
    if normalized:
        hub_sum = sum(hubs.values())
        if hub_sum > 0:
            hubs = {n: h / hub_sum for n, h in hubs.items()}

        authority_sum = sum(authorities.values())
        if authority_sum > 0:
            authorities = {n: a / authority_sum for n, a in authorities.items()}

    return hubs, authorities
