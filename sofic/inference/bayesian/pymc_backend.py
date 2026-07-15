"""Optional PyMC model builders for Bayesian process inference."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np


def _safe_name(prefix: str, value: object) -> str:
    text = repr(value)
    return prefix + "_" + "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def markov_chain_model(posterior: Any, *, observed_as_counts: bool = False) -> Any:
    """Build a PyMC model for a :class:`MarkovChainPosterior`.

    The default likelihood uses a ``Potential`` with the same ordered-sequence
    evidence convention as cmpy.  ``observed_as_counts=True`` switches to a
    multinomial count likelihood with the same posterior but a different
    marginal likelihood constant.
    """
    import pymc as pm

    contexts = tuple(posterior.contexts)
    symbols = tuple(posterior.alphabet)
    counts = np.zeros((len(contexts), len(symbols)), dtype=int)
    prior_alpha = np.zeros_like(counts, dtype=float)
    for i, context in enumerate(contexts):
        for j, symbol in enumerate(symbols):
            counts[i, j] = int(posterior.counts.get_word_count((*context, symbol)))
            prior_alpha[i, j] = posterior.prior.get_alpha((*context, symbol))

    coords = {
        "context": np.array([str(context) for context in contexts], dtype=object),
        "symbol": np.array([str(symbol) for symbol in symbols], dtype=object),
    }
    with pm.Model(coords=coords) as model:
        theta = pm.Dirichlet("theta", a=prior_alpha, dims=("context", "symbol"))
        if observed_as_counts:
            pm.Multinomial("counts", n=counts.sum(axis=1), p=theta, observed=counts, dims=("context", "symbol"))
        else:
            pm.Potential("sequence_loglik", (counts * pm.math.log(theta)).sum())
    return model


def epsilon_machine_model(posterior: Any, *, start_node: Hashable | None = None) -> Any:
    """Build a fixed-start PyMC model for an epsilon-machine posterior."""
    import pymc as pm

    dist = posterior.dirichlet
    if start_node is None:
        probs = posterior.start_node_probabilities()
        if not probs:
            raise ValueError("no viable start nodes")
        start_node = max(probs, key=probs.get)
    if start_node not in dist.get_possible_start_nodes():
        raise ValueError(f"start node {start_node!r} is not viable")

    rows: dict[Hashable, list[tuple[Hashable, Any]]] = {}
    for edge in dist.valid_edges:
        rows.setdefault(edge[0], []).append(edge)

    with pm.Model() as model:
        for source, edges in rows.items():
            alpha = np.array([dist.get_edge_alpha(start_node, edge) for edge in edges], dtype=float)
            counts = np.array([dist.get_edge_count(start_node, edge) or 0 for edge in edges], dtype=int)
            theta = pm.Dirichlet(_safe_name("theta", source), a=alpha, shape=len(edges))
            pm.Potential(_safe_name("sequence_loglik", source), (counts * pm.math.log(theta)).sum())
    return model
