"""Information carried by each level of a holonomy decomposition.

.. note:: Original, uncited: the literature does not pair holonomy
   coordinates with a stationary measure.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = ["HolonomyInformation", "holonomy_information"]


class HolonomyInformation(NamedTuple):
    """Per-level split of the stationary state entropy.

    ``distribution`` is the joint law of the level coordinates
    ``(L_1, ..., L_h)``, top level first, with ``"*"`` for an unused level.
    ``level_entropies[i]`` is ``H[L_{i+1} | L_1, ..., L_i]``; they sum to
    ``total``, the entropy of the state distribution.
    """

    distribution: Any
    level_entropies: tuple[float, ...]
    total: float


def holonomy_information(machine: Any) -> HolonomyInformation:
    """Split the stationary state entropy across holonomy levels.

    Each state of a unifilar stochastic generator is lifted to its canonical
    maximal tile chain (:meth:`~sofic.algebra.holonomy.HolonomyDecomposition.lift`)
    and so to cascade coordinates ``(L_1, ..., L_h)``. Distinct states get
    distinct coordinates, so ``H[L_1, ..., L_h] = H[S]`` and the chain rule
    gives ``H[S] = sum_i H[L_i | L_<i]``. For an ε-machine ``H[S]`` is the
    statistical complexity ``C_mu``, and the split says how much of the
    stored information is resolved at each level of the hierarchy, from the
    coarsest approximation (top) to the finest.

    The split depends on the chosen lift. A different chain for the same
    state gives a different but equally valid split. The canonical lift
    always takes the first tile containing the state.

    .. note:: Original, uncited.

    Examples
    --------
    >>> from sofic.examples import even_process
    >>> info = holonomy_information(even_process())
    >>> round(info.total, 6) == round(even_process().statistical_complexity(), 6)
    True
    """
    import dit
    from dit.shannon import conditional_entropy, entropy

    from sofic.algebra.adapters import transformation_semigroup
    from sofic.algebra.holonomy import HolonomyDecomposition

    hd = HolonomyDecomposition(transformation_semigroup(machine))
    index = machine.reindex()
    pi = machine.stationary_distribution()
    outcomes = []
    pmf = []
    for i in range(len(index)):
        p = float(pi[i])
        if p <= 0:
            continue
        coords = hd.coordinates(index.state(i))
        outcomes.append(tuple("*" if c is None else str(c) for c in coords))
        pmf.append(p)
    if hd.n_levels == 0:
        return HolonomyInformation(None, (), 0.0)
    dist = dit.Distribution(outcomes, pmf)
    entropies = tuple(float(conditional_entropy(dist, [i], list(range(i)))) for i in range(hd.n_levels))
    return HolonomyInformation(dist, entropies, float(entropy(dist)))
