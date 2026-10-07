"""Time-reversal helpers for stochastic generators."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, TypeVar

from sofic.base import StateMachine
from sofic.exceptions import UnifilarityError
from sofic.generators.prob import (
    as_prob,
    has_symbolic,
    is_positive_mass,
    is_zero,
    simplify_prob,
)
from sofic.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, TransitionGraph

S = TypeVar("S", bound=StateMachine)


def is_markov_like(model: StateMachine) -> bool:
    """Return whether ``model`` has only Markov transition probabilities on edges."""
    for state in model.states():
        if model.graph.state_attrs(state).get(ATTR_EMISSION_DIST) is not None:
            return False
    for transition in model.transitions():
        data = transition.data
        if ATTR_EMISSION in data or ATTR_EMISSION_DIST in data:
            return False
        if ATTR_PROB not in data:
            return False
    return True


def reverse_is_finite(model: StateMachine, *, rtol: float = 1e-9) -> bool:
    """Decide whether the reverse mixed-state presentation of a unifilar ``model`` is finite.

    A finite forward ε-machine does not imply a finite reverse one
    (:cite:`Ellison2011`). Because the
    seed belief is uniform and ``model`` is unifilar, the reverse mixed states
    -- the retrodictive beliefs after finitely many symbols, transient and
    recurrent -- are exactly the normalized vectors
    :math:`(\\Pr(x \\mid s))_{s}` over all words :math:`x`, so they are finite
    iff every log-likelihood ratio
    :math:`\\log \\Pr(x \\mid s) - \\log \\Pr(x \\mid s')` takes finitely many
    values.

    ``True`` implies a finite reverse ε-machine. ``False`` does not imply an
    infinite one: infinitely many transient beliefs may converge to finitely
    many recurrent causal states. Alternating biased coins is an example -- its
    likelihood ratio diverges, yet its reverse ε-machine has two states.

    Reading a symbol ``a`` moves the pair ``(s, s')`` to
    ``(delta(s, a), delta(s', a))`` and multiplies the ratio by
    ``p(a|s) / p(a|s')``. Placing that weight on the pair graph over
    ``states x states`` gives

        the reverse mixed states are finite
          iff every directed cycle of the pair graph has weight one.

    A cycle of weight :math:`\\gamma \\neq 1` traversed :math:`k` times yields
    ratios :math:`\\gamma^k`, so infinitely many beliefs; conversely unit cycle
    weights make the weight a potential difference, so the ratio depends only on
    the current pair and there are at most :math:`\\lvert S \\rvert^2` of them.

    That criterion is the twins property of weighted automata
    :cite:`Mohri2009`, and in this form it is Theorem 5 of
    :cite:`AllauzenMohri2003`: a trim cycle-unambiguous weighted automaton over a
    commutative cancellative semiring has the twins property iff every cycle of
    :math:`A \\cap A^{-1}` -- the pair graph -- has weight one. Unifilarity gives
    exactly one path per (state, word), so the ε-machine is deterministic and
    hence cycle-unambiguous, and :math:`(\\mathbb{R}_{>0}, \\times)` is
    commutative and cancellative. Their Theorem 6 decides it in
    :math:`O(\\lvert Q \\rvert^2 + \\lvert E \\rvert^2)`: cycles live only inside
    strongly connected components, so it suffices to build a potential within
    each component and check every intra-component edge against it. That is the
    algorithm below.

    Their "twins iff determinizable" equivalence is *not* what is being invoked:
    it is stated for trim unambiguous automata over the tropical semiring, and
    they note that over the real semiring twins does not imply determinizable for
    infinitely ambiguous automata. Seeding the belief with full support makes
    every state initial, so this automaton is :math:`\\lvert S \\rvert`-ambiguous
    over the real semiring. The equivalence here rests on the direct argument
    above instead, which unifilarity makes available.

    Note that this is *not* a structural condition. Machines with identical
    transition structure can differ, since a cycle weight can equal one by
    algebraic coincidence.
    """
    _pair_graph, inconsistent = _inconsistent_pair_components(model, rtol=rtol)
    return not inconsistent


def reverse_epsilon_machine_is_finite(model: StateMachine, *, rtol: float = 1e-9) -> bool:
    """Decide whether the reverse ε-machine of a unifilar ``model`` has finitely many recurrent states.

    .. warning::
       Experimental. The criterion below has a proof sketch and agrees with
       sampled posterior limits on every strongly connected 3-state binary
       machine tested, but it has not been published.

    :func:`reverse_is_finite` tests the stronger property that *every* reverse
    belief after finitely many symbols is one of finitely many. A pair of
    forward states ``(s, s')`` whose paths never merge has a log-likelihood
    ratio that diverges almost surely, sending the posterior to a face of the
    simplex; those infinitely many transient beliefs leave no recurrent trace.
    Infinitely many recurrent states need a ratio that grows around a cycle and
    is then *frozen* by the two paths merging, so

        the reverse ε-machine is finite
          iff no strongly connected component of the pair graph that contains
          a cycle of weight other than one can reach the diagonal.

    If such a component exists, traversing its cycle ``k`` times and then
    merging happens with positive probability and freezes the ratio at
    :math:`c \\gamma^k`, a distinct recurrent belief for each ``k``. If none
    exists, merging pairs only cross components with consistent potentials, so
    their frozen ratios take finitely many values. :cite:`Ellison2011`'s
    explosive example and the single-atom binary witness fail it; alternating
    biased coins passes it although :func:`reverse_is_finite` is ``False``.
    """
    import networkx as nx

    pair_graph, inconsistent = _inconsistent_pair_components(model, rtol=rtol)
    if not inconsistent:
        return True
    diagonal = [node for node in pair_graph.nodes if node[0] == node[1]]
    can_merge = set(diagonal)
    for node in diagonal:
        can_merge |= nx.ancestors(pair_graph, node)
    return not any(component & can_merge for component in inconsistent)


def warn_if_reverse_truncated(forward: StateMachine, *, stacklevel: int = 3) -> None:
    """Warn when a numerically built reverse ε-machine must be a finite truncation."""
    import warnings

    if not reverse_epsilon_machine_is_finite(forward):
        warnings.warn(
            "the reverse ε-machine of this process is infinite; the returned machine merges "
            "beliefs within numerical tolerance and is a finite approximation",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def _inconsistent_pair_components(model: StateMachine, *, rtol: float) -> tuple[Any, list[set[Any]]]:
    """Return the pair graph and its strongly connected components with a non-unit cycle."""
    import math

    import networkx as nx

    states = list(model.states())
    edges: dict[tuple[Hashable, Any], tuple[Hashable, Any]] = {}
    for state in states:
        for transition in model.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_EMISSION)
            key = (state, symbol)
            if key in edges:
                raise UnifilarityError(f"state {state!r} has multiple {symbol!r} edges")
            edges[key] = (transition.target, as_prob(transition.data.get(ATTR_PROB, 0.0)))

    symbolic = has_symbolic([prob for _, prob in edges.values()])

    symbols = {symbol for _, symbol in edges}
    # MultiDiGraph: two symbols can carry a pair to the same successor with
    # different ratios, and collapsing those parallel edges would hide the very
    # inconsistency this test looks for.
    pair_graph = nx.MultiDiGraph()
    pair_graph.add_nodes_from((s, t) for s in states for t in states)
    for s in states:
        for t in states:
            for symbol in symbols:
                head, tail = edges.get((s, symbol)), edges.get((t, symbol))
                if head is None or tail is None or is_zero(head[1]) or is_zero(tail[1]):
                    continue
                # Exact machines compare ratios; numeric ones accumulate log-ratios,
                # since products of many ratios underflow and make any absolute
                # tolerance meaningless.
                weight = head[1] / tail[1] if symbolic else math.log(float(head[1]) / float(tail[1]))
                pair_graph.add_edge((s, t), (head[0], tail[0]), weight=weight)

    identity = as_prob(1) if symbolic else 0.0

    def combine(left: Any, right: Any) -> Any:
        return left * right if symbolic else left + right

    def agrees(left: Any, right: Any) -> bool:
        if symbolic:
            return is_zero(simplify_prob(left - right))
        return abs(left - right) <= rtol

    inconsistent: list[set[Any]] = []
    for component in nx.strongly_connected_components(pair_graph):
        if len(component) == 1:
            node = next(iter(component))
            if not pair_graph.has_edge(node, node):
                continue
        sub = pair_graph.subgraph(component)
        root = next(iter(component))
        potential: dict[Any, Any] = {root: identity}
        stack = [root]
        while stack:
            current = stack.pop()
            for _, successor, data in sub.out_edges(current, data=True):
                if successor not in potential:
                    potential[successor] = combine(potential[current], data["weight"])
                    stack.append(successor)
        if any(
            not agrees(combine(potential[source], data["weight"]), potential[target])
            for source, target, data in sub.edges(data=True)
        ):
            inconsistent.append(set(component))
    return pair_graph, inconsistent


def time_reverse_stochastic(model: S) -> S:  # noqa: UP047 - keep Python 3.11 compatibility.
    """Build the time-reversed chain using the forward stationary distribution."""
    pi = model.stationary_distribution()
    idx = model.reindex()
    rev = model.copy()
    rev.graph = TransitionGraph()
    for state in idx.states:
        rev.graph.add_state(state)

    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    for source in idx.states:
        i = idx.index(source)
        for transition in model.graph.out_transitions(source):
            target = transition.target
            j = idx.index(target)
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            if not is_positive_mass(prob) or is_zero(pi[j]):
                continue
            if symbolic or has_symbolic([prob]):
                rev_prob = simplify_prob(as_prob(pi[i]) * as_prob(prob) / as_prob(pi[j]))
            else:
                rev_prob = float(pi[i] * float(prob) / float(pi[j]))
            attrs: dict[str, Any] = {ATTR_PROB: as_prob(rev_prob)}
            if ATTR_EMISSION in transition.data:
                attrs[ATTR_EMISSION] = transition.data[ATTR_EMISSION]
            rev.graph.add_transition(target, source, **attrs)

    if hasattr(rev, "initial_distribution"):
        if symbolic:
            rev.initial_distribution = {idx.state(i): as_prob(pi[i]) for i in range(len(idx))}
        else:
            rev.initial_distribution = {idx.state(i): float(pi[i]) for i in range(len(idx))}
    return rev
