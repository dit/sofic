"""Co-lexicographic structure of epsilon-machines.

.. warning::

   Everything in this module is original and uncited. Wheeler automata are a
   purely topological theory :cite:`Gagie2017` :cite:`Alanko2020`; a literature
   search turned up no treatment of weighted, probabilistic, or
   information-theoretic Wheeler automata, so no canonical source exists for
   the quantities defined here. They are offered as proposals, and each
   docstring says so.

The bridge from Wheeler theory to computational mechanics is that
co-lexicographic order compares words from the last symbol backwards, which is
the *recency* order on pasts. A presentation is Wheeler exactly when its state
partition of history space is an interval partition under recency, so each
state owns a contiguous band of pasts rather than an arbitrary set. That makes
the stationary distribution a genuine cumulative distribution over pasts, and
it prices the constraint: a process whose causal states are not recency
intervals must split them to get one, and the extra states cost entropy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sofic.automata.wheeler import WheelerError, WheelerOrder, wheeler_order
from sofic.generators.mealy import MealyHMM

#: How many edge-machine refinements :func:`wheeler_presentation` will try.
DEFAULT_MAX_REFINEMENT = 4

#: Refinements past this many states are abandoned rather than searched.
DEFAULT_MAX_STATES = 2048


def wheeler_presentation(
    machine: Any,
    *,
    max_refinement: int = DEFAULT_MAX_REFINEMENT,
    max_states: int = DEFAULT_MAX_STATES,
) -> MealyHMM:
    """Return a Wheeler presentation of the process ``machine`` generates.

    Returns ``machine`` itself when it is already Wheeler. Otherwise, when the
    Markov order ``R`` is finite, returns the order-``R`` de Bruijn
    presentation of :func:`debruijn_presentation`, whose states *are* the
    length-``R`` words and so sort co-lexicographically by construction.
    Failing that it tries :func:`~sofic.generators.edge_machine.hmm_to_edge_machine`
    refinements, whose order-``k`` states are length-``k`` transition paths and
    are therefore entered on a single symbol.

    Raises :class:`~sofic.automata.wheeler.WheelerError` if no refinement in
    range works. Wheeler languages are star-free :cite:`Alanko2021`, so a
    process that counts modulo anything -- the even process is the standard
    example -- has no Wheeler presentation at any order.

    .. note:: Original, uncited: no literature treats probabilistic Wheeler
       presentations.
    """
    from sofic.generators.edge_machine import hmm_to_edge_machine

    if wheeler_order(machine) is not None:
        return machine

    order = machine.markov_order() if hasattr(machine, "markov_order") else float("inf")
    budget = max_refinement if order == float("inf") else min(int(order), max_refinement)
    if order != float("inf"):
        debruijn = debruijn_presentation(machine, max(int(order), 1))
        if debruijn is not None and wheeler_order(debruijn) is not None:
            return debruijn
    for iterations in range(1, budget + 1):
        refined = hmm_to_edge_machine(machine, iterations=iterations)
        size = len(list(refined.states()))
        if size > max_states:
            raise WheelerError(
                f"refinement {iterations} needs {size} states, over max_states={max_states}; "
                "raise the cap if the process really is Wheeler at this depth"
            )
        if wheeler_order(refined) is not None:
            return refined
    raise WheelerError(
        f"no Wheeler presentation within {budget} edge-machine refinements; Wheeler languages "
        "are star-free, so a process with a nontrivial syntactic group has none at any order"
    )


def debruijn_presentation(machine: Any, order: int) -> MealyHMM | None:
    """Return the order-``order`` de Bruijn presentation, or ``None``.

    States are the length-``order`` words of the process, which is well defined
    only once ``order`` reaches the Markov order ``R``, so that each word pins
    down one causal state; below that this returns ``None``. Because the states
    *are* words, sorting them co-lexicographically satisfies the Wheeler axioms
    outright, which is why every finite-Markov-order process has a Wheeler
    presentation even when its epsilon-machine has none.

    Distinct from :func:`~sofic.generators.edge_machine.hmm_to_edge_machine`,
    whose states are transition paths: a path remembers where it started, so
    the edge machine can stay unsortable where the de Bruijn form is not.

    .. note:: Original, uncited as a Wheeler construction, though the de Bruijn
       presentation itself is standard.
    """
    from sofic.generators.synchronization import graph_from_epsilon_machine
    from sofic.graph import ATTR_EMISSION, ATTR_PROB

    if order < 1:
        raise ValueError("order must be at least one")
    graph = graph_from_epsilon_machine(machine)
    alphabet = sorted(graph.alphabet, key=repr)

    emissions: dict[tuple[Any, Any], tuple[Any, Any]] = {}
    for transition in machine.transitions():
        symbol = transition.data.get(ATTR_EMISSION)
        if symbol is not None:
            emissions[(transition.source, symbol)] = (transition.data[ATTR_PROB], transition.target)

    reached: dict[tuple[Any, ...], frozenset[Any]] = {(): frozenset(graph.states)}
    for _step in range(order):
        extended: dict[tuple[Any, ...], frozenset[Any]] = {}
        for word, states in reached.items():
            for symbol in alphabet:
                image = graph.delta_set(states, symbol)
                if image:
                    extended[(*word, symbol)] = image
        reached = extended
    if not reached or any(len(states) != 1 for states in reached.values()):
        return None

    result = MealyHMM(observation_alphabet=frozenset(alphabet))
    for word in reached:
        result.graph.add_state(word)
    for word, states in reached.items():
        state = next(iter(states))
        for symbol in alphabet:
            found = emissions.get((state, symbol))
            if found is None:
                continue
            probability, _target = found
            successor = (*word, symbol)[-order:]
            if successor in reached:
                result.add_transition(word, successor, symbol, probability)

    index = result.reindex()
    stationary = result.stationary_distribution()
    result.initial_distribution = {index.state(i): float(stationary[i]) for i in range(len(index))}
    return result


def wheeler_statistical_complexity(machine: Any, **kwargs: Any) -> float:
    """``C_W``: the entropy of the states of a Wheeler presentation, in bits.

    Every Wheeler presentation refines the causal-state partition, so
    ``C_W >= C_mu`` always, with equality exactly when the epsilon-machine is
    itself Wheeler. The gap is the entropic price of co-lex sortability: the
    extra memory a process must carry for its states to be intervals of the
    recency order on pasts, over and above the memory needed to predict it.

    A second bound comes free. A Wheeler presentation is input consistent, so
    every state determines the symbol that entered it, making the last symbol a
    function of the state: ``C_W >= H[X_0]`` for a stationary process. A fair
    coin therefore has ``C_mu = 0`` but ``C_W = 1``, since sortability forces it
    to remember a bit it does not need. Together, ``C_W >= max(C_mu, H[X_0])``.

    .. note:: Original, uncited: proposed here, with no canonical source.
    """
    from sofic.generators.measures import state_entropy

    return float(state_entropy(wheeler_presentation(machine, **kwargs)))


def wheeler_complexity_gap(machine: Any, **kwargs: Any) -> float:
    """``C_W - C_mu``: bits of memory spent purely on co-lex sortability.

    Zero exactly when the epsilon-machine is already Wheeler.

    .. note:: Original, uncited: proposed here, with no canonical source.
    """
    return wheeler_statistical_complexity(machine, **kwargs) - float(machine.statistical_complexity())


def colex_cdf(machine: Any) -> tuple[tuple[Any, ...], np.ndarray]:
    """Cumulative stationary distribution over states in Wheeler order.

    Returns the ordered states and the running total of their stationary
    probabilities, so entry ``i`` is the chance the current past falls at or
    below state ``i`` in the recency order. The Wheeler order is what makes
    this cumulative sum mean anything: under any other state numbering the
    partial sums are arbitrary, whereas here they sweep history space from the
    most remote pasts to the most recent, which is what arithmetic coding over
    pasts needs.

    Raises :class:`~sofic.automata.wheeler.WheelerError` if ``machine`` is not
    Wheeler; refine it with :func:`wheeler_presentation` first.

    .. note:: Original, uncited: proposed here, with no canonical source.
    """
    order = _require_order(machine)
    index = machine.reindex()
    stationary = np.asarray(machine.stationary_distribution(), dtype=float)
    masses = np.array([stationary[index.index(state)] for state in order.states])
    return order.states, np.cumsum(masses)


def cylinder_measure(machine: Any, low: int, high: int) -> float:
    """Stationary probability that the current state lies in Wheeler ranks ``[low, high]``.

    Because the Wheeler order sorts states by the co-lex rank of the words
    reaching them, a rank interval is a *cylinder of pasts* under the recency
    order, and this is its measure. Pair it with
    :meth:`~sofic.automata.wheeler_index.WheelerIndex.forward_search`, which
    returns exactly such an interval for a word.

    .. note:: Original, uncited: proposed here, with no canonical source.
    """
    _states, cumulative = colex_cdf(machine)
    if not 0 <= low <= high < len(cumulative):
        raise IndexError(f"interval ({low}, {high}) outside 0..{len(cumulative) - 1}")
    below = cumulative[low - 1] if low > 0 else 0.0
    return float(cumulative[high] - below)


def word_cylinder_measure(machine: Any, word: Sequence[Any]) -> float:
    """Stationary probability that the current state is one reachable by ``word``.

    Not the probability of seeing ``word``: it is the mass of the co-lex
    interval that ``word`` selects, i.e. how much of history space is
    consistent with having just read it.

    .. note:: Original, uncited: proposed here, with no canonical source.
    """
    from sofic.automata.wheeler_index import WheelerIndex

    interval = WheelerIndex.from_model(machine).forward_search(word)
    if interval is None:
        return 0.0
    return cylinder_measure(machine, *interval)


def _require_order(machine: Any) -> WheelerOrder:
    order = wheeler_order(machine)
    if order is None:
        raise WheelerError(f"{type(machine).__qualname__} is not Wheeler; call wheeler_presentation() first")
    return order
