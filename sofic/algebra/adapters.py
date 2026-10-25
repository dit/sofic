"""Turn sofic models into transformation semigroups.

Every deterministic (unifilar) symbol-labeled presentation -- a DFA, a
:class:`~sofic.automata.unifilar.UnifilarAutomaton`, a unifilar HMM or
ε-machine, or a right-resolving shift presentation -- has an underlying
semiautomaton: one partial map of the states per symbol. Completing the maps
with an absorbing sink gives the transition semigroup.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

from sofic.algebra.transformation import DEFAULT_MAX_SIZE, TransformationSemigroup
from sofic.graph import ATTR_EMISSION_EDGE, ATTR_PROB, ATTR_SYMBOL, EPSILON

__all__ = ["semiautomaton", "transformation_semigroup"]


def semiautomaton(
    machine: Any,
    *,
    alphabet: Iterable[Hashable] | None = None,
) -> tuple[list[Hashable], dict[Hashable, dict[Hashable, Hashable]]]:
    """Return ``(states, maps)`` for a deterministic symbol-labeled machine.

    ``maps[symbol][state]`` is the unique successor. Edges with probability
    zero are ignored, so a stochastic generator contributes its support.
    Raises ``ValueError`` if some state has two successors on one symbol or
    an ε-edge.
    """
    states = list(machine.states())
    symbols: dict[Hashable, None] = dict.fromkeys(alphabet) if alphabet is not None else {}
    if alphabet is None:
        for attr in ("input_alphabet", "symbol_alphabet", "observation_alphabet"):
            declared = getattr(machine, attr, None)
            if declared:
                symbols.update(dict.fromkeys(_sorted(declared)))
                break
    maps: dict[Hashable, dict[Hashable, Hashable]] = {s: {} for s in symbols}
    for transition in machine.transitions():
        symbol = transition.data.get(ATTR_SYMBOL, transition.data.get(ATTR_EMISSION_EDGE))
        if symbol is None:
            continue
        if symbol is EPSILON:
            raise ValueError("ε-transitions have no transition semigroup; remove them first")
        prob = transition.data.get(ATTR_PROB)
        if prob is not None and _is_zero(prob):
            continue
        table = maps.setdefault(symbol, {})
        if table.get(transition.source, transition.target) != transition.target:
            raise ValueError(f"not unifilar: state {transition.source!r} has two successors on {symbol!r}")
        table[transition.source] = transition.target
    return states, maps


def transformation_semigroup(
    machine: Any,
    *,
    alphabet: Iterable[Hashable] | None = None,
    monoid: bool = True,
    max_size: int = DEFAULT_MAX_SIZE,
) -> TransformationSemigroup:
    """Return the transition semigroup of a deterministic presentation.

    Accepts a :class:`TransformationSemigroup` (returned unchanged), a DFA or
    other deterministic labeled automaton, a unifilar HMM or ε-machine
    (probabilities dropped, support kept), or a shift presentation. A
    non-right-resolving :class:`~sofic.shifts.sofic.SoficShift` is replaced by
    its right Fischer cover, the canonical right-resolving presentation
    :cite:`LindMarcus1995`; other presentations are used as given, so the
    result describes *that* presentation. For a presentation-independent
    invariant of a language use :func:`~sofic.algebra.syntactic.syntactic_monoid`.

    Undefined moves go to an added absorbing ``SINK`` state.
    """
    if isinstance(machine, TransformationSemigroup):
        return machine
    from sofic.shifts.sofic import SoficShift

    if isinstance(machine, SoficShift) and not machine.is_unifilar():
        from sofic.shifts.cover_construction import right_fischer_cover

        machine = right_fischer_cover(machine)
    elif hasattr(machine, "is_unifilar") and not machine.is_unifilar():
        raise ValueError(f"{type(machine).__name__} is not unifilar; determinize it first")
    states, maps = semiautomaton(machine, alphabet=alphabet)
    if not states:
        raise ValueError("machine has no states")
    if not maps:
        raise ValueError(f"{type(machine).__name__} has no symbol-labeled edges")
    return TransformationSemigroup.from_maps(maps, states=states, monoid=monoid, max_size=max_size)


def _is_zero(prob: Any) -> bool:
    try:
        return bool(prob == 0)
    except TypeError:
        return False


def _sorted(items: Iterable[Hashable]) -> list[Hashable]:
    values = list(items)
    try:
        return sorted(values)  # type: ignore[type-var]
    except TypeError:
        return sorted(values, key=repr)
