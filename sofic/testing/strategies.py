"""Hypothesis strategies for sofic models.

These helpers live outside the main package imports so Hypothesis remains a
test-only dependency. Install ``sofic[test]`` to use them.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

from sofic.automata.buchi import BuchiAutomaton
from sofic.automata.dfa import DFA
from sofic.automata.enumeration.icdfa import icdfa_string_to_dfa, iter_icdfa_empty_strings
from sofic.automata.nfa import NFA
from sofic.automata.nwa import NestedWordAutomaton
from sofic.automata.transducers import MealyMachine
from sofic.automata.vpa import VisiblyPushdownAutomaton
from sofic.exceptions import SoficValidationError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.markov import MarkovChain
from sofic.generators.mealy import MealyHMM
from sofic.generators.topological_epsilon_enumeration import (
    idfa_string_to_epsilon_machine,
    iter_topological_epsilon_strings,
)
from sofic.graph import EPSILON
from sofic.shifts.sft import ShiftOfFiniteType
from sofic.shifts.sofic import SoficShift

DEFAULT_ALPHABET = ("0", "1")
DEFAULT_OUTPUT_ALPHABET = ("a", "b")


def dfas(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
) -> Any:
    """Return a strategy for complete initially-connected DFAs.

    The DFA transition structures are drawn from the canonical ICDFA
    enumeration, and accepting states are drawn independently.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    k = len(symbols)

    @st.composite
    def strategy(draw: Any) -> DFA:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        transitions = draw(st.sampled_from(_icdfa_transition_strings(k, n)))
        final_states = draw(st.frozensets(st.integers(min_value=0, max_value=n - 1)))
        return icdfa_string_to_dfa(
            transitions,
            symbols,
            n=n,
            k=k,
            final_states=final_states,
            symbol_order=symbols,
        )

    return strategy()


def epsilon_machines(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
    max_pool: int | None = None,
) -> Any:
    """Return a strategy for topological epsilon-machines.

    Machines are drawn from the canonical topological epsilon-machine
    enumeration and use the enumeration module's uniform row probabilities.

    When ``max_pool`` is set, only the first ``max_pool`` enumerated transition
    strings per ``(k, n)`` are retained.  Use this for large alphabets (e.g.
    ``k = 3``) where the full enumeration is infeasible.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    if max_pool is not None and max_pool < 1:
        raise ValueError("max_pool must be positive when provided")
    k = len(symbols)
    state_counts: tuple[int, ...] = ()
    for n in range(min_states, max_states + 1):
        if max_pool is not None:
            pool = _bounded_topological_epsilon_transition_strings(k, n, max_pool)
        else:
            pool = _topological_epsilon_transition_strings(k, n)
        if pool:
            state_counts += (n,)
    if not state_counts:
        raise ValueError("no topological epsilon-machine strings exist for the requested bounds")

    @st.composite
    def strategy(draw: Any) -> EpsilonMachine:
        n = draw(st.sampled_from(state_counts))
        if max_pool is not None:
            pool = _bounded_topological_epsilon_transition_strings(k, n, max_pool)
        else:
            pool = _topological_epsilon_transition_strings(k, n)
        transitions = draw(st.sampled_from(pool))
        return idfa_string_to_epsilon_machine(transitions, n=n, k=k, alphabet=symbols)

    return strategy()


def nfas(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
    max_transitions: int = 6,
    allow_epsilon: bool = True,
) -> Any:
    """Return a strategy for small, possibly nondeterministic NFAs.

    States are ``0, ..., n-1``. The initial set is a non-empty random subset,
    the accepting set an arbitrary subset, and up to ``max_transitions``
    distinct labeled edges are drawn; when ``allow_epsilon`` is true, edges may
    carry :data:`~sofic.graph.EPSILON`.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    _validate_nonnegative("max_transitions", max_transitions)

    @st.composite
    def strategy(draw: Any) -> NFA:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        return _draw_labeled_automaton(draw, NFA, symbols, n, max_transitions, allow_epsilon)

    return strategy()


def buchi_automata(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
    max_transitions: int = 6,
    allow_epsilon: bool = False,
) -> Any:
    """Return a strategy for small nondeterministic Büchi automata.

    Same shape as :func:`nfas`; accepting states are the Büchi set. Epsilon
    edges are off by default.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    _validate_nonnegative("max_transitions", max_transitions)

    @st.composite
    def strategy(draw: Any) -> BuchiAutomaton:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        return _draw_labeled_automaton(draw, BuchiAutomaton, symbols, n, max_transitions, allow_epsilon)

    return strategy()


def lassos(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    max_prefix_length: int = 3,
    max_loop_length: int = 3,
) -> Any:
    """Return a strategy for ultimately periodic words ``(prefix, loop)``.

    Both parts are tuples over ``alphabet``; ``loop`` is non-empty.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_nonnegative("max_prefix_length", max_prefix_length)
    _validate_positive("max_loop_length", max_loop_length)
    symbol = st.sampled_from(symbols)
    return st.tuples(
        st.lists(symbol, max_size=max_prefix_length).map(tuple),
        st.lists(symbol, min_size=1, max_size=max_loop_length).map(tuple),
    )


def wheeler_nfas(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 4,
    max_extra_transitions: int = 3,
) -> Any:
    """Return a strategy for epsilon-free NFAs that satisfy the Wheeler axioms.

    A random rank order is fixed first: rank ``0`` is the unique source and
    initial state, and ranks ``1, ..., n-1`` receive non-decreasing incoming
    labels (input consistency plus the first axiom). For each label, sorted
    source and target multisets are zipped, so ``u < v`` implies
    ``delta(u) <= delta(v)`` (the second axiom). State names are a random
    permutation of the ranks. Labels are ordered by ``repr``, matching the
    default ``symbol_key`` of :func:`sofic.automata.wheeler.is_wheeler`.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    _validate_nonnegative("max_extra_transitions", max_extra_transitions)
    ordered = tuple(sorted(symbols, key=repr))

    @st.composite
    def strategy(draw: Any) -> NFA:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        drawn = draw(st.lists(st.sampled_from(ordered), min_size=n - 1, max_size=n - 1))
        labels = sorted(drawn, key=ordered.index)
        names = draw(st.permutations(range(n)))
        nfa = NFA(
            input_alphabet=frozenset(symbols),
            initial_states=frozenset({names[0]}),
            accepting_states=draw(st.frozensets(st.sampled_from(names))),
        )
        for state in names:
            nfa.graph.add_state(state)
        for symbol in ordered:
            block = [rank for rank in range(1, n) if labels[rank - 1] == symbol]
            if not block:
                continue
            extra = draw(st.lists(st.sampled_from(block), max_size=max_extra_transitions))
            targets = sorted(block + extra)
            sources = sorted(draw(st.lists(st.integers(0, n - 1), min_size=len(targets), max_size=len(targets))))
            for source, target in sorted(set(zip(sources, targets, strict=True))):
                nfa.add_transition(names[source], names[target], symbol)
        return nfa

    return strategy()


def markov_chains(
    *,
    min_states: int = 1,
    max_states: int = 3,
    max_out_degree: int = 3,
) -> Any:
    """Return a strategy for small Markov chains.

    Every state gets ``1..max_out_degree`` distinct successors with positive
    integer weights normalized per row; the initial distribution is positive on
    a random non-empty subset of states.
    """
    st = _hypothesis_strategies()
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    _validate_positive("max_out_degree", max_out_degree)

    @st.composite
    def strategy(draw: Any) -> MarkovChain:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        chain = MarkovChain(initial_distribution=_draw_distribution(draw, n))
        for state in range(n):
            chain.graph.add_state(state)
        for source in range(n):
            targets = draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=max_out_degree, unique=True))
            for target, prob in zip(targets, _draw_row(draw, len(targets)), strict=True):
                chain.add_transition(source, target, prob)
        return chain

    return strategy()


def mealy_hmms(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
    max_out_degree: int = 3,
) -> Any:
    """Return a strategy for small, generally non-unifilar Mealy HMMs.

    Every state gets ``1..max_out_degree`` distinct ``(target, symbol)`` edges
    with positive integer weights normalized per state; the initial
    distribution is positive on a random non-empty subset of states.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    _validate_positive("max_out_degree", max_out_degree)

    @st.composite
    def strategy(draw: Any) -> MealyHMM:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        hmm = MealyHMM(observation_alphabet=frozenset(symbols), initial_distribution=_draw_distribution(draw, n))
        for state in range(n):
            hmm.graph.add_state(state)
        edge = st.tuples(st.integers(0, n - 1), st.sampled_from(symbols))
        for source in range(n):
            edges = draw(st.lists(edge, min_size=1, max_size=max_out_degree, unique=True))
            for (target, symbol), prob in zip(edges, _draw_row(draw, len(edges)), strict=True):
                hmm.add_transition(source, target, symbol, prob)
        return hmm

    return strategy()


def sofic_shifts(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
    max_out_degree: int = 3,
) -> Any:
    """Return a strategy for sofic shifts given by random labeled graphs.

    Every state gets ``1..max_out_degree`` distinct ``(target, symbol)`` edges,
    so presentations are free of dead ends but generally not right-resolving.
    """
    st = _hypothesis_strategies()
    symbols = _validate_alphabet(alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    _validate_positive("max_out_degree", max_out_degree)

    @st.composite
    def strategy(draw: Any) -> SoficShift:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        shift = SoficShift(symbol_alphabet=frozenset(symbols))
        for state in range(n):
            shift.graph.add_state(state)
        edge = st.tuples(st.integers(0, n - 1), st.sampled_from(symbols))
        for source in range(n):
            for target, symbol in draw(st.lists(edge, min_size=1, max_size=max_out_degree, unique=True)):
                shift.add_transition(source, target, symbol)
        return shift

    return strategy()


def sfts(
    *,
    alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    max_forbidden: int = 3,
    max_word_length: int = 3,
    max_states: int = 16,
) -> Any:
    """Return a strategy for SFTs built from random forbidden-word sets.

    Draws up to ``max_forbidden`` words of length ``1..max_word_length`` and
    calls :meth:`~sofic.shifts.sft.ShiftOfFiniteType.from_forbidden_words`;
    sets whose presentation exceeds ``max_states`` are rejected via
    ``assume``.
    """
    st = _hypothesis_strategies()
    assume = _hypothesis_assume()
    symbols = _validate_alphabet(alphabet)
    _validate_nonnegative("max_forbidden", max_forbidden)
    _validate_positive("max_word_length", max_word_length)
    _validate_positive("max_states", max_states)
    word = st.lists(st.sampled_from(symbols), min_size=1, max_size=max_word_length).map(tuple)

    @st.composite
    def strategy(draw: Any) -> ShiftOfFiniteType:
        forbidden = draw(st.sets(word, max_size=max_forbidden))
        try:
            return ShiftOfFiniteType.from_forbidden_words(forbidden, frozenset(symbols), max_states=max_states)
        except SoficValidationError:
            assume(False)
            raise  # pragma: no cover - assume(False) aborts the example

    return strategy()


def vpas(
    *,
    call_alphabet: Sequence[Any] = ("c",),
    return_alphabet: Sequence[Any] = ("r",),
    internal_alphabet: Sequence[Any] = ("i",),
    stack_alphabet: Sequence[Any] = ("A", "B"),
    bottom_stack_symbol: Any = "Z",
    min_states: int = 1,
    max_states: int = 3,
) -> Any:
    """Return a strategy for small, possibly nondeterministic VPAs.

    The bottom symbol is included or omitted at random. Returns may be guarded
    by a stack symbol, the bottom symbol (when present), or nothing (wildcard).
    """
    st = _hypothesis_strategies()
    calls, returns, internals, stack = _validate_visible_alphabets(
        call_alphabet, return_alphabet, internal_alphabet, stack_alphabet, bottom_stack_symbol
    )
    _validate_state_bounds(min_states=min_states, max_states=max_states)

    @st.composite
    def strategy(draw: Any) -> VisiblyPushdownAutomaton:
        n = draw(st.integers(min_states, max_states))
        states = list(range(n))
        with_bottom = draw(st.booleans())
        bottom = bottom_stack_symbol if with_bottom else None
        vpa = VisiblyPushdownAutomaton(
            call_alphabet=frozenset(calls),
            return_alphabet=frozenset(returns),
            internal_alphabet=frozenset(internals),
            stack_alphabet=frozenset(stack) | ({bottom} if with_bottom else frozenset()),
            bottom_stack_symbol=bottom,
            initial_state=0,
            accepting_states=frozenset(draw(st.sets(st.sampled_from(states)))),
        )
        for state in states:
            vpa.graph.add_state(state)
        state = st.sampled_from(states)
        for source, target, symbol in draw(st.lists(st.tuples(state, state, st.sampled_from(internals)), max_size=4)):
            vpa.add_internal_transition(source, target, symbol)
        calls_drawn = draw(
            st.lists(st.tuples(state, state, st.sampled_from(calls), st.sampled_from(stack)), max_size=3)
        )
        for source, target, symbol, push in calls_drawn:
            vpa.add_call_transition(source, target, symbol, push)
        guards = [*stack, None] + ([bottom] if with_bottom else [])
        returns_drawn = draw(
            st.lists(st.tuples(state, state, st.sampled_from(returns), st.sampled_from(guards)), max_size=3)
        )
        for source, target, symbol, guard in returns_drawn:
            vpa.add_return_transition(source, target, symbol, guard)
        return vpa

    return strategy()


def nwas(
    *,
    call_alphabet: Sequence[Any] = ("c",),
    return_alphabet: Sequence[Any] = ("r",),
    internal_alphabet: Sequence[Any] = ("i",),
    hier_alphabet: Sequence[Any] = ("A", "B"),
    bottom_hier_state: Any = "Z",
    min_states: int = 1,
    max_states: int = 3,
) -> Any:
    """Return a strategy for small, possibly nondeterministic nested word automata.

    The bottom hierarchical state is included or omitted at random. Calls store
    a non-bottom hierarchical state; returns are guarded by any hierarchical
    state, and only a bottom-guarded return fires on a pending return.
    """
    st = _hypothesis_strategies()
    calls, returns, internals, hier = _validate_visible_alphabets(
        call_alphabet, return_alphabet, internal_alphabet, hier_alphabet, bottom_hier_state
    )
    _validate_state_bounds(min_states=min_states, max_states=max_states)

    @st.composite
    def strategy(draw: Any) -> NestedWordAutomaton:
        n = draw(st.integers(min_states, max_states))
        states = list(range(n))
        with_bottom = draw(st.booleans())
        guards = [*hier] + ([bottom_hier_state] if with_bottom else [])
        nwa = NestedWordAutomaton(
            call_alphabet=frozenset(calls),
            return_alphabet=frozenset(returns),
            internal_alphabet=frozenset(internals),
            hier_alphabet=frozenset(guards),
            bottom_hier_state=bottom_hier_state if with_bottom else None,
            initial_state=0,
            accepting_states=frozenset(draw(st.sets(st.sampled_from(states)))),
        )
        for state in states:
            nwa.graph.add_state(state)
        state = st.sampled_from(states)
        for source, target, symbol in draw(st.lists(st.tuples(state, state, st.sampled_from(internals)), max_size=4)):
            nwa.add_internal_transition(source, target, symbol)
        calls_drawn = draw(st.lists(st.tuples(state, state, st.sampled_from(calls), st.sampled_from(hier)), max_size=3))
        for source, target, symbol, hier_state in calls_drawn:
            nwa.add_call_transition(source, target, symbol, hier_state)
        returns_drawn = draw(
            st.lists(st.tuples(state, state, st.sampled_from(returns), st.sampled_from(guards)), max_size=3)
        )
        for source, target, symbol, hier_state in returns_drawn:
            nwa.add_return_transition(source, target, symbol, hier_state)
        return nwa

    return strategy()


def mealy_transducers(
    *,
    input_alphabet: Sequence[Any] = DEFAULT_ALPHABET,
    output_alphabet: Sequence[Any] = DEFAULT_OUTPUT_ALPHABET,
    min_states: int = 1,
    max_states: int = 3,
    max_transitions: int = 6,
    allow_epsilon_output: bool = True,
) -> Any:
    """Return a strategy for small nondeterministic Mealy transducers.

    Every edge reads one input symbol (no epsilon inputs, so transductions are
    finite) and writes one output symbol, or :data:`~sofic.graph.EPSILON` when
    ``allow_epsilon_output`` is true. The initial set is a non-empty subset.
    """
    st = _hypothesis_strategies()
    inputs = _validate_alphabet(input_alphabet)
    outputs = _validate_alphabet(output_alphabet)
    _validate_state_bounds(min_states=min_states, max_states=max_states)
    _validate_nonnegative("max_transitions", max_transitions)
    emitted = outputs + ((EPSILON,) if allow_epsilon_output else ())

    @st.composite
    def strategy(draw: Any) -> MealyMachine:
        n = draw(st.integers(min_value=min_states, max_value=max_states))
        state = st.integers(0, n - 1)
        machine = MealyMachine(
            input_alphabet=frozenset(inputs),
            output_alphabet=frozenset(outputs),
            initial_states=draw(st.frozensets(state, min_size=1)),
        )
        for index in range(n):
            machine.graph.add_state(index)
        edge = st.tuples(state, state, st.sampled_from(inputs), st.sampled_from(emitted))
        for source, target, symbol, output in draw(st.lists(edge, max_size=max_transitions, unique=True)):
            machine.add_transition(source, target, symbol, output)
        return machine

    return strategy()


def _draw_labeled_automaton(
    draw: Any,
    cls: type[NFA],
    symbols: tuple[Any, ...],
    n: int,
    max_transitions: int,
    allow_epsilon: bool,
) -> Any:
    st = _hypothesis_strategies()
    state = st.integers(0, n - 1)
    labels = symbols + ((EPSILON,) if allow_epsilon else ())
    automaton = cls(
        input_alphabet=frozenset(symbols),
        initial_states=draw(st.frozensets(state, min_size=1)),
        accepting_states=draw(st.frozensets(state)),
    )
    for index in range(n):
        automaton.graph.add_state(index)
    edge = st.tuples(state, state, st.sampled_from(labels))
    for source, target, symbol in draw(st.lists(edge, max_size=max_transitions, unique=True)):
        automaton.add_transition(source, target, symbol)
    return automaton


def _draw_row(draw: Any, size: int) -> list[float]:
    st = _hypothesis_strategies()
    weights = draw(st.lists(st.integers(1, 9), min_size=size, max_size=size))
    total = sum(weights)
    return [weight / total for weight in weights]


def _draw_distribution(draw: Any, n: int) -> dict[int, float]:
    st = _hypothesis_strategies()
    support = draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=n, unique=True))
    return dict(zip(support, _draw_row(draw, len(support)), strict=True))


def _hypothesis_assume() -> Any:
    _hypothesis_strategies()
    from hypothesis import assume

    return assume


def _hypothesis_strategies() -> Any:
    try:
        from hypothesis import strategies as st
    except ImportError as exc:  # pragma: no cover - exercised only without test extra
        raise ImportError("sofic.testing.strategies requires Hypothesis; install sofic[test].") from exc
    return st


def _validate_alphabet(alphabet: Sequence[Any]) -> tuple[Any, ...]:
    symbols = tuple(alphabet)
    if not symbols:
        raise ValueError("alphabet must be non-empty")
    try:
        unique = frozenset(symbols)
    except TypeError as exc:
        raise ValueError("alphabet symbols must be hashable") from exc
    if len(unique) != len(symbols):
        raise ValueError("alphabet symbols must be unique")
    return symbols


def _validate_visible_alphabets(
    calls: Sequence[Any],
    returns: Sequence[Any],
    internals: Sequence[Any],
    stack: Sequence[Any],
    bottom: Any,
) -> tuple[tuple[Any, ...], ...]:
    roles = tuple(_validate_alphabet(alphabet) for alphabet in (calls, returns, internals))
    if len(frozenset().union(*roles)) != sum(len(role) for role in roles):
        raise ValueError("call, return, and internal alphabets must be disjoint")
    stack_symbols = _validate_alphabet(stack)
    if bottom in stack_symbols:
        raise ValueError("bottom symbol must not be in the pushable stack alphabet")
    return (*roles, stack_symbols)


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_state_bounds(*, min_states: int, max_states: int) -> None:
    if min_states < 1:
        raise ValueError("min_states must be positive")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if min_states > max_states:
        raise ValueError("min_states must be less than or equal to max_states")


@cache
def _icdfa_transition_strings(k: int, n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(iter_icdfa_empty_strings(k, n))


@cache
def _topological_epsilon_transition_strings(k: int, n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(iter_topological_epsilon_strings(k, n))


@cache
def _bounded_topological_epsilon_transition_strings(
    k: int,
    n: int,
    max_pool: int,
) -> tuple[tuple[int, ...], ...]:
    pool: list[tuple[int, ...]] = []
    for transitions in iter_topological_epsilon_strings(k, n):
        pool.append(tuple(transitions))
        if len(pool) >= max_pool:
            break
    return tuple(pool)
