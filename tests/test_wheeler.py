"""Tests for co-lexicographic (Wheeler) ordering."""

import itertools
import random
from collections import defaultdict

import numpy as np
import pytest
from hypothesis import given, settings

from sofic.automata.algorithms import equivalent
from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.automata.wheeler import (
    WheelerError,
    check_wheeler_axioms,
    colex_width,
    is_input_consistent,
    is_wheeler,
    labeled_graph,
    minimum_wdfa,
    wheeler_canonical_form,
    wheeler_isomorphic,
    wheeler_order,
    wheeler_state_index,
    wnfa_to_wdfa,
)
from sofic.automata.wheeler_index import WheelerIndex
from sofic.examples.epsilon_machines import (
    butterfly_process,
    even_process,
    fair_coin,
    golden_mean,
    golden_mean_markov,
    nemo_process,
    wheeler_infinite_order_process,
)
from sofic.generators.synchronization import (
    _interval_power_automaton,
    _subset_power_automaton,
    cryptic_order_from_graph,
    graph_from_epsilon_machine,
    labeled_graph_of,
    markov_order_from_graph,
    reset_threshold_from_graph,
)
from sofic.generators.topological_epsilon_enumeration import iter_topological_epsilon_machines
from sofic.generators.wheeler_epsilon import (
    colex_cdf,
    cylinder_measure,
    debruijn_presentation,
    wheeler_presentation,
    wheeler_statistical_complexity,
    word_cylinder_measure,
)
from sofic.shifts.sft import ShiftOfFiniteType
from sofic.shifts.sofic import SoficShift
from sofic.shifts.wheeler import (
    higher_block_presentation,
    is_wheeler_shift,
    wheeler_cover,
    wheeler_order_of_shift,
)
from sofic.testing.strategies import dfas, epsilon_machines

BINARY = frozenset({0, 1})


def shift_from_edges(edges, alphabet=(0, 1)):
    shift = SoficShift(symbol_alphabet=frozenset(alphabet))
    for state in {end for edge in edges for end in (edge[0], edge[2])}:
        shift.graph.add_state(state)
    for source, symbol, target in edges:
        shift.add_transition(source, target, symbol)
    return shift


GOLDEN_MEAN_SHIFT = ((0, 0, 0), (0, 1, 1), (1, 0, 0))
EVEN_SHIFT = ((0, 0, 0), (0, 1, 1), (1, 1, 0))


def sigma_star_dfa(order):
    """Order-``order`` de Bruijn presentation of ``{a, b}*`` as a Wheeler DFA."""
    words = ["".join(w) for length in range(order + 1) for w in itertools.product("ab", repeat=length)]
    dfa = DFA(
        input_alphabet=frozenset("ab"),
        initial_states=frozenset({""}),
        accepting_states=frozenset(words),
    )
    for word in words:
        dfa.graph.add_state(word)
    for word in words:
        for symbol in "ab":
            dfa.add_transition(word, (word + symbol)[-order:], symbol)
    return dfa


def substring_wnfa(text):
    """Substring automaton of ``text``: every state initial and accepting."""
    nfa = NFA(
        input_alphabet=frozenset(text),
        initial_states=frozenset(range(len(text) + 1)),
        accepting_states=frozenset(range(len(text) + 1)),
    )
    for position in range(len(text) + 1):
        nfa.graph.add_state(position)
    for position, symbol in enumerate(text):
        nfa.add_transition(position, position + 1, symbol)
    return nfa


# -- Known answers ---------------------------------------------------------


def test_golden_mean_is_wheeler_with_a_before_b():
    order = wheeler_order(golden_mean())
    assert order is not None
    assert order.states == ("A", "B")
    assert order.rank == {"A": 0, "B": 1}


@pytest.mark.parametrize("machine", [even_process(), nemo_process(), butterfly_process()])
def test_canonical_non_wheeler_machines(machine):
    assert not is_input_consistent(machine)
    assert not is_wheeler(machine)
    assert colex_width(machine) > 1


@pytest.mark.parametrize("machine", [golden_mean(), golden_mean_markov()])
def test_canonical_wheeler_machines(machine):
    assert is_input_consistent(machine)
    assert is_wheeler(machine)
    assert colex_width(machine) == 1


def test_wheeler_does_not_imply_finite_markov_order():
    machine = wheeler_infinite_order_process()
    assert machine.wheeler_order().states == ("A", "E", "B", "C", "D")
    assert machine.colex_width() == 1
    assert machine.markov_order() == float("inf")


def test_input_consistency_is_necessary_but_not_sufficient():
    # One state entered on two symbols cannot sit in any co-lex order, yet
    # axiom one only compares *distinct* states, so width misses it.
    full_shift = shift_from_edges(((0, 0, 0), (0, 1, 0)))
    assert not is_input_consistent(full_shift)
    assert not is_wheeler(full_shift)
    assert colex_width(full_shift) == 1


def test_wheeler_order_recovers_colex_order_of_words():
    shift = ShiftOfFiniteType.from_forbidden_words({(1, 1)}, BINARY).trim_transient()
    order = wheeler_order(shift)
    assert order is not None
    assert list(order.states) == sorted(order.states, key=lambda word: tuple(reversed(word)))


# -- Enumeration regression ------------------------------------------------


@pytest.mark.parametrize(("states", "expected"), [(1, 2), (2, 3), (3, 12), (4, 49)])
def test_binary_wheeler_counts(states, expected):
    found = sum(1 for m in iter_topological_epsilon_machines(2, states) if is_wheeler(m))
    assert found == expected


@pytest.mark.slow
def test_binary_wheeler_count_five_states():
    found = sum(1 for m in iter_topological_epsilon_machines(2, 5) if is_wheeler(m))
    assert found == 256


@pytest.mark.slow
def test_wheeler_order_agrees_with_brute_force_search():
    for alphabet, states in ((2, 3), (2, 4), (3, 2), (3, 3)):
        for machine in iter_topological_epsilon_machines(alphabet, states):
            graph = labeled_graph(machine)
            brute = any(check_wheeler_axioms(graph, p) for p in itertools.permutations(graph.states))
            assert brute == (wheeler_order(machine) is not None)


# -- Properties ------------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(epsilon_machines(max_states=4))
def test_returned_order_satisfies_the_axioms(machine):
    order = wheeler_order(machine)
    if order is not None:
        assert check_wheeler_axioms(labeled_graph(machine), order.states)
        assert sorted(order.states, key=repr) == sorted(machine.states(), key=repr)


@settings(deadline=None, max_examples=50)
@given(epsilon_machines(max_states=4))
def test_width_one_iff_wheeler_for_input_consistent_machines(machine):
    if is_input_consistent(machine):
        assert (colex_width(machine) == 1) == is_wheeler(machine)
    else:
        assert not is_wheeler(machine)


@settings(deadline=None, max_examples=50)
@given(dfas(alphabet=(0, 1), max_states=4))
def test_wheeler_implies_width_one(dfa):
    if is_wheeler(dfa):
        assert colex_width(dfa) == 1


@settings(deadline=None, max_examples=30)
@given(epsilon_machines(max_states=4))
def test_canonical_form_is_invariant_under_copying(machine):
    if is_wheeler(machine):
        assert wheeler_isomorphic(machine, machine.copy())
        assert wheeler_canonical_form(machine) == wheeler_canonical_form(machine.copy())


# -- Interval power automaton ----------------------------------------------


def wheeler_order_for(graph):
    from sofic.automata.wheeler import wheeler_order_of_graph

    return wheeler_order_of_graph(labeled_graph_of(graph))


def synchronization_orders(graph):
    return (
        markov_order_from_graph(graph),
        cryptic_order_from_graph(graph),
        reset_threshold_from_graph(graph),
    )


def force_subset_power_automaton(monkeypatch):
    """Make :func:`power_automaton` take the unrestricted subset branch."""
    monkeypatch.setattr(
        "sofic.generators.synchronization.wheeler_order_of_graph",
        lambda _graph, **_kwargs: None,
    )


@pytest.mark.parametrize("machine", [golden_mean(), golden_mean_markov(), wheeler_infinite_order_process()])
def test_interval_and_subset_power_automata_agree(machine):
    graph = graph_from_epsilon_machine(machine)
    order = wheeler_order_for(graph)
    assert order is not None
    interval = _interval_power_automaton(graph, order)
    subset = _subset_power_automaton(graph)
    assert interval.start == subset.start
    assert interval.transitions == subset.transitions


@pytest.mark.parametrize("machine", [golden_mean(), golden_mean_markov(), wheeler_infinite_order_process()])
def test_synchronization_orders_are_unchanged_by_the_interval_fast_path(machine, monkeypatch):
    graph = graph_from_epsilon_machine(machine)
    assert wheeler_order_for(graph) is not None
    with_intervals = synchronization_orders(graph)
    force_subset_power_automaton(monkeypatch)
    assert synchronization_orders(graph) == with_intervals


@settings(deadline=None, max_examples=40)
@given(epsilon_machines(max_states=4))
def test_power_automaton_paths_agree_on_every_wheeler_machine(machine):
    graph = graph_from_epsilon_machine(machine)
    order = wheeler_order_for(graph)
    if order is None:
        return
    interval = _interval_power_automaton(graph, order)
    subset = _subset_power_automaton(graph)
    assert interval.start == subset.start
    assert interval.transitions == subset.transitions


def test_interval_power_automaton_is_polynomially_bounded():
    machine = wheeler_infinite_order_process()
    graph = graph_from_epsilon_machine(machine)
    size = len(graph.states)
    pa = _subset_power_automaton(graph)
    assert len(pa.transitions) <= size * (size + 1) // 2


# -- Minimization and determinization --------------------------------------


def test_minimum_wdfa_collapses_a_de_bruijn_presentation():
    dfa = sigma_star_dfa(2)
    assert is_wheeler(dfa)
    minimal = minimum_wdfa(dfa)
    assert len(list(minimal.states())) == 3
    assert is_wheeler(minimal)
    assert equivalent(dfa, minimal, frozenset("ab"))
    # Already minimal, so minimizing again is a no-op.
    assert len(list(minimum_wdfa(minimal).states())) == 3


def test_minimum_wdfa_rejects_non_wheeler_input():
    dfa = DFA(input_alphabet=BINARY, initial_states=frozenset({"A"}), accepting_states=frozenset({"A"}))
    for state in ("A", "B"):
        dfa.graph.add_state(state)
    dfa.add_transition("A", "A", 0)
    dfa.add_transition("A", "B", 1)
    dfa.add_transition("B", "A", 1)
    with pytest.raises(WheelerError):
        minimum_wdfa(dfa)


def test_wnfa_to_wdfa_stays_within_the_interval_bound():
    nfa = substring_wnfa("abra")
    assert is_wheeler(nfa)
    dfa = wnfa_to_wdfa(nfa)
    states, arity = len(list(nfa.states())), len(nfa.input_alphabet)
    assert len(list(dfa.states())) <= 2 * states - 1 - arity
    assert is_wheeler(dfa)
    assert equivalent(nfa, dfa, nfa.input_alphabet)


def test_wheeler_state_index_is_in_wheeler_order():
    index = wheeler_state_index(golden_mean())
    assert tuple(index.states) == ("A", "B")


# -- Succinct index --------------------------------------------------------


def test_index_membership_agrees_with_the_factor_language():
    shift = ShiftOfFiniteType.from_forbidden_words({(1, 1)}, BINARY)
    index = WheelerIndex.from_model(shift)
    for length in range(1, 8):
        expected = set(shift.factor_language(length))
        for word in itertools.product((0, 1), repeat=length):
            assert index.contains(word) == (word in expected)


def test_index_membership_agrees_with_dfa_recognition():
    dfa = sigma_star_dfa(1)
    dfa.accepting_states = frozenset({"a"})
    index = WheelerIndex.from_model(dfa)
    for length in range(5):
        for word in itertools.product("ab", repeat=length):
            assert index.contains(word) == dfa.recognizes(word)


def test_index_finds_substrings():
    text = "abracadabra"
    index = WheelerIndex.from_model(substring_wnfa(text))
    substrings = {text[i:j] for i in range(len(text) + 1) for j in range(i, len(text) + 1)}
    for length in range(4):
        for word in itertools.product("abcdrx", repeat=length):
            assert index.contains(word) == ("".join(word) in substrings)


def test_index_ranks_and_unranks_in_colex_order():
    shift = ShiftOfFiniteType.from_forbidden_words({(1, 1)}, BINARY)
    index = WheelerIndex.from_model(shift)
    for length in range(1, 8):
        words = sorted(shift.factor_language(length))
        assert index.count_words(length) == len(words)
        assert list(index.words_of_length(length)) == sorted(words, key=lambda w: tuple(reversed(w)))
        for word in words:
            assert index.unrank_word(index.rank_word(word), length) == word


def test_index_word_counts_follow_the_golden_mean_fibonacci():
    index = WheelerIndex.from_model(shift_from_edges(GOLDEN_MEAN_SHIFT))
    assert [index.count_words(length) for length in range(1, 9)] == [2, 3, 5, 8, 13, 21, 34, 55]


def test_index_rejects_out_of_range_ranks_and_unknown_words():
    index = WheelerIndex.from_model(shift_from_edges(GOLDEN_MEAN_SHIFT))
    with pytest.raises(IndexError):
        index.unrank_word(index.count_words(3), 3)
    with pytest.raises(ValueError):
        index.rank_word((1, 1))


def test_index_samples_only_language_words():
    index = WheelerIndex.from_model(shift_from_edges(GOLDEN_MEAN_SHIFT))
    rng = random.Random(0)
    for _draw in range(25):
        assert index.contains(index.sample_word(6, rng))


def test_index_forward_search_returns_reachable_states():
    index = WheelerIndex.from_model(shift_from_edges(GOLDEN_MEAN_SHIFT))
    assert index.states_reached((1,)) == (1,)
    assert index.forward_search((1, 1)) is None
    assert index.count_states(()) == 2


# -- Shifts ----------------------------------------------------------------


def test_wheeler_cover_of_the_golden_mean_shift():
    shift = shift_from_edges(GOLDEN_MEAN_SHIFT)
    assert wheeler_order_of_shift(shift) == 0
    cover = wheeler_cover(shift)
    assert len(list(cover.states())) == 2
    assert cover.is_wheeler()


def test_wheeler_cover_adds_memory_for_the_full_shift():
    full_shift = shift_from_edges(((0, 0, 0), (0, 1, 0)))
    assert wheeler_order_of_shift(full_shift) == 1
    cover = wheeler_cover(full_shift)
    assert len(list(cover.states())) == 2
    assert cover.is_wheeler()


def test_the_even_shift_has_no_wheeler_presentation():
    # Wheeler languages are star-free; the even shift counts 1s modulo two.
    even = shift_from_edges(EVEN_SHIFT)
    assert wheeler_order_of_shift(even) is None
    assert not is_wheeler_shift(even)
    with pytest.raises(WheelerError):
        wheeler_cover(even)


@pytest.mark.parametrize("edges", [GOLDEN_MEAN_SHIFT, ((0, 0, 0), (0, 1, 0))])
def test_wheeler_cover_preserves_the_factor_language(edges):
    shift = shift_from_edges(edges)
    cover = wheeler_cover(shift)
    for length in range(1, 8):
        assert sorted(cover.factor_language(length)) == sorted(shift.factor_language(length))


def test_wheeler_cover_right_resolves_a_left_resolving_presentation():
    reversed_golden_mean = shift_from_edges(tuple((t, a, s) for s, a, t in GOLDEN_MEAN_SHIFT))
    cover = wheeler_cover(reversed_golden_mean)
    assert cover.is_wheeler()


def test_higher_block_presentation_is_input_consistent():
    even = shift_from_edges(EVEN_SHIFT)
    for order in (1, 2, 3):
        assert is_input_consistent(higher_block_presentation(even, order))


# -- Stochastic layer ------------------------------------------------------


@pytest.mark.parametrize("machine", [golden_mean(), golden_mean_markov(), wheeler_infinite_order_process()])
def test_wheeler_presentation_is_the_machine_itself_when_wheeler(machine):
    assert wheeler_presentation(machine) is machine
    assert wheeler_statistical_complexity(machine) == pytest.approx(machine.statistical_complexity())


@settings(deadline=None, max_examples=30)
@given(epsilon_machines(max_states=4))
def test_wheeler_complexity_is_never_below_statistical_complexity(machine):
    try:
        found = wheeler_statistical_complexity(machine)
    except WheelerError:
        return
    assert found >= machine.statistical_complexity() - 1e-9


def stationary_symbol_entropy(presentation):
    """``H[X_0]`` read off an input-consistent presentation's incoming labels."""
    index = presentation.reindex()
    stationary = np.asarray(presentation.stationary_distribution(), dtype=float)
    in_labels = labeled_graph(presentation).in_labels()
    mass = defaultdict(float)
    for state in presentation.states():
        for symbol in in_labels[state]:
            mass[symbol] += stationary[index.index(state)]
    probabilities = np.array(list(mass.values()))
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


@settings(deadline=None, max_examples=30)
@given(epsilon_machines(max_states=4))
def test_wheeler_complexity_is_never_below_the_single_symbol_entropy(machine):
    # A Wheeler presentation is input consistent, so its state determines the
    # symbol that entered it and C_W >= H[X_0] on top of C_W >= C_mu.
    if not machine.is_irreducible():
        return
    try:
        presentation = wheeler_presentation(machine)
    except WheelerError:
        return
    found = wheeler_statistical_complexity(machine)
    assert found >= stationary_symbol_entropy(presentation) - 1e-9


def test_a_fair_coin_pays_a_whole_bit_for_sortability():
    coin = fair_coin()
    assert coin.statistical_complexity() == pytest.approx(0.0)
    assert wheeler_statistical_complexity(coin) == pytest.approx(1.0)


def test_every_finite_order_process_has_a_wheeler_presentation():
    # Finite Markov order R means the length-R words pin down the causal state,
    # so the de Bruijn presentation exists and its states sort by construction.
    # Machines whose own causal states are not sortable pay for it in C_W.
    checked = 0
    for states in (2, 3):
        for machine in iter_topological_epsilon_machines(2, states):
            if is_wheeler(machine) or machine.markov_order() == float("inf"):
                continue
            presentation = wheeler_presentation(machine)
            assert is_wheeler(presentation)
            assert wheeler_statistical_complexity(machine) > machine.statistical_complexity()
            checked += 1
    assert checked > 0


def test_debruijn_presentation_needs_the_markov_order():
    machine = next(m for m in iter_topological_epsilon_machines(2, 3) if not is_wheeler(m) and m.markov_order() == 3)
    assert debruijn_presentation(machine, 1) is None
    assert is_wheeler(debruijn_presentation(machine, 3))


@pytest.mark.parametrize("machine", [even_process(), nemo_process(), butterfly_process()])
def test_non_star_free_processes_have_no_wheeler_presentation(machine):
    with pytest.raises(WheelerError):
        wheeler_presentation(machine)


def test_colex_cdf_is_the_cumulative_stationary_distribution():
    machine = golden_mean()
    states, cumulative = colex_cdf(machine)
    assert states == ("A", "B")
    assert cumulative[0] == pytest.approx(2 / 3)
    assert cumulative[-1] == pytest.approx(1.0)
    assert cylinder_measure(machine, 0, 1) == pytest.approx(1.0)
    assert cylinder_measure(machine, 1, 1) == pytest.approx(1 / 3)


def test_word_cylinder_measure_weighs_the_reachable_interval():
    machine = golden_mean()
    assert word_cylinder_measure(machine, (0,)) == pytest.approx(2 / 3)
    assert word_cylinder_measure(machine, (1,)) == pytest.approx(1 / 3)
    assert word_cylinder_measure(machine, (1, 1)) == 0.0


def test_colex_cdf_rejects_non_wheeler_machines():
    with pytest.raises(WheelerError):
        colex_cdf(even_process())


# Smallest process whose ε-machine is Wheeler but whose time reversal's is not.
# Found by exhaustive search over binary topological ε-machines: forward and
# reverse Wheelerness first disagree at three states (reverse-only) and at four
# states (forward-only), so neither implies the other.  Edges are
# ``(source, target, symbol)``.
_REVERSAL_WITNESS_FORWARD = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 2, 1), (2, 3, 0), (3, 1, 1)]
_REVERSAL_WITNESS_REVERSE = [(0, 1, 1), (0, 2, 0), (1, 0, 0), (1, 3, 1), (2, 2, 0), (2, 3, 1), (3, 0, 0)]


def _presentation(edges) -> DFA:
    states = {source for source, _, _ in edges} | {target for _, target, _ in edges}
    dfa = DFA(
        input_alphabet=frozenset({0, 1}),
        initial_states=frozenset({min(states)}),
        accepting_states=frozenset(states),
    )
    for state in sorted(states):
        dfa.graph.add_state(state)
    for source, target, symbol in edges:
        dfa.add_transition(source, target, symbol)
    return dfa


def _factor_words(edges, length):
    states = {source for source, _, _ in edges} | {target for _, target, _ in edges}
    reached = {(state, ()) for state in states}
    for _ in range(length):
        reached = {
            (target, word + (symbol,)) for state, word in reached for source, target, symbol in edges if source == state
        }
    return {word for _, word in reached}


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6])
def test_the_reversal_witness_presentations_are_genuine_reverses(length):
    forward = _factor_words(_REVERSAL_WITNESS_FORWARD, length)
    backward = _factor_words(_REVERSAL_WITNESS_REVERSE, length)
    assert backward == {word[::-1] for word in forward}


def test_wheelerness_is_not_preserved_by_time_reversal():
    assert is_wheeler(_presentation(_REVERSAL_WITNESS_FORWARD))
    assert not is_wheeler(_presentation(_REVERSAL_WITNESS_REVERSE))


def test_the_reversal_witness_fails_sortability_not_input_consistency():
    """Both presentations are input consistent; only the forward one sorts."""
    assert is_input_consistent(_presentation(_REVERSAL_WITNESS_FORWARD))
    assert is_input_consistent(_presentation(_REVERSAL_WITNESS_REVERSE))
    assert colex_width(_presentation(_REVERSAL_WITNESS_REVERSE)) > 1
