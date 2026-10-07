"""Property-based and metamorphic tests for symbolic shifts.

Every oracle here touches only the raw labeled graph (networkx reachability,
explicit path and word enumeration), never sofic's own trimming, subset
construction, or cover code.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable
from itertools import product
from typing import Any

import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sofic.exceptions import SoficValidationError
from sofic.graph import ATTR_EMISSION, ATTR_PROB, ATTR_SYMBOL
from sofic.shifts import (
    LeftFischerCover,
    LeftKriegerCover,
    RightFischerCover,
    RightKriegerCover,
    ShiftOfFiniteType,
    SlidingBlockCode,
    SoficDyckShift,
    SoficShift,
    TopologicalMarkovChain,
    higher_block_presentation,
    right_resolve,
)
from sofic.testing import sfts, sofic_shifts
from tests import oracles

ALPHABET = ("0", "1")
TOL = 1e-9
_FEW = settings(max_examples=max(10, settings.default.max_examples // 4))


# --------------------------------------------------------------------------- graph oracles


def _edges(shift: Any) -> list[tuple[Hashable, Any, Hashable]]:
    return [(t.source, t.data.get(ATTR_SYMBOL), t.target) for t in shift.transitions()]


def _digraph(shift: Any) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(shift.states())
    graph.add_edges_from((source, target) for source, _symbol, target in _edges(shift))
    return graph


def _on_cycle(graph: nx.DiGraph) -> set[Hashable]:
    cyclic: set[Hashable] = set()
    for component in nx.strongly_connected_components(graph):
        node = next(iter(component))
        if len(component) > 1 or graph.has_edge(node, node):
            cyclic |= component
    return cyclic


def _bi_infinite_states(shift: Any) -> tuple[set[Hashable], set[Hashable]]:
    """``(states with an infinite past, states with an infinite future)``."""
    graph = _digraph(shift)
    cyclic = _on_cycle(graph)
    past = set(cyclic).union(*(nx.descendants(graph, node) for node in cyclic))
    future = set(cyclic).union(*(nx.ancestors(graph, node) for node in cyclic))
    return past, future


def brute_language(shift: Any, n: int) -> frozenset[tuple[Any, ...]]:
    """Labels of length-``n`` paths that extend to bi-infinite paths."""
    past, future = _bi_infinite_states(shift)
    if not past & future:
        return frozenset()
    if n == 0:
        return frozenset({()})
    out: dict[Hashable, list[tuple[Any, Hashable]]] = defaultdict(list)
    for source, symbol, target in _edges(shift):
        out[source].append((symbol, target))
    words: set[tuple[Any, ...]] = set()
    frontier = [(state, ()) for state in past]
    for _ in range(n):
        frontier = [(target, word + (symbol,)) for state, word in frontier for symbol, target in out[state]]
    words.update(word for state, word in frontier if state in future)
    return frozenset(words)


def language(shift: Any, n: int) -> frozenset[tuple[Any, ...]]:
    words = list(shift.factor_language(n))
    assert len(words) == len(set(words)), "factor_language yielded duplicates"
    return frozenset(words)


def is_right_resolving(shift: Any) -> bool:
    seen = set()
    for source, symbol, _target in _edges(shift):
        if (source, symbol) in seen:
            return False
        seen.add((source, symbol))
    return True


def is_left_resolving(shift: Any) -> bool:
    seen = set()
    for _source, symbol, target in _edges(shift):
        if (target, symbol) in seen:
            return False
        seen.add((target, symbol))
    return True


def log2_spectral_radius(shift: Any) -> float:
    states = list(shift.states())
    if not states:
        return 0.0
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for source, _symbol, target in _edges(shift):
        matrix[index[source], index[target]] += 1.0
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    return math.log2(radius) if radius > TOL else 0.0


def is_irreducible_presentation(shift: Any) -> bool:
    """Whether the bi-infinite part of the presentation is one strongly connected component."""
    past, future = _bi_infinite_states(shift)
    core = past & future
    return bool(core) and nx.is_strongly_connected(_digraph(shift).subgraph(core))


def relabel(shift: SoficShift, state_map: dict, symbol_map: dict) -> SoficShift:
    result = SoficShift(symbol_alphabet=frozenset(symbol_map[a] for a in shift.symbol_alphabet))
    for state in shift.states():
        result.graph.add_state(state_map[state])
    for source, symbol, target in _edges(shift):
        result.add_transition(state_map[source], state_map[target], symbol_map[symbol])
    return result


def follower_signature(shift: Any, state: Hashable, depth: int) -> frozenset[tuple[Any, ...]]:
    out: dict[Hashable, list[tuple[Any, Hashable]]] = defaultdict(list)
    for source, symbol, target in _edges(shift):
        out[source].append((symbol, target))
    words: set[tuple[Any, ...]] = set()
    frontier = [(state, ())]
    for _ in range(depth):
        frontier = [(target, word + (symbol,)) for current, word in frontier for symbol, target in out[current]]
        words.update(word for _state, word in frontier)
    return frozenset(words)


irreducible_shifts = sofic_shifts(max_states=4).filter(is_irreducible_presentation)


# --------------------------------------------------------------------------- factor language


@given(sofic_shifts(max_states=4), st.integers(0, 5))
def test_factor_language_matches_bi_extendable_paths(shift, n):
    assert language(shift, n) == brute_language(shift, n)


@given(sofic_shifts(max_states=4))
def test_factor_language_is_factorial_and_extendable(shift):
    for n in range(1, 5):
        longer, shorter = language(shift, n + 1), language(shift, n)
        assert {word[1:] for word in longer} == shorter
        assert {word[:-1] for word in longer} == shorter


# --------------------------------------------------------------------------- SFTs from forbidden words


def _contains_forbidden(word: tuple[Any, ...], forbidden: frozenset[tuple[Any, ...]]) -> bool:
    return any(word[i : i + len(f)] == f for f in forbidden for i in range(len(word) - len(f) + 1))


def sft_language_oracle(forbidden: frozenset, alphabet: tuple, max_n: int) -> dict[int, frozenset]:
    """Words of length ``<= max_n`` that sit inside a long ``F``-free word with ``K`` symbols on each side.

    With memory ``M = max|f| - 1``, ``K = |A|^M + M`` symbols on each side force
    a repeated ``M``-block, so the extension pumps to a bi-infinite point.
    """
    if not forbidden:
        return {n: frozenset(product(alphabet, repeat=n)) for n in range(max_n + 1)}
    memory = max(len(word) for word in forbidden) - 1
    pad = len(alphabet) ** memory + memory
    allowed = [()]
    for _ in range(max_n + 2 * pad):
        allowed = [
            word + (a,)
            for word in allowed
            for a in alphabet
            if not _contains_forbidden((word + (a,))[-memory - 1 :], forbidden)
        ]
    return (
        {n: frozenset(word[pad : pad + n] for word in allowed) for n in range(max_n + 1)}
        if allowed
        else {n: frozenset() for n in range(max_n + 1)}
    )


@given(sfts(max_word_length=3))
def test_sft_language_equals_forbidden_word_oracle(sft):
    forbidden = frozenset(sft.forbidden_words())
    expected = sft_language_oracle(forbidden, ALPHABET, 4)
    for n in range(1, 5):
        words = language(sft, n)
        assert not any(_contains_forbidden(word, forbidden) for word in words)
        assert words == expected[n]


@given(sfts(max_word_length=3))
def test_sft_inferred_forbidden_words_regenerate_the_shift(sft):
    forbidden = sft.forbidden_words(max_length=3, minimal=True)
    rebuilt = ShiftOfFiniteType.from_forbidden_words(set(forbidden), frozenset(ALPHABET))
    for n in range(5):
        assert language(rebuilt, n) == language(sft, n)


# --------------------------------------------------------------------------- entropy across presentations


def _presentations(shift: SoficShift) -> dict[str, SoficShift]:
    found = {
        "right_resolve": right_resolve(shift),
        "right_krieger": RightKriegerCover.from_presentation(shift),
        "left_krieger": LeftKriegerCover.from_presentation(shift),
        "higher_block_1": higher_block_presentation(shift, 1),
        "higher_block_2": higher_block_presentation(shift, 2),
    }
    try:
        found["right_fischer"] = RightFischerCover.from_presentation(shift)
        found["left_fischer"] = LeftFischerCover.from_presentation(shift)
    except SoficValidationError:
        pass
    return found


@given(sofic_shifts(max_states=3))
def test_topological_entropy_is_presentation_invariant(shift):
    h = shift.topological_entropy()
    for name, presentation in _presentations(shift).items():
        assert presentation.topological_entropy() == pytest.approx(h, abs=1e-7), name
        for n in range(4):
            assert language(presentation, n) == language(shift, n), name


@given(sofic_shifts(max_states=3))
def test_topological_entropy_is_log_spectral_radius_of_right_resolving_cover(shift):
    h = shift.topological_entropy()
    for cover in (RightKriegerCover.from_presentation(shift), right_resolve(shift)):
        assert is_right_resolving(cover)
        assert log2_spectral_radius(cover) == pytest.approx(h, abs=1e-7)


@given(sofic_shifts(max_states=3))
def test_topological_entropy_is_below_every_word_count_rate(shift):
    """``|L_{m+n}| <= |L_m| |L_n|``, so ``h = inf_n log|L_n| / n``."""
    h = shift.topological_entropy()
    for n in range(1, 8):
        assert n * h <= math.log2(len(brute_language(shift, n))) + 1e-7


def test_right_fischer_cover_rejects_reducible_shift_with_one_terminal_component():
    shift = SoficShift(symbol_alphabet=frozenset(ALPHABET))
    shift.add_transition("A", "A", "0")
    shift.add_transition("A", "B", "0")
    shift.add_transition("A", "B", "1")
    shift.add_transition("B", "B", "1")
    with pytest.raises(SoficValidationError, match="reducible"):
        RightFischerCover.from_presentation(shift)
    with pytest.raises(SoficValidationError, match="reducible"):
        LeftFischerCover.from_presentation(shift)
    assert RightKriegerCover.from_presentation(shift).topological_entropy() == pytest.approx(0.0)


def _component_shifts(shift: SoficShift) -> list[SoficShift]:
    graph = _digraph(shift)
    components = []
    for component in nx.strongly_connected_components(graph):
        node = next(iter(component))
        if len(component) == 1 and not graph.has_edge(node, node):
            continue
        sub = SoficShift(symbol_alphabet=shift.symbol_alphabet)
        for source, symbol, target in _edges(shift):
            if source in component and target in component:
                sub.add_transition(source, target, symbol)
        components.append(sub)
    return components


@given(sofic_shifts(max_states=3))
def test_right_fischer_cover_exists_iff_one_component_presents_the_shift(shift):
    """An irreducible shift is presented by one irreducible component of any presentation.

    The left half of a left-transitive point contains every word and eventually
    stays inside one component (Lind & Marcus, ch. 3-4).
    """
    try:
        RightFischerCover.from_presentation(shift)
        irreducible = True
    except SoficValidationError:
        irreducible = False
    lengths = range(1, 7)
    full = {n: brute_language(shift, n) for n in lengths}
    presenting = [all(brute_language(c, n) == full[n] for n in lengths) for c in _component_shifts(shift)]
    assert any(presenting) == irreducible


# --------------------------------------------------------------------------- Fischer and Krieger covers


@given(irreducible_shifts)
def test_right_fischer_cover_is_minimal_irreducible_right_resolving(shift):
    cover = RightFischerCover.from_presentation(shift)
    assert isinstance(cover, RightFischerCover)
    assert is_right_resolving(cover)
    assert list(cover.states()) and nx.is_strongly_connected(_digraph(cover))
    for n in range(6):
        assert language(cover, n) == language(shift, n)
    states = list(cover.states())
    depth = len(states)
    signatures = {follower_signature(cover, state, depth) for state in states}
    assert len(signatures) == len(states), "Fischer cover states must be follower-separated"
    for other in (RightKriegerCover.from_presentation(shift), right_resolve(shift)):
        assert len(states) <= len(list(other.trim_transient().states()))


@given(irreducible_shifts)
def test_left_fischer_cover_mirrors_right_fischer_cover_of_reverse(shift):
    left = LeftFischerCover.from_presentation(shift)
    mirrored = RightFischerCover.from_presentation(shift.reverse())
    assert is_left_resolving(left)
    assert len(list(left.states())) == len(list(mirrored.states()))
    for n in range(6):
        assert language(left, n) == language(shift, n)
        assert language(left.reverse(), n) == language(mirrored, n)


@given(sofic_shifts(max_states=4))
def test_krieger_covers_resolve_and_preserve_language(shift):
    right = RightKriegerCover.from_presentation(shift)
    left = LeftKriegerCover.from_presentation(shift)
    mirrored = RightKriegerCover.from_presentation(shift.reverse())
    assert is_right_resolving(right)
    assert is_left_resolving(left)
    assert len(list(left.states())) == len(list(mirrored.states()))
    for n in range(6):
        assert language(right, n) == language(shift, n)
        assert language(left, n) == language(shift, n)
        assert language(left.reverse(), n) == language(mirrored, n)
        assert language(shift.reverse(), n) == {word[::-1] for word in language(shift, n)}


# --------------------------------------------------------------------------- Parry measure


def _unifilar_entropy_rate(hmm: Any) -> float:
    """``sum_i pi_i H(row_i)`` from raw edges; checks ``pi`` is stationary on the way."""
    pi = {state: float(hmm.initial_distribution.get(state, 0.0)) for state in hmm.states()}
    rows: dict[Hashable, list[float]] = defaultdict(list)
    flow: dict[Hashable, float] = defaultdict(float)
    seen = set()
    for transition in hmm.transitions():
        prob = float(transition.data[ATTR_PROB])
        key = (transition.source, transition.data[ATTR_EMISSION])
        assert key not in seen, "Parry measure must be unifilar"
        seen.add(key)
        rows[transition.source].append(prob)
        flow[transition.target] += pi[transition.source] * prob
    for state in pi:
        assert sum(rows[state]) == pytest.approx(1.0, abs=1e-9)
        assert flow[state] == pytest.approx(pi[state], abs=1e-9)
    return -sum(pi[s] * sum(p * math.log2(p) for p in rows[s] if p > 0) for s in pi)


@st.composite
def irreducible_adjacency(draw: Any) -> np.ndarray:
    n = draw(st.integers(1, 4))
    matrix = np.array(draw(st.lists(st.integers(0, 2), min_size=n * n, max_size=n * n)), dtype=float).reshape(n, n)
    graph = nx.DiGraph([(i, j) for i in range(n) for j in range(n) if matrix[i, j] > 0])
    graph.add_nodes_from(range(n))
    assume(matrix.any() and nx.is_strongly_connected(graph))
    return matrix


@given(irreducible_adjacency())
def test_parry_measure_entropy_rate_equals_h_top_for_tmc(matrix):
    tmc = TopologicalMarkovChain.from_adjacency(matrix)
    expected = math.log2(float(np.max(np.abs(np.linalg.eigvals(matrix)))))
    assert tmc.topological_entropy() == pytest.approx(expected, abs=1e-9)
    assert _unifilar_entropy_rate(tmc.parry_measure()) == pytest.approx(expected, abs=1e-7)


@_FEW
@given(sfts(max_word_length=3).filter(is_irreducible_presentation))
def test_parry_measure_entropy_rate_equals_h_top_for_sft(sft):
    shift = SoficShift(graph=sft.graph.copy(), symbol_alphabet=sft.symbol_alphabet)
    parry = shift.parry_measure()
    parry.validate_stochastic()
    assert _unifilar_entropy_rate(parry) == pytest.approx(sft.topological_entropy(), abs=1e-7)


def test_parry_measure_of_unifilar_presentation_drops_transient_states():
    """Golden mean core ``A <-> B`` plus a source ``S`` and a dead end ``D``."""
    shift = SoficShift(symbol_alphabet=frozenset(ALPHABET))
    shift.add_transition("A", "A", "0")
    shift.add_transition("A", "B", "1")
    shift.add_transition("B", "A", "0")
    shift.add_transition("B", "D", "1")
    shift.add_transition("S", "A", "1")
    parry = shift.parry_measure()
    parry.validate_stochastic()
    assert set(parry.states()) == {"A", "B"}
    assert _unifilar_entropy_rate(parry) == pytest.approx(math.log2((1 + math.sqrt(5)) / 2))


@_FEW
@given(irreducible_shifts)
def test_parry_measure_of_sofic_shift_is_maximal_and_supported_on_language(shift):
    parry = shift.parry_measure()
    assert _unifilar_entropy_rate(parry) == pytest.approx(shift.topological_entropy(), abs=1e-7)
    for n in range(1, 4):
        support = {word for word, p in oracles.word_distribution(parry, n, ALPHABET).items() if p > 1e-12}
        assert support == language(shift, n)


@_FEW
@given(irreducible_shifts)
def test_topological_anatomy_splits_h_top(shift):
    anatomy = shift.topological_anatomy()
    assert anatomy["h_top"] == pytest.approx(shift.topological_entropy(), abs=1e-7)
    assert anatomy["h_top"] == pytest.approx(anatomy["b_top"] + anatomy["r_top"], abs=1e-7)
    assert anatomy["b_top"] >= -1e-9
    assert anatomy["r_top"] >= -1e-9


# --------------------------------------------------------------------------- sliding block codes


@st.composite
def block_codes(draw: Any, alphabet: tuple = ALPHABET, outputs: tuple = ("a", "b", "c")) -> SlidingBlockCode:
    memory = draw(st.integers(0, 1))
    anticipation = draw(st.integers(0, 1))
    window = memory + anticipation + 1
    blocks = list(product(alphabet, repeat=window))
    images = draw(st.lists(st.sampled_from(outputs), min_size=len(blocks), max_size=len(blocks)))
    return SlidingBlockCode(
        dict(zip(blocks, images, strict=True)),
        memory=memory,
        anticipation=anticipation,
        input_alphabet=alphabet,
    )


words_over_alphabet = st.lists(st.sampled_from(ALPHABET), max_size=8).map(tuple)


@given(block_codes(), words_over_alphabet)
def test_apply_word_is_pointwise_block_map(code, word):
    image = code.apply_word(word)
    assert len(image) == max(0, len(word) - code.window + 1)
    for i, symbol in enumerate(image):
        assert symbol == code.block_map[word[i : i + code.window]]


@given(block_codes(), block_codes(alphabet=("a", "b", "c"), outputs=ALPHABET), words_over_alphabet)
def test_compose_is_function_composition(first, second, word):
    composed = first.compose(second)
    assert composed.window == first.window + second.window - 1
    assert composed.apply_word(word) == second.apply_word(first.apply_word(word))


@given(sofic_shifts(max_states=3), block_codes())
def test_image_language_is_image_of_language(shift, code):
    image = code.apply(shift)
    for n in range(4):
        expected = {code.apply_word(word) for word in language(shift, n + code.window - 1)}
        assert language(image, n) == expected
    assert image.topological_entropy() <= shift.topological_entropy() + 1e-7


def _higher_block_code(window: int) -> SlidingBlockCode:
    return SlidingBlockCode({block: "".join(block) for block in product(ALPHABET, repeat=window)}, memory=window - 1)


@given(sofic_shifts(max_states=3), st.integers(1, 3))
def test_higher_block_conjugacy_preserves_entropy(shift, window):
    image = _higher_block_code(window).apply(shift)
    assert image.topological_entropy() == pytest.approx(shift.topological_entropy(), abs=1e-7)
    for n in range(1, 4):
        assert len(language(image, n)) == len(language(shift, n + window - 1))


# --------------------------------------------------------------------------- Dyck shifts


@st.composite
def dyck_shifts(draw: Any) -> SoficDyckShift:
    n = draw(st.integers(1, 3))
    shift = SoficDyckShift(
        call_alphabet=frozenset({"(", "["}),
        return_alphabet=frozenset({")", "]"}),
        internal_alphabet=frozenset({"i"}),
    )
    for state in range(n):
        shift.graph.add_state(state)
    state = st.integers(0, n - 1)
    calls = [
        shift.add_call_transition(s, t, a)
        for s, t, a in draw(st.lists(st.tuples(state, state, st.sampled_from("([")), min_size=1, max_size=3))
    ]
    returns = [
        shift.add_return_transition(s, t, a)
        for s, t, a in draw(st.lists(st.tuples(state, state, st.sampled_from(")]")), min_size=1, max_size=3))
    ]
    for s, t in draw(st.lists(st.tuples(state, state), max_size=2)):
        shift.add_internal_transition(s, t, "i")
    for call, ret in draw(st.lists(st.tuples(st.sampled_from(calls), st.sampled_from(returns)), max_size=4)):
        shift.add_matched_pair(call, ret)
    shift.validate()
    return shift


@given(dyck_shifts())
def test_dyck_reverse_mirrors_language_and_is_an_involution(shift):
    reversed_shift = shift.reverse()
    reversed_shift.validate()
    twice = reversed_shift.reverse()
    twice.validate()
    assert twice.call_alphabet == shift.call_alphabet
    assert twice.return_alphabet == shift.return_alphabet
    for n in range(6):
        words = language(shift, n)
        assert language(reversed_shift, n) == {word[::-1] for word in words}
        assert language(twice, n) == words


@given(dyck_shifts(), st.lists(st.sampled_from(["(", "[", ")", "]", "i"]), max_size=5).map(tuple))
def test_dyck_admissibility_agrees_with_factor_language(shift, word):
    assert shift.is_admissible_word(word) == (word in language(shift, len(word)))


# --------------------------------------------------------------------------- metamorphic


_LABELS = st.one_of(st.integers(-5, 50), st.text(max_size=3), st.tuples(st.integers(0, 3), st.text(max_size=1)))


@given(sofic_shifts(max_states=4), st.data())
def test_relabeling_states_and_symbols_preserves_entropy_and_language(shift, data):
    states = list(shift.states())
    new_states = data.draw(st.lists(_LABELS, min_size=len(states), max_size=len(states), unique=True))
    new_symbols = data.draw(st.permutations(["x", "y"]))
    symbol_map = dict(zip(ALPHABET, new_symbols, strict=True))
    relabeled = relabel(shift, dict(zip(states, new_states, strict=True)), symbol_map)
    assert relabeled.topological_entropy() == pytest.approx(shift.topological_entropy(), abs=1e-9)
    for n in range(5):
        mapped = {tuple(symbol_map[a] for a in word) for word in language(shift, n)}
        assert language(relabeled, n) == mapped


@given(
    sofic_shifts(max_states=3),
    st.lists(st.tuples(st.booleans(), st.integers(0, 2), st.sampled_from(ALPHABET)), min_size=1, max_size=4),
)
def test_adding_transient_states_changes_nothing(shift, extra):
    """Sources (no in-edges) and sinks (no out-edges) carry no bi-infinite path."""
    states = list(shift.states())
    grown = shift.copy()
    for i, (is_source, target, symbol) in enumerate(extra):
        name = f"transient{i}"
        grown.graph.add_state(name)
        anchor = states[target % len(states)]
        if is_source:
            grown.add_transition(name, anchor, symbol)
        else:
            grown.add_transition(anchor, name, symbol)
    assert grown.topological_entropy() == pytest.approx(shift.topological_entropy(), abs=1e-9)
    assert len(list(RightKriegerCover.from_presentation(grown).states())) == len(
        list(RightKriegerCover.from_presentation(shift).states())
    )
    for n in range(5):
        assert language(grown, n) == language(shift, n)
