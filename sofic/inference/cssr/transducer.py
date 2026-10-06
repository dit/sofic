"""transCSSR: sample-based epsilon-transducer reconstruction.

Generalizes Causal-State Splitting Reconstruction (Shalizi, Shalizi &
Crutchfield, arXiv:cs/0210025) from a single process to an input-output channel,
following the ε-transducer of Barnett & Crutchfield (J. Stat. Phys. 161:2
(2015)) and the transCSSR algorithm (Darmon & Rapp, ``ddarmon/transCSSR``).

Causal states are equivalence classes of joint ``(input, output)`` pasts that
induce the same conditional next-output law ``P(y | history, x)`` for every input
symbol ``x``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Literal

from sofic.exceptions import StochasticValidationError
from sofic.generators.epsilon_transducer import EpsilonTransducer
from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, TransitionGraph
from sofic.inference.cssr.counts import (
    JointHistory,
    JointSuffixCounts,
    StateAggregate,
    _history_aggregate,
    _merge_aggregate,
    _state_aggregate,
)
from sofic.inference.cssr.process import _recurrent_states, suggest_max_history
from sofic.inference.cssr.significance import TableTest, _aggregate_score, aggregates_differ


def _observed(counts: JointSuffixCounts, history: JointHistory) -> int:
    return sum(sum(counter.values()) for counter in counts.next_counts.get(history, {}).values())


def _homogenize(
    counts: JointSuffixCounts,
    *,
    max_history: int,
    alpha: float,
    test: TableTest,
    min_count: int,
) -> list[set[JointHistory]]:
    """transCSSR homogenization: grow joint suffixes one ``(input, output)`` pair into the past.

    As in flat CSSR, a child ``p h`` of suffix ``h`` stays in its parent's state
    unless its conditional output law differs; histories seen fewer than
    ``min_count`` times stay with their parent.
    """
    in_alpha, out_alpha = counts.input_alphabet, counts.output_alphabet
    states: list[set[JointHistory]] = [{()}]
    aggregates: list[StateAggregate] = [_history_aggregate(counts, ())]
    pairs = _pairs_from(counts)

    def differ(left: StateAggregate, right: StateAggregate) -> bool:
        return aggregates_differ(
            left, right, input_alphabet=in_alpha, output_alphabet=out_alpha, alpha=alpha, test=test
        )

    for length in range(max_history):
        for parent_id in range(len(states)):
            for history in sorted((h for h in states[parent_id] if len(h) == length), key=repr):
                for pair in pairs:
                    child = (pair, *history)
                    if _observed(counts, child) == 0:
                        continue
                    child_agg = _history_aggregate(counts, child)
                    target = parent_id
                    if _observed(counts, child) >= min_count and differ(aggregates[parent_id], child_agg):
                        best_id, best_score = None, float("inf")
                        for candidate_id, candidate_agg in enumerate(aggregates):
                            if candidate_id == parent_id or differ(candidate_agg, child_agg):
                                continue
                            score = _aggregate_score(
                                candidate_agg, child_agg, input_alphabet=in_alpha, output_alphabet=out_alpha
                            )
                            if score < best_score:
                                best_id, best_score = candidate_id, score
                        if best_id is None:
                            states.append(set())
                            aggregates.append({})
                            best_id = len(states) - 1
                        target = best_id
                    states[target].add(child)
                    _merge_aggregate(aggregates[target], child_agg)
    return states


def _pairs_from(counts: JointSuffixCounts) -> list[tuple[Any, Any]]:
    return [(x, y) for x in counts.input_alphabet for y in counts.output_alphabet]


def _history_emits(counts: JointSuffixCounts, history: JointHistory, pair: tuple[Any, Any]) -> bool:
    by_input = counts.next_counts.get(history)
    if by_input is None:
        return False
    counter = by_input.get(pair[0])
    return bool(counter) and counter.get(pair[1], 0) > 0


def _edges(
    states: list[set[JointHistory]],
    counts: JointSuffixCounts,
    alive: set[int],
    *,
    max_history: int,
    alpha: float,
    test: TableTest,
) -> dict[int, dict[tuple[Any, Any], dict[int, set[JointHistory]]]]:
    """Successor states by ``(input, output)`` pair, with the histories that lead there.

    Shorter histories move to their one-pair extension; length-``max_history`` histories
    drop their oldest pair, and the length-``max_history + 1`` history is re-tested against
    the truncated history's state (see
    :func:`sofic.inference.cssr.process._suffix_edges`).
    """
    in_alpha, out_alpha = counts.input_alphabet, counts.output_alphabet
    history_to_state = {h: index for index in alive for h in states[index]}
    aggregates = {index: _state_aggregate(counts, states[index]) for index in alive}
    edges: dict[int, dict[tuple[Any, Any], dict[int, set[JointHistory]]]] = {}
    for index in alive:
        by_pair: dict[tuple[Any, Any], dict[int, set[JointHistory]]] = defaultdict(lambda: defaultdict(set))
        for history in states[index]:
            for pair in _pairs_from(counts):
                if not _history_emits(counts, history, pair):
                    continue
                extended = (*history, pair)
                if len(extended) <= max_history:
                    target = history_to_state.get(extended)
                else:
                    target = history_to_state.get(extended[1:])
                    extended_agg = _history_aggregate(counts, extended)
                    if extended_agg and (
                        target is None
                        or aggregates_differ(
                            aggregates[target],
                            extended_agg,
                            input_alphabet=in_alpha,
                            output_alphabet=out_alpha,
                            alpha=alpha,
                            test=test,
                        )
                    ):
                        best_score = float("inf")
                        for candidate in sorted(alive):
                            if aggregates_differ(
                                aggregates[candidate],
                                extended_agg,
                                input_alphabet=in_alpha,
                                output_alphabet=out_alpha,
                                alpha=alpha,
                                test=test,
                            ):
                                continue
                            score = _aggregate_score(
                                aggregates[candidate], extended_agg, input_alphabet=in_alpha, output_alphabet=out_alpha
                            )
                            if score < best_score:
                                target, best_score = candidate, score
                if target is not None:
                    by_pair[pair][target].add(history)
        edges[index] = by_pair
    return edges


def _determinize(
    states: list[set[JointHistory]],
    counts: JointSuffixCounts,
    alive: set[int],
    *,
    max_history: int,
    alpha: float,
    test: TableTest,
) -> tuple[list[set[JointHistory]], set[int]]:
    """Split alive states until each ``(input, output)`` pair has one alive successor."""
    states = [set(h) for h in states]
    alive = set(alive)
    while True:
        edges = _edges(states, counts, alive, max_history=max_history, alpha=alpha, test=test)
        split = next(
            (
                (index, pair)
                for index in sorted(alive)
                for pair in sorted(edges[index], key=repr)
                if len(edges[index][pair]) > 1
            ),
            None,
        )
        if split is None:
            return states, alive
        index, pair = split
        groups = sorted(edges[index][pair].values(), key=lambda g: (-len(g), sorted(map(repr, g))))
        for group in groups[1:]:
            states[index] -= group
            states.append(set(group))
            alive.add(len(states) - 1)


def _build_transducer(
    states: list[set[JointHistory]],
    counts: JointSuffixCounts,
    alive: set[int],
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    max_history: int,
    alpha: float,
    test: TableTest,
) -> EpsilonTransducer:
    edges = _edges(states, counts, alive, max_history=max_history, alpha=alpha, test=test)
    history_to_state = {h: index for index in alive for h in states[index]}

    visits: Counter[int] = Counter()
    pairs = tuple(zip(inputs, outputs, strict=True))
    for t in range(len(pairs) + 1):
        for hist_len in range(min(t, max_history), -1, -1):
            state = history_to_state.get(pairs[t - hist_len : t])
            if state is not None:
                visits[state] += 1
                break

    classes = _recurrent_states(edges)
    if not classes:
        raise StochasticValidationError("no recurrent inferred states; the sample is too short for this max_history")
    keep = max(classes, key=lambda members: (sum(visits[s] for s in members), -min(members)))

    graph = TransitionGraph()
    labels = {state: f"s{rank}" for rank, state in enumerate(sorted(keep))}
    for state in sorted(keep):
        graph.add_state(labels[state])
    used_inputs: set[Any] = set()
    used_outputs: set[Any] = set()
    for state in sorted(keep):
        longest = max(len(h) for h in states[state])
        aggregate = _state_aggregate(counts, {h for h in states[state] if len(h) == longest})
        for input_symbol in counts.input_alphabet:
            row = [
                (labels[next(iter(edges[state][(input_symbol, output_symbol)]))], output_symbol, float(count))
                for output_symbol, count in sorted(
                    aggregate.get(input_symbol, Counter()).items(), key=lambda i: repr(i[0])
                )
                if count > 0 and edges[state].get((input_symbol, output_symbol))
            ]
            total = sum(weight for _label, _output, weight in row)
            for target_label, output_symbol, weight in row:
                graph.add_transition(
                    labels[state],
                    target_label,
                    **{ATTR_SYMBOL: input_symbol, ATTR_OUTPUT: output_symbol, ATTR_PROB: weight / total},
                )
                used_inputs.add(input_symbol)
                used_outputs.add(output_symbol)

    kept_visits = {state: visits[state] for state in keep if visits[state] > 0}
    total_visits = float(sum(kept_visits.values()))
    initial = (
        {labels[state]: count / total_visits for state, count in kept_visits.items()}
        if total_visits > 0
        else {labels[min(keep)]: 1.0}
    )
    result = EpsilonTransducer(
        input_alphabet=frozenset(used_inputs),
        output_alphabet=frozenset(used_outputs),
        initial_states=frozenset(initial),
        initial_distribution=initial,
        graph=graph,
    )
    result.validate()
    return result


def _default_max_history(n: int, alphabet_size: int, min_count: int) -> int:
    if alphabet_size <= 0:
        return 1
    # The joint (input, output) history space grows as ``alphabet_size ** L``, so
    # keep the default depth modest relative to the single-process CSSR bound.
    return max(1, min(5, n // max(1, alphabet_size * min_count)))


def learn_epsilon_transducer_cssr(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    input_alphabet: Sequence[Any] | None = None,
    output_alphabet: Sequence[Any] | None = None,
    max_history: int | Literal["auto"] | None = None,
    alpha: float = 0.001,
    test: TableTest = "g",
    min_count: int = 5,
    correction: Literal["bonferroni"] | None = None,
) -> EpsilonTransducer:
    """Reconstruct an ε-transducer from paired input/output sequences (transCSSR).

    ``alpha`` is the per-test significance level for the causal-state split
    decision; the transCSSR/CSSR default of ``0.001`` favors fewer, more robust
    states. ``max_history`` bounds the joint-history depth and ``min_count`` the minimum
    occurrences before a history is eligible to seed a new state.

    ``max_history="auto"`` applies :func:`~sofic.inference.cssr.suggest_max_history`
    to the joint ``(input, output)`` sequence. ``test="exact"`` uses the Monte
    Carlo exact G-test when expected counts are small (see
    :func:`~sofic.inference.cssr.morphs_differ`), and
    ``correction="bonferroni"`` divides ``alpha`` by the number of
    (history, input symbol) tests that can split a state.
    """
    xs = tuple(inputs)
    ys = tuple(outputs)
    if len(xs) != len(ys):
        raise ValueError("inputs and outputs must have equal length")
    if len(xs) < 2:
        raise ValueError("sequences must contain at least two symbols")
    joint_alphabet_size = (
        len(set(xs)) * len(set(ys))
        if input_alphabet is None or output_alphabet is None
        else len(tuple(input_alphabet)) * len(tuple(output_alphabet))
    )
    if max_history == "auto":
        max_length = suggest_max_history(list(zip(xs, ys, strict=True)), alpha=alpha)
    else:
        max_length = (
            max_history if max_history is not None else _default_max_history(len(xs), joint_alphabet_size, min_count)
        )
    counts = JointSuffixCounts.from_sequences(
        xs,
        ys,
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
        max_length=max_length + 1,
    )
    if correction == "bonferroni":
        eligible = sum(
            1
            for history in counts.next_counts
            if 0 < len(history) <= max_length and _observed(counts, history) >= max(1, min_count)
        )
        alpha /= max(1, eligible * len(counts.input_alphabet))
    elif correction is not None:
        raise ValueError(f"unknown correction {correction!r}")

    states = _homogenize(counts, max_history=max_length, alpha=alpha, test=test, min_count=min_count)
    everything = set(range(len(states)))
    edges = _edges(states, counts, everything, max_history=max_length, alpha=alpha, test=test)
    alive = set().union(*_recurrent_states(edges)) or everything
    states, alive = _determinize(states, counts, alive, max_history=max_length, alpha=alpha, test=test)
    return _build_transducer(states, counts, alive, xs, ys, max_history=max_length, alpha=alpha, test=test)
