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
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats

from sofic.exceptions import StochasticValidationError
from sofic.generators.epsilon_transducer import EpsilonTransducer
from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, TransitionGraph

JointHistory = tuple[tuple[Any, Any], ...]


@dataclass
class JointSuffixCounts:
    """Empirical counts of joint pasts and following input-conditioned outputs."""

    input_alphabet: tuple[Any, ...]
    output_alphabet: tuple[Any, ...]
    history_counts: Counter[JointHistory] = field(default_factory=Counter)
    #: ``next_counts[history][input]`` is a Counter over following output symbols.
    next_counts: dict[JointHistory, dict[Any, Counter[Any]]] = field(default_factory=dict)

    @classmethod
    def from_sequences(
        cls,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        *,
        input_alphabet: Sequence[Any] | None = None,
        output_alphabet: Sequence[Any] | None = None,
        max_length: int,
    ) -> JointSuffixCounts:
        xs = tuple(inputs)
        ys = tuple(outputs)
        if len(xs) != len(ys):
            raise ValueError("inputs and outputs must have equal length")
        if not xs:
            raise ValueError("sequences must be non-empty")
        in_alpha = tuple(sorted(set(xs), key=repr)) if input_alphabet is None else tuple(input_alphabet)
        out_alpha = tuple(sorted(set(ys), key=repr)) if output_alphabet is None else tuple(output_alphabet)
        counts = cls(input_alphabet=in_alpha, output_alphabet=out_alpha)
        pairs = tuple(zip(xs, ys, strict=True))
        n = len(pairs)
        for t in range(n):
            for length in range(0, min(t, max_length) + 1):
                history = pairs[t - length : t]
                counts.history_counts[history] += 1
                by_input = counts.next_counts.setdefault(history, {})
                by_input.setdefault(xs[t], Counter())[ys[t]] += 1
        return counts

    def output_counts(self, histories: set[JointHistory], input_symbol: Any) -> Counter[Any]:
        observed: Counter[Any] = Counter()
        for history in histories:
            by_input = self.next_counts.get(history)
            if by_input is None:
                continue
            counter = by_input.get(input_symbol)
            if counter is not None:
                observed.update(counter)
        return observed

    def state_morph(self, histories: set[JointHistory], input_symbol: Any) -> dict[Any, float]:
        """Return ``P(output | histories, input_symbol)``."""
        observed = self.output_counts(histories, input_symbol)
        total = sum(observed.values())
        if total == 0:
            return {}
        return {symbol: observed.get(symbol, 0) / total for symbol in self.output_alphabet}


def _output_contingency(left: Counter[Any], right: Counter[Any], alphabet: tuple[Any, ...]) -> np.ndarray | None:
    active = [symbol for symbol in alphabet if left.get(symbol, 0) + right.get(symbol, 0) > 0]
    if not active:
        return None
    table = np.array(
        [[left.get(symbol, 0) for symbol in active], [right.get(symbol, 0) for symbol in active]],
        dtype=float,
    )
    if np.allclose(table[0], table[1]):
        return None
    if table.shape[1] < 2:
        left_total = table[0].sum()
        right_total = table[1].sum()
        if left_total == 0.0 or right_total == 0.0:
            return None
        if np.isclose(table[0, 0] / left_total, table[1, 0] / right_total):
            return None
    return table


#: Aggregated output counts of a state: ``agg[input_symbol]`` is a Counter over outputs.
StateAggregate = dict[Any, Counter[Any]]


def _history_aggregate(counts: JointSuffixCounts, history: JointHistory) -> StateAggregate:
    return {input_symbol: Counter(counter) for input_symbol, counter in counts.next_counts.get(history, {}).items()}


def _merge_aggregate(target: StateAggregate, source: StateAggregate) -> None:
    for input_symbol, counter in source.items():
        target.setdefault(input_symbol, Counter()).update(counter)


def aggregates_differ(
    left: StateAggregate,
    right: StateAggregate,
    *,
    input_alphabet: tuple[Any, ...],
    output_alphabet: tuple[Any, ...],
    alpha: float,
    test: Literal["g", "chi2"] = "g",
) -> bool:
    """Return whether two aggregated morphs differ on ``P(output | ., input)`` for some input."""
    for input_symbol in input_alphabet:
        table = _output_contingency(
            left.get(input_symbol, Counter()),
            right.get(input_symbol, Counter()),
            output_alphabet,
        )
        if table is None:
            continue
        if _table_significant(table, alpha=alpha, test=test):
            return True
    return False


def _aggregate_score(
    left: StateAggregate,
    right: StateAggregate,
    *,
    input_alphabet: tuple[Any, ...],
    output_alphabet: tuple[Any, ...],
) -> float:
    total = 0.0
    for input_symbol in input_alphabet:
        table = _output_contingency(
            left.get(input_symbol, Counter()),
            right.get(input_symbol, Counter()),
            output_alphabet,
        )
        if table is None:
            continue
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                statistic, _p, _dof, _expected = stats.chi2_contingency(table, lambda_="log-likelihood")
            if np.isfinite(statistic):
                total += float(statistic)
        except ValueError:
            continue
    return total


def _table_significant(table: np.ndarray, *, alpha: float, test: Literal["g", "chi2"]) -> bool:
    try:
        if test == "g":
            with np.errstate(invalid="ignore", divide="ignore"):
                statistic, _p, _dof, expected = stats.chi2_contingency(table, lambda_="log-likelihood")
            if not np.isfinite(statistic) or np.any(expected == 0):
                return False
            dof = max(1, table.shape[1] - 1)
            return float(statistic) > float(stats.chi2.ppf(1.0 - alpha, dof))
        statistic, p_value, _dof, expected = stats.chi2_contingency(table)
    except ValueError:
        return False
    if np.any(expected == 0):
        return False
    return float(p_value) < alpha


def _homogenize(
    counts: JointSuffixCounts,
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2"],
    min_count: int,
) -> tuple[dict[int, set[JointHistory]], dict[JointHistory, int]]:
    in_alpha = counts.input_alphabet
    out_alpha = counts.output_alphabet
    states: dict[int, set[JointHistory]] = {0: {()}}
    state_agg: dict[int, StateAggregate] = {0: _history_aggregate(counts, ())}
    history_to_state: dict[JointHistory, int] = {(): 0}
    next_state_id = 1

    for _length in range(Lmax + 1):
        for state_id in sorted(states):
            for history in list(states[state_id]):
                for pair in _observed_pairs(counts, history):
                    child = (*history, pair)
                    if child in history_to_state or counts.history_counts.get(child, 0) == 0:
                        continue
                    child_agg = _history_aggregate(counts, child)
                    if counts.history_counts.get(child, 0) < min_count:
                        # Too rare to split reliably; inherit the parent's causal state.
                        states[state_id].add(child)
                        history_to_state[child] = state_id
                        _merge_aggregate(state_agg[state_id], child_agg)
                        continue
                    if aggregates_differ(
                        state_agg[state_id],
                        child_agg,
                        input_alphabet=in_alpha,
                        output_alphabet=out_alpha,
                        alpha=alpha,
                        test=test,
                    ):
                        best_state: int | None = None
                        best_score = float("inf")
                        for candidate_id, candidate_agg in state_agg.items():
                            if aggregates_differ(
                                candidate_agg,
                                child_agg,
                                input_alphabet=in_alpha,
                                output_alphabet=out_alpha,
                                alpha=alpha,
                                test=test,
                            ):
                                continue
                            score = _aggregate_score(
                                candidate_agg,
                                child_agg,
                                input_alphabet=in_alpha,
                                output_alphabet=out_alpha,
                            )
                            if score < best_score:
                                best_score = score
                                best_state = candidate_id
                        if best_state is None:
                            best_state = next_state_id
                            states[next_state_id] = set()
                            state_agg[next_state_id] = {}
                            next_state_id += 1
                        states[best_state].add(child)
                        history_to_state[child] = best_state
                        _merge_aggregate(state_agg[best_state], child_agg)
                    else:
                        states[state_id].add(child)
                        history_to_state[child] = state_id
                        _merge_aggregate(state_agg[state_id], child_agg)
    return states, history_to_state


def _observed_pairs(counts: JointSuffixCounts, history: JointHistory) -> list[tuple[Any, Any]]:
    by_input = counts.next_counts.get(history)
    if by_input is None:
        return []
    pairs: list[tuple[Any, Any]] = []
    for input_symbol, counter in by_input.items():
        for output_symbol in counter:
            pairs.append((input_symbol, output_symbol))
    return pairs


def _determinize(
    states: dict[int, set[JointHistory]],
    history_to_state: dict[JointHistory, int],
    counts: JointSuffixCounts,
) -> dict[int, set[JointHistory]]:
    current = {state_id: set(histories) for state_id, histories in states.items()}
    next_state_id = (max(current) + 1) if current else 0
    changed = True
    while changed:
        changed = False
        for state_id in sorted(current):
            histories = current[state_id]
            if len(histories) <= 1:
                continue
            for pair in _pairs_from(counts):
                buckets: dict[int, set[JointHistory]] = defaultdict(set)
                for history in histories:
                    if not _history_emits(counts, history, pair):
                        continue
                    child = (*history, pair)
                    target = history_to_state.get(child)
                    if target is None:
                        continue
                    buckets[target].add(history)
                if len(buckets) <= 1:
                    continue
                ordered = sorted(buckets.items(), key=lambda item: (-len(item[1]), repr(min(item[1], key=repr))))
                _keep_target, keep_histories = ordered[0]
                current[state_id] = keep_histories
                for _target, split_histories in ordered[1:]:
                    current[next_state_id] = split_histories
                    for history in split_histories:
                        history_to_state[history] = next_state_id
                    next_state_id += 1
                changed = True
                break
            if changed:
                break
    return current


def _pairs_from(counts: JointSuffixCounts) -> list[tuple[Any, Any]]:
    return [(x, y) for x in counts.input_alphabet for y in counts.output_alphabet]


def _history_emits(counts: JointSuffixCounts, history: JointHistory, pair: tuple[Any, Any]) -> bool:
    by_input = counts.next_counts.get(history)
    if by_input is None:
        return False
    counter = by_input.get(pair[0])
    return bool(counter) and counter.get(pair[1], 0) > 0


def _drop_transient(
    states: dict[int, set[JointHistory]],
    history_to_state: dict[JointHistory, int],
    counts: JointSuffixCounts,
) -> dict[int, set[JointHistory]]:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(states)
    for state_id, histories in states.items():
        for history in histories:
            for pair in _pairs_from(counts):
                if not _history_emits(counts, history, pair):
                    continue
                target = history_to_state.get((*history, pair))
                if target is not None:
                    graph.add_edge(state_id, target)
    if graph.number_of_edges() == 0:
        return states

    recurrent: set[int] = set()
    for component in nx.strongly_connected_components(graph):
        subgraph = graph.subgraph(component)
        has_cycle = subgraph.number_of_edges() > 0 and (
            len(component) > 1 or any(subgraph.has_edge(node, node) for node in component)
        )
        if not has_cycle:
            continue
        if not any(graph.has_edge(v, w) for v in component for w in graph.nodes if w not in component):
            recurrent.update(component)
    if not recurrent:
        return states
    return {state_id: histories for state_id, histories in states.items() if state_id in recurrent}


def _state_visits(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    history_to_state: dict[JointHistory, int],
    *,
    length: int,
) -> Counter[int]:
    visits: Counter[int] = Counter()
    pairs = tuple(zip(inputs, outputs, strict=True))
    for t in range(len(pairs)):
        for hist_len in range(min(t, length), -1, -1):
            history = pairs[t - hist_len : t]
            state = history_to_state.get(history)
            if state is not None:
                visits[state] += 1
                break
    return visits


def _build_transducer(
    states: dict[int, set[JointHistory]],
    counts: JointSuffixCounts,
    history_to_state: dict[JointHistory, int],
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    length: int,
) -> EpsilonTransducer:
    visits = _state_visits(inputs, outputs, history_to_state, length=length)
    if not visits:
        raise StochasticValidationError("no empirical causal-state visits")

    graph = TransitionGraph()
    labels = {state_id: f"s{state_id}" for state_id in states}
    for label in labels.values():
        graph.add_state(label)

    used_inputs: set[Any] = set()
    used_outputs: set[Any] = set()
    for state_id, histories in states.items():
        source = labels[state_id]
        for input_symbol in counts.input_alphabet:
            morph = counts.state_morph(histories, input_symbol)
            if not morph:
                continue
            row: list[tuple[str, Any, float]] = []
            for output_symbol, prob in morph.items():
                if prob <= 0.0:
                    continue
                emitting = [
                    history for history in histories if _history_emits(counts, history, (input_symbol, output_symbol))
                ]
                targets = {history_to_state.get((*history, (input_symbol, output_symbol))) for history in emitting}
                targets.discard(None)
                if len(targets) != 1:
                    continue
                target_id = next(iter(targets))
                if target_id not in labels:
                    continue
                row.append((labels[target_id], output_symbol, prob))
            total = sum(prob for _label, _out, prob in row)
            if total <= 0.0:
                continue
            for target_label, output_symbol, prob in row:
                graph.add_transition(
                    source,
                    target_label,
                    **{ATTR_SYMBOL: input_symbol, ATTR_OUTPUT: output_symbol, ATTR_PROB: prob / total},
                )
                used_inputs.add(input_symbol)
                used_outputs.add(output_symbol)

    total_visits = float(sum(visits.values()))
    initial = {labels[state_id]: visits[state_id] / total_visits for state_id in states if visits.get(state_id, 0) > 0}
    if not initial:
        initial = {labels[next(iter(states))]: 1.0}

    result = EpsilonTransducer(
        input_alphabet=frozenset(used_inputs),
        output_alphabet=frozenset(used_outputs),
        initial_states=frozenset(initial),
        initial_distribution=initial,
        graph=graph,
    )
    result.validate()
    return result


def _default_lmax(n: int, alphabet_size: int, min_count: int) -> int:
    if alphabet_size <= 0:
        return 1
    # The joint (input, output) history space grows as ``alphabet_size ** L``, so
    # keep the default depth modest relative to the single-process CSSR bound.
    return max(1, min(5, n // max(1, alphabet_size * min_count)))


def transcssr(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    input_alphabet: Sequence[Any] | None = None,
    output_alphabet: Sequence[Any] | None = None,
    Lmax: int | None = None,
    alpha: float = 0.001,
    test: Literal["g", "chi2"] = "g",
    min_count: int = 5,
) -> EpsilonTransducer:
    """Reconstruct an ε-transducer from paired input/output sequences (transCSSR).

    ``alpha`` is the per-test significance level for the causal-state split
    decision; the transCSSR/CSSR default of ``0.001`` favors fewer, more robust
    states. ``Lmax`` bounds the joint-history depth and ``min_count`` the minimum
    occurrences before a history is eligible to seed a new state.
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
    max_length = Lmax if Lmax is not None else _default_lmax(len(xs), joint_alphabet_size, min_count)
    counts = JointSuffixCounts.from_sequences(
        xs,
        ys,
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
        max_length=max_length + 1,
    )

    states, history_to_state = _homogenize(counts, Lmax=max_length, alpha=alpha, test=test, min_count=min_count)
    states = _determinize(states, history_to_state, counts)
    history_to_state = {history: state_id for state_id, histories in states.items() for history in histories}
    states = _drop_transient(states, history_to_state, counts)
    history_to_state = {history: state_id for state_id, histories in states.items() for history in histories}
    return _build_transducer(states, counts, history_to_state, xs, ys, length=max_length)
