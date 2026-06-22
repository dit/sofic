"""Sample-based ε-machine reconstruction (CSSR and subtree merging).

CSSR follows Shalizi, Shalizi & Crutchfield (arXiv:cs/0210025). Subtree merging
follows Crutchfield & Young (PRL 1989; PRE 1994).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats

from pensive.exceptions import StochasticValidationError
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

History = tuple[Any, ...]


@dataclass
class SuffixCounts:
    """Empirical counts of histories and following symbols in a sequence."""

    alphabet: tuple[Any, ...]
    history_counts: Counter[History] = field(default_factory=Counter)
    next_counts: dict[History, Counter[Any]] = field(default_factory=lambda: defaultdict(Counter))

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Any],
        *,
        alphabet: Sequence[Any] | None = None,
        max_length: int | None = None,
    ) -> SuffixCounts:
        seq = tuple(sequence)
        if not seq:
            raise ValueError("sequence must be non-empty")
        alphabet = tuple(sorted(set(seq), key=repr)) if alphabet is None else tuple(alphabet)
        unknown = set(seq) - set(alphabet)
        if unknown:
            raise ValueError(f"symbols {unknown!r} not in alphabet")
        max_len = max_length if max_length is not None else len(seq)
        counts = cls(alphabet=alphabet)
        n = len(seq)
        for t in range(n):
            for length in range(0, min(t, max_len) + 1):
                history = seq[t - length : t]
                counts.history_counts[history] += 1
                nxt = seq[t]
                counts.next_counts[history][nxt] += 1
        return counts

    def morph(self, history: History, *, smoothing: float = 0.0) -> dict[Any, float]:
        """MLE (optional additive smoothing) of P(next symbol | history)."""
        counts = self.next_counts.get(history, Counter())
        total = sum(counts.values())
        if total == 0:
            uniform = 1.0 / len(self.alphabet)
            return dict.fromkeys(self.alphabet, uniform)
        denom = total + smoothing * len(self.alphabet)
        return {symbol: (counts.get(symbol, 0) + smoothing) / denom for symbol in self.alphabet}

    def state_morph(self, histories: set[History], *, smoothing: float = 0.0) -> dict[Any, float]:
        """Weighted average of history morphs with weights from occurrence counts."""
        weights = {history: float(self.history_counts.get(history, 0)) for history in histories}
        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            return self.morph((), smoothing=smoothing)
        result = dict.fromkeys(self.alphabet, 0.0)
        for history, weight in weights.items():
            morph = self.morph(history, smoothing=smoothing)
            for symbol in self.alphabet:
                result[symbol] += weight * morph[symbol]
        return {symbol: prob / total_weight for symbol, prob in result.items()}

    def marginal_morph(self) -> dict[Any, float]:
        """Global next-symbol distribution (IID morph at L=0)."""
        counts = Counter()
        for _history, counter in self.next_counts.items():
            counts.update(counter)
        grand = sum(counts.values())
        if grand == 0:
            uniform = 1.0 / len(self.alphabet)
            return dict.fromkeys(self.alphabet, uniform)
        return {symbol: counts.get(symbol, 0) / grand for symbol in self.alphabet}


def _observed_counts_for_morph(
    counts: SuffixCounts,
    histories: set[History],
) -> Counter[Any]:
    observed = Counter()
    for history in histories:
        observed.update(counts.next_counts.get(history, Counter()))
    return observed


def _contingency_rows(
    counts: SuffixCounts,
    left_histories: set[History],
    right_histories: set[History],
) -> np.ndarray | None:
    left_obs = _observed_counts_for_morph(counts, left_histories)
    right_obs = _observed_counts_for_morph(counts, right_histories)
    active = [symbol for symbol in counts.alphabet if left_obs.get(symbol, 0) + right_obs.get(symbol, 0) > 0]
    if not active:
        return None
    table = np.array(
        [
            [left_obs.get(symbol, 0) for symbol in active],
            [right_obs.get(symbol, 0) for symbol in active],
        ],
        dtype=float,
    )
    if np.allclose(table[0], table[1]):
        return None
    if table.shape[1] < 2:
        left_total = table[0].sum()
        right_total = table[1].sum()
        if left_total == 0.0 or right_total == 0.0:
            return None
        left_prob = table[0, 0] / left_total
        right_prob = table[1, 0] / right_total
        if np.isclose(left_prob, right_prob):
            return None
    return table


def morphs_differ(
    counts: SuffixCounts,
    left_histories: set[History],
    right_histories: set[History],
    *,
    alpha: float = 0.05,
    test: Literal["g", "chi2", "tv"] = "g",
    delta: float = 0.0,
) -> bool:
    """Return whether two history sets have significantly different morphs."""
    if test == "tv":
        left = counts.state_morph(left_histories)
        right = counts.state_morph(right_histories)
        distance = 0.5 * sum(abs(left[s] - right[s]) for s in counts.alphabet)
        return distance > delta

    table = _contingency_rows(counts, left_histories, right_histories)
    if table is None:
        return False
    if test == "g":
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                statistic, _p_value, _dof, expected = stats.chi2_contingency(table, lambda_="log-likelihood")
        except ValueError:
            return False
        if not np.isfinite(statistic) or np.any(expected == 0):
            return False
        dof = max(1, table.shape[1] - 1)
        critical = float(stats.chi2.ppf(1.0 - alpha, dof))
        return float(statistic) > critical
    try:
        statistic, p_value, _dof, expected = stats.chi2_contingency(table)
    except ValueError:
        return False
    if np.any(expected == 0):
        return False
    return float(p_value) < alpha


def morph_test_score(
    counts: SuffixCounts,
    left_histories: set[History],
    right_histories: set[History],
    *,
    test: Literal["g", "chi2", "tv"] = "g",
) -> float:
    """Score for matching morphs (lower is more similar)."""
    if test == "tv":
        left = counts.state_morph(left_histories)
        right = counts.state_morph(right_histories)
        return 0.5 * sum(abs(left[s] - right[s]) for s in counts.alphabet)
    table = _contingency_rows(counts, left_histories, right_histories)
    if table is None:
        return 0.0
    try:
        if test == "g":
            with np.errstate(invalid="ignore", divide="ignore"):
                statistic, _p, _dof, _expected = stats.chi2_contingency(table, lambda_="log-likelihood")
            return float(statistic) if np.isfinite(statistic) else 0.0
        statistic, _p, _dof, _expected = stats.chi2_contingency(table)
        return float(statistic)
    except ValueError:
        return 0.0


def _default_lmax(n: int, alphabet_size: int, min_count: int) -> int:
    if alphabet_size <= 0:
        return 1
    return max(1, min(15, n // max(1, alphabet_size * min_count)))


def _successor_history(history: History, symbol: Any, length: int) -> History:
    extended = history + (symbol,)
    if length <= 0:
        return ()
    if len(extended) <= length:
        return extended
    return extended[-length:]


def _cssr_homogenize(
    counts: SuffixCounts,
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
) -> tuple[dict[int, set[History]], dict[History, int]]:
    """Return state id -> histories and history -> state id."""
    states: dict[int, set[History]] = {0: {()}}
    history_to_state: dict[History, int] = {(): 0}
    next_state_id = 1

    for _length in range(Lmax + 1):
        for state_id in sorted(states):
            histories = set(states[state_id])
            for history in list(histories):
                for symbol in counts.alphabet:
                    child = history + (symbol,)
                    if child in history_to_state:
                        continue
                    if counts.history_counts.get(child, 0) == 0:
                        continue
                    child_histories = {child}
                    if morphs_differ(
                        counts,
                        histories,
                        child_histories,
                        alpha=alpha,
                        test=test,
                    ):
                        best_state: int | None = None
                        best_score = float("inf")
                        for candidate_id, candidate_histories in states.items():
                            if morphs_differ(
                                counts,
                                candidate_histories,
                                child_histories,
                                alpha=alpha,
                                test=test,
                            ):
                                continue
                            score = morph_test_score(
                                counts,
                                candidate_histories,
                                child_histories,
                                test=test,
                            )
                            if score < best_score:
                                best_score = score
                                best_state = candidate_id
                        if best_state is None:
                            best_state = next_state_id
                            states[next_state_id] = set()
                            next_state_id += 1
                        states[best_state].add(child)
                        history_to_state[child] = best_state
                    else:
                        states[state_id].add(child)
                        history_to_state[child] = state_id
    return states, history_to_state


def _cssr_determinize(
    states: dict[int, set[History]],
    history_to_state: dict[History, int],
    counts: SuffixCounts,
    *,
    length: int,
) -> dict[int, set[History]]:
    """Split homogeneous states until transitions are unifilar."""
    current = {state_id: set(histories) for state_id, histories in states.items()}
    changed = True
    next_state_id = max(current) + 1 if current else 0

    while changed:
        changed = False
        for state_id in sorted(current):
            histories = current[state_id]
            if len(histories) <= 1:
                continue
            for symbol in counts.alphabet:
                buckets: dict[int, set[History]] = defaultdict(set)
                for history in histories:
                    if counts.next_counts.get(history, Counter()).get(symbol, 0) == 0:
                        continue
                    child = history + (symbol,)
                    target = history_to_state.get(child)
                    if target is None:
                        continue
                    buckets[target].add(history)
                if len(buckets) <= 1:
                    continue
                # Keep the largest bucket in the original state; split others.
                ordered = sorted(buckets.items(), key=lambda item: (-len(item[1]), min(item[1])))
                keep_target, keep_histories = ordered[0]
                current[state_id] = keep_histories
                for _target, split_histories in ordered[1:]:
                    new_id = next_state_id
                    next_state_id += 1
                    current[new_id] = split_histories
                    for history in split_histories:
                        history_to_state[history] = new_id
                changed = True
                break
            if changed:
                break
    return current


def _merge_similar_states(
    states: dict[int, set[History]],
    history_to_state: dict[History, int],
    counts: SuffixCounts,
    *,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
) -> dict[int, set[History]]:
    """Merge inferred states whose pooled morphs are statistically indistinguishable."""
    current = {state_id: set(histories) for state_id, histories in states.items()}
    changed = True
    while changed:
        changed = False
        state_ids = sorted(current)
        for index, left_id in enumerate(state_ids):
            if left_id not in current:
                continue
            for right_id in state_ids[index + 1 :]:
                if right_id not in current:
                    continue
                if morphs_differ(
                    counts,
                    current[left_id],
                    current[right_id],
                    alpha=alpha,
                    test=test,
                ):
                    continue
                current[left_id].update(current.pop(right_id))
                for history in current[left_id]:
                    history_to_state[history] = left_id
                changed = True
                break
            if changed:
                break
    return current


def _drop_transient_states(
    states: dict[int, set[History]],
    history_to_state: dict[History, int],
    counts: SuffixCounts,
    *,
    length: int,
) -> dict[int, set[History]]:
    """Keep only states in bottom strongly connected components."""
    import networkx as nx

    successors: dict[int, dict[Any, set[int]]] = defaultdict(lambda: defaultdict(set))
    for state_id, histories in states.items():
        for history in histories:
            for symbol in counts.alphabet:
                if counts.next_counts.get(history, Counter()).get(symbol, 0) == 0:
                    continue
                child = history + (symbol,)
                target = history_to_state.get(child)
                if target is None:
                    continue
                successors[state_id][symbol].add(target)

    graph = nx.DiGraph()
    for state_id in states:
        graph.add_node(state_id)
    for state_id, by_symbol in successors.items():
        for targets in by_symbol.values():
            for target in targets:
                graph.add_edge(state_id, target)

    if graph.number_of_edges() == 0:
        return states

    recurrent: set[int] = set()
    for component in nx.strongly_connected_components(graph):
        if not component:
            continue
        subgraph = graph.subgraph(component)
        has_cycle = subgraph.number_of_edges() > 0 and (
            len(component) > 1 or any(subgraph.has_edge(node, node) for node in component)
        )
        if not has_cycle:
            continue
        outgoing = any(graph.has_edge(v, w) for v in component for w in graph.nodes if w not in component)
        if not outgoing:
            recurrent.update(component)

    if not recurrent:
        return states
    return {state_id: histories for state_id, histories in states.items() if state_id in recurrent}


def _empirical_state_visits(
    sequence: Sequence[Any],
    history_to_state: dict[History, int],
    *,
    length: int,
) -> Counter[int]:
    visits: Counter[int] = Counter()
    seq = tuple(sequence)
    for t in range(len(seq)):
        for hist_len in range(0, min(t, length) + 1):
            history = seq[t - hist_len : t]
            state = history_to_state.get(history)
            if state is not None:
                visits[state] += 1
    return visits


def _counts_to_mealy(
    states: dict[int, set[History]],
    counts: SuffixCounts,
    history_to_state: dict[History, int],
    sequence: Sequence[Any],
    *,
    length: int,
) -> EpsilonMachine:
    visits = _empirical_state_visits(sequence, history_to_state, length=length)
    if not visits:
        raise StochasticValidationError("no empirical state visits")

    graph = TransitionGraph()
    state_labels = {state_id: f"s{state_id}" for state_id in states}
    for label in state_labels.values():
        graph.add_state(label)

    for state_id, histories in states.items():
        label = state_labels[state_id]
        morph = counts.state_morph(histories)
        for symbol in counts.alphabet:
            prob = morph[symbol]
            if prob <= 0.0:
                continue
            emitting = [
                history for history in histories if counts.next_counts.get(history, Counter()).get(symbol, 0) > 0
            ]
            if not emitting:
                continue
            child_histories = {history + (symbol,) for history in emitting}
            targets = {history_to_state.get(child) for child in child_histories}
            targets.discard(None)
            if not targets:
                continue
            if len(targets) > 1:
                raise StochasticValidationError(f"non-unifilar inferred transition from {label!r} on {symbol!r}")
            target_label = state_labels[next(iter(targets))]
            graph.add_transition(label, target_label, **{ATTR_PROB: prob, ATTR_EMISSION: symbol})

    total_visits = float(sum(visits.values()))
    initial = {state_labels[state_id]: visits[state_id] / total_visits for state_id in states if visits[state_id] > 0}
    if not initial:
        initial = {state_labels[next(iter(states))]: 1.0}

    machine = EpsilonMachine(
        graph=graph,
        initial_distribution=initial,
        observation_alphabet=frozenset(counts.alphabet),
    )
    machine.validate()
    return machine


def cssr(
    sequence: Sequence[Any],
    *,
    alphabet: Sequence[Any] | None = None,
    Lmax: int | None = None,
    alpha: float = 0.05,
    test: Literal["g", "chi2", "tv"] = "g",
    min_count: int = 5,
) -> EpsilonMachine:
    """Reconstruct an ε-machine by Causal-State Splitting Reconstruction (CSSR)."""
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    alphabet_size = len(set(seq)) if alphabet is None else len(tuple(alphabet))
    max_length = Lmax if Lmax is not None else _default_lmax(len(seq), alphabet_size, min_count)
    counts = SuffixCounts.from_sequence(seq, alphabet=alphabet, max_length=max_length + 1)

    states, history_to_state = _cssr_homogenize(
        counts,
        Lmax=max_length,
        alpha=alpha,
        test=test,
    )
    states = _cssr_determinize(states, history_to_state, counts, length=max_length)
    states = _merge_similar_states(states, history_to_state, counts, alpha=alpha, test=test)
    states = _drop_transient_states(states, history_to_state, counts, length=max_length)
    history_to_state = {history: state_id for state_id, histories in states.items() for history in histories}
    return _counts_to_mealy(states, counts, history_to_state, seq, length=max_length)


def _morph_distance(
    counts: SuffixCounts,
    left: History,
    right: History,
    *,
    delta: float,
) -> float:
    left_morph = counts.morph(left)
    right_morph = counts.morph(right)
    return 0.5 * sum(abs(left_morph[s] - right_morph[s]) for s in counts.alphabet)


def _morphs_equivalent(
    counts: SuffixCounts,
    left: History,
    right: History,
    *,
    delta: float,
) -> bool:
    left_morph = counts.morph(left)
    right_morph = counts.morph(right)
    if delta > 0.0:
        return _morph_distance(counts, left, right, delta=delta) <= delta
    return all(np.isclose(left_morph[symbol], right_morph[symbol], rtol=0.0, atol=1e-3) for symbol in counts.alphabet)


def _cluster_histories_by_morph(
    counts: SuffixCounts,
    histories: set[History],
    *,
    delta: float,
) -> dict[int, set[History]]:
    parent: dict[History, History] = {history: history for history in histories}

    def find(history: History) -> History:
        root = history
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(left: History, right: History) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    history_list = sorted(histories)
    for index, left in enumerate(history_list):
        for right in history_list[index + 1 :]:
            if _morphs_equivalent(counts, left, right, delta=delta):
                union(left, right)

    clusters: dict[History, set[History]] = defaultdict(set)
    for history in histories:
        clusters[find(history)].add(history)

    states: dict[int, set[History]] = {}
    for state_id, (_root, members) in enumerate(clusters.items()):
        states[state_id] = set(members)
    return states


def subtree_merge(
    sequence: Sequence[Any],
    *,
    L: int,
    delta: float = 0.0,
    alphabet: Sequence[Any] | None = None,
) -> EpsilonMachine:
    """Reconstruct an ε-machine by merging depth-``L`` subtrees (Crutchfield--Young)."""
    if L < 0:
        raise ValueError("L must be non-negative")
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    counts = SuffixCounts.from_sequence(seq, alphabet=alphabet, max_length=L + 1)

    histories = {history for history in counts.history_counts if len(history) <= L}
    histories.add(())

    states = _cluster_histories_by_morph(counts, histories, delta=delta)
    history_to_state = {history: state_id for state_id, members in states.items() for history in members}

    states = _cssr_determinize(states, history_to_state, counts, length=L)
    history_to_state = {
        history: state_id for state_id, histories_in_state in states.items() for history in histories_in_state
    }
    states = _merge_similar_states(states, history_to_state, counts, alpha=0.05, test="tv")
    states = _drop_transient_states(states, history_to_state, counts, length=L)
    history_to_state = {
        history: state_id for state_id, histories_in_state in states.items() for history in histories_in_state
    }
    return _counts_to_mealy(states, counts, history_to_state, seq, length=L)
