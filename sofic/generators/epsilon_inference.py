"""Sample-based ε-machine reconstruction (CSSR, subtree merging, and spectral).

CSSR follows Shalizi, Shalizi & Crutchfield (arXiv:cs/0210025). Subtree merging
follows Crutchfield & Young (PRL 1989; PRE 1994). Spectral reconstruction learns
a weighted finite automaton by Hankel SVD :cite:`Balle2014,Hsu2012` and extracts
causal states as mixed states of the learned operators :cite:`Ellison2009`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Literal

import numpy as np
from scipy import stats

from sofic.exceptions import StochasticValidationError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

History = tuple[Any, ...]


@dataclass
class SuffixCounts:
    """Empirical counts of histories and following symbols in a sequence."""

    alphabet: tuple[Any, ...]
    history_counts: Counter[History] = field(default_factory=Counter)
    next_counts: dict[History, Counter[Any]] = field(default_factory=lambda: defaultdict(Counter))

    #: History key used as the fallback for an empty history set (overridden by stack counts).
    empty_history: ClassVar[History] = ()

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
            return self.morph(self.empty_history, smoothing=smoothing)
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

    def restricted_to(self, histories: set[History]) -> SuffixCounts:
        """Return a plain :class:`SuffixCounts` proxy limited to ``histories``.

        The morph/comparison helpers only read the history sets handed to them, so
        stack inference can reuse them by projecting its configuration counts onto a
        flat proxy without changing any results.
        """
        proxy = SuffixCounts(alphabet=self.alphabet)
        proxy.history_counts = Counter({h: self.history_counts.get(h, 0) for h in histories})
        proxy.next_counts = defaultdict(Counter)
        for history in histories:
            proxy.next_counts[history] = self.next_counts.get(history, Counter())
        return proxy


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


def _g_statistic(table: np.ndarray) -> float | None:
    """G-test statistic of a contingency table, as ``scipy.stats.chi2_contingency`` computes it.

    Includes Yates' correction for one degree of freedom, and returns ``None`` if an
    expected count is zero. Inlined because the test runs once per pair of
    histories, and the general scipy routine dominated inference time.
    """
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    if np.any(expected == 0):
        return None
    observed = table
    if (table.shape[0] - 1) * (table.shape[1] - 1) == 1:
        diff = expected - observed
        observed = observed + np.sign(diff) * np.minimum(0.5, np.abs(diff))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(observed > 0, observed * np.log(observed / expected), 0.0)
    return 2.0 * float(terms.sum())


@lru_cache(maxsize=256)
def _chi2_critical(alpha: float, dof: int) -> float:
    return float(stats.chi2.ppf(1.0 - alpha, dof))


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
        statistic = _g_statistic(table)
        if statistic is None or not np.isfinite(statistic):
            return False
        return statistic > _chi2_critical(alpha, max(1, table.shape[1] - 1))
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
            statistic = _g_statistic(table)
            return statistic if statistic is not None and np.isfinite(statistic) else 0.0
        statistic, _p, _dof, _expected = stats.chi2_contingency(table)
        return float(statistic)
    except ValueError:
        return 0.0


def _default_lmax(n: int, alphabet_size: int, min_count: int) -> int:
    if alphabet_size <= 0:
        return 1
    return max(1, min(15, n // max(1, alphabet_size * min_count)))


def _grow_history(history: History, symbol: Any) -> History:
    """Default successor: append the symbol without truncation (flat ε-machine CSSR)."""
    return history + (symbol,)


def _cssr_homogenize(
    counts: SuffixCounts,
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
    successor_fn: Callable[[History, Any], History] = _grow_history,
) -> tuple[dict[int, set[History]], dict[History, int]]:
    """Return state id -> histories and history -> state id."""
    states: dict[int, set[History]] = {0: {counts.empty_history}}
    history_to_state: dict[History, int] = {counts.empty_history: 0}
    next_state_id = 1

    for _length in range(Lmax + 1):
        for state_id in sorted(states):
            histories = set(states[state_id])
            for history in list(histories):
                for symbol in counts.alphabet:
                    child = successor_fn(history, symbol)
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
    successor_fn: Callable[[History, Any], History] = _grow_history,
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
                    child = successor_fn(history, symbol)
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
    """Merge inferred states whose pooled morphs are statistically indistinguishable.

    Merging on the morph alone can fuse states with incompatible ``symbol ->
    successor`` maps, yielding a non-unifilar partition. Callers must re-run
    :func:`_cssr_determinize` afterwards to restore unifilarity.
    """
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
    successor_fn: Callable[[History, Any], History] = _grow_history,
) -> dict[int, set[History]]:
    """Keep only states in bottom strongly connected components."""
    import networkx as nx

    successors: dict[int, dict[Any, set[int]]] = defaultdict(lambda: defaultdict(set))
    for state_id, histories in states.items():
        for history in histories:
            for symbol in counts.alphabet:
                if counts.next_counts.get(history, Counter()).get(symbol, 0) == 0:
                    continue
                target = history_to_state.get(successor_fn(history, symbol))
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


def _cssr_default_lmax(n: int, alphabet_size: int) -> int:
    """A third of ``log_k n``, between 1 and 10.

    Each length-``L`` word is then seen about ``n ** (2/3)`` times. Longer suffixes
    multiply the number of significance tests, and with them the false splits.
    """
    k = max(2, alphabet_size)
    return max(1, min(10, int(np.log(n) / (3 * np.log(k)))))


def _suffix_homogenize(
    counts: SuffixCounts,
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
    min_count: int = 1,
) -> list[set[History]]:
    """CSSR homogenization: grow suffixes one symbol into the past, up to length ``Lmax``.

    Each child suffix ``a x`` stays in its parent's state unless its next-symbol
    distribution differs significantly; then it joins the most similar state that
    does not differ, or starts a new one. As in :cite:`Shalizi2002`, states keep
    the suffixes of every length they collect. Suffixes seen fewer than ``min_count``
    times are not tested: the significance test is unreliable on so few counts.
    """
    states: list[set[History]] = [{()}]
    for length in range(Lmax):
        for parent_id in range(len(states)):
            parent = states[parent_id]
            for history in sorted((h for h in parent if len(h) == length), key=repr):
                for symbol in counts.alphabet:
                    child = (symbol, *history)
                    if sum(counts.next_counts.get(child, Counter()).values()) < max(1, min_count):
                        continue
                    if not morphs_differ(counts, parent, {child}, alpha=alpha, test=test):
                        parent.add(child)
                        continue
                    best_id, best_score = None, float("inf")
                    for candidate_id, candidate in enumerate(states):
                        if candidate_id == parent_id:
                            continue
                        if morphs_differ(counts, candidate, {child}, alpha=alpha, test=test):
                            continue
                        score = morph_test_score(counts, candidate, {child}, test=test)
                        if score < best_score:
                            best_id, best_score = candidate_id, score
                    if best_id is None:
                        states.append({child})
                    else:
                        states[best_id].add(child)
    return states


def _suffix_successor(history: History, symbol: Any, Lmax: int) -> History:
    """The suffix that follows ``history`` on ``symbol``: extended, or truncated at ``Lmax``."""
    extended = (*history, symbol)
    return extended[1:] if len(extended) > Lmax else extended


def _suffix_edges(
    states: list[set[History]],
    counts: SuffixCounts,
    alive: set[int],
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
    resolve: bool = True,
) -> dict[int, dict[Any, dict[int, set[History]]]]:
    """Successor states of each alive state, by symbol, with the suffixes that lead there.

    A suffix shorter than ``Lmax`` moves to the state holding its one-symbol extension.
    A length-``Lmax`` suffix must drop its oldest symbol, which can forget the phase
    of a non-Markovian process: for the even process, the truncation of ``0111`` is
    ``111``, whose parity is unknown. So the length-``Lmax + 1`` suffix is tested
    against the truncated suffix's state, and if its morph differs, it moves to the
    alive state whose morph it matches best instead. ``resolve=False`` always truncates.
    """
    history_to_state = {h: index for index in alive for h in states[index]}
    edges: dict[int, dict[Any, dict[int, set[History]]]] = {}
    for index in alive:
        by_symbol: dict[Any, dict[int, set[History]]] = defaultdict(lambda: defaultdict(set))
        for history in states[index]:
            for symbol, count in counts.next_counts.get(history, Counter()).items():
                if count == 0:
                    continue
                extended = (*history, symbol)
                if len(extended) <= Lmax:
                    target = history_to_state.get(extended)
                else:
                    target = history_to_state.get(extended[1:])
                    if (
                        resolve
                        and counts.next_counts.get(extended)
                        and (
                            target is None or morphs_differ(counts, states[target], {extended}, alpha=alpha, test=test)
                        )
                    ):
                        best_score = float("inf")
                        for candidate in sorted(alive):
                            if morphs_differ(counts, states[candidate], {extended}, alpha=alpha, test=test):
                                continue
                            score = morph_test_score(counts, states[candidate], {extended}, test=test)
                            if score < best_score:
                                target, best_score = candidate, score
                if target is not None:
                    by_symbol[symbol][target].add(history)
        edges[index] = by_symbol
    return edges


def _recurrent_states(edges: dict[int, dict[Any, dict[int, set[History]]]]) -> list[set[int]]:
    """Closed communicating classes (with at least one edge) of the state graph."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(edges)
    for source, by_symbol in edges.items():
        for targets in by_symbol.values():
            graph.add_edges_from((source, target) for target in targets)
    condensed = nx.condensation(graph)
    classes = []
    for node in condensed:
        members = set(condensed.nodes[node]["members"])
        if condensed.out_degree(node) == 0 and graph.subgraph(members).number_of_edges() > 0:
            classes.append(members)
    return classes


def _suffix_determinize(
    states: list[set[History]],
    counts: SuffixCounts,
    alive: set[int],
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
    resolve: bool = True,
) -> tuple[list[set[History]], set[int]]:
    """Split alive states until each (state, symbol) pair has a single alive successor.

    Successors in pruned (transient) states are ignored, as in :cite:`Shalizi2002`.
    """
    states = [set(h) for h in states]
    alive = set(alive)
    while True:
        edges = _suffix_edges(states, counts, alive, Lmax=Lmax, alpha=alpha, test=test, resolve=resolve)
        split = None
        for index in sorted(alive):
            for symbol in sorted(edges[index], key=repr):
                if len(edges[index][symbol]) > 1:
                    split = (index, symbol)
                    break
            if split:
                break
        if split is None:
            return states, alive
        index, symbol = split
        groups = sorted(edges[index][symbol].values(), key=lambda g: (-len(g), sorted(map(repr, g))))
        for group in groups[1:]:
            states[index] -= group
            states.append(set(group))
            alive.add(len(states) - 1)


def _suffix_machine(
    states: list[set[History]],
    counts: SuffixCounts,
    sequence: Sequence[Any],
    alive: set[int],
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
    resolve: bool = True,
) -> EpsilonMachine:
    """Build the ε-machine on the most-visited recurrent class of the alive states."""
    edges = _suffix_edges(states, counts, alive, Lmax=Lmax, alpha=alpha, test=test, resolve=resolve)
    history_to_state = {h: index for index in alive for h in states[index]}

    visits: Counter[int] = Counter()
    seq = tuple(sequence)
    for t in range(len(seq) + 1):
        for length in range(min(t, Lmax), -1, -1):
            state = history_to_state.get(seq[t - length : t])
            if state is not None:
                visits[state] += 1
                break

    classes = _recurrent_states(edges)
    if not classes:
        raise StochasticValidationError("no recurrent inferred states; the sample is too short for this Lmax")
    keep = max(classes, key=lambda members: (sum(visits[s] for s in members), -min(members)))

    labels = {state: f"s{rank}" for rank, state in enumerate(sorted(keep))}
    transitions = TransitionGraph()
    for state in sorted(keep):
        transitions.add_state(labels[state])
    for state in sorted(keep):
        longest = max(len(h) for h in states[state])
        observed = _observed_counts_for_morph(counts, {h for h in states[state] if len(h) == longest})
        weights = {
            symbol: (next(iter(targets)), float(observed.get(symbol, 0)))
            for symbol, targets in edges[state].items()
            if observed.get(symbol, 0) > 0
        }
        if not weights:
            weights = {
                symbol: (next(iter(targets)), float(sum(len(h) for h in targets.values())))
                for symbol, targets in edges[state].items()
            }
        total = sum(weight for _, weight in weights.values())
        for symbol, (target, weight) in sorted(weights.items(), key=lambda item: repr(item[0])):
            transitions.add_transition(
                labels[state], labels[target], **{ATTR_PROB: weight / total, ATTR_EMISSION: symbol}
            )

    kept_visits = {state: visits[state] for state in keep if visits[state] > 0}
    total_visits = float(sum(kept_visits.values()))
    initial = (
        {labels[state]: count / total_visits for state, count in kept_visits.items()}
        if total_visits > 0
        else {labels[min(keep)]: 1.0}
    )
    machine = EpsilonMachine(
        graph=transitions,
        initial_distribution=initial,
        observation_alphabet=frozenset(counts.alphabet),
    )
    machine.validate()
    return machine


def _suffix_reconstruct(
    states: list[set[History]],
    counts: SuffixCounts,
    sequence: Sequence[Any],
    *,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
) -> EpsilonMachine:
    """Prune transient states, determinize, and build the machine from homogeneous ``states``."""

    def reconstruct(resolve: bool) -> EpsilonMachine:
        everything = set(range(len(states)))
        edges = _suffix_edges(states, counts, everything, Lmax=Lmax, alpha=alpha, test=test, resolve=resolve)
        alive = set().union(*_recurrent_states(edges)) or everything
        split, alive = _suffix_determinize(states, counts, alive, Lmax=Lmax, alpha=alpha, test=test, resolve=resolve)
        return _suffix_machine(split, counts, sequence, alive, Lmax=Lmax, alpha=alpha, test=test, resolve=resolve)

    machine = reconstruct(resolve=True)
    # Resolving truncated successors needs Lmax at least the synchronization length. When it
    # is shorter, resolution can close off a state that never emits some observed symbol.
    if {t.data[ATTR_EMISSION] for t in machine.transitions()} < set(sequence):
        machine = reconstruct(resolve=False)
    return machine


def cssr(
    sequence: Sequence[Any],
    *,
    alphabet: Sequence[Any] | None = None,
    Lmax: int | None = None,
    alpha: float = 0.01,
    test: Literal["g", "chi2", "tv"] = "g",
    min_count: int = 5,
) -> EpsilonMachine:
    """Reconstruct an ε-machine by Causal-State Splitting Reconstruction :cite:`Shalizi2004`.

    Suffixes are grown one symbol into the past up to length ``Lmax`` and grouped by
    their next-symbol distributions (homogenization), then states are split until
    every transition is deterministic (determinization). The result is restricted
    to its most-visited closed class, so it is always a valid recurrent machine.

    Parameters
    ----------
    sequence
        Observed symbols.
    alphabet
        Symbol alphabet; defaults to the symbols in ``sequence``.
    Lmax
        Longest suffix considered; by default a third of ``log_k len(sequence)``,
        between 1 and 10. It should be at least the synchronization length of the
        source (for a Markov source, its order). Larger values run many more
        significance tests, and some of them split states by chance.
    alpha
        Significance level of each morph-equality test. The worked example of
        :cite:`Shalizi2002` uses 0.01; smaller values guard against spurious states
        when ``Lmax`` is large.
    test
        ``"g"`` (G-test), ``"chi2"``, or ``"tv"`` (total-variation threshold).
    min_count
        Suffixes seen fewer than this many times are not tested or placed in a state.

    Notes
    -----
    A process that is not exactly synchronizable (no finite past determines its
    state, such as :func:`~sofic.examples.processes.ABC`) has no finite-``Lmax``
    reconstruction. CSSR then returns more states than the ε-machine, with an
    entropy rate that approaches the true one from above as ``Lmax`` grows.
    """
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    alphabet_size = len(set(seq)) if alphabet is None else len(tuple(alphabet))
    max_length = Lmax if Lmax is not None else _cssr_default_lmax(len(seq), alphabet_size)
    if max_length < 0:
        raise ValueError("Lmax must be non-negative")
    counts = SuffixCounts.from_sequence(seq, alphabet=alphabet, max_length=max_length + 1)

    homogeneous = _suffix_homogenize(counts, Lmax=max_length, alpha=alpha, test=test, min_count=min_count)
    return _suffix_reconstruct(homogeneous, counts, seq, Lmax=max_length, alpha=alpha, test=test)


#: Significance level used by subtree merging when ``delta = 0`` and to resolve truncated successors.
_SUBTREE_ALPHA = 0.01


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
    if delta > 0.0:
        return _morph_distance(counts, left, right, delta=delta) <= delta
    return not morphs_differ(counts, {left}, {right}, alpha=_SUBTREE_ALPHA, test="g")


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
    """Reconstruct an ε-machine by merging depth-``L`` subtrees (Crutchfield--Young).

    Histories up to length ``L`` are clustered by next-symbol distribution: within
    total-variation distance ``delta``, or, when ``delta = 0``, unless a G-test at
    significance 0.01 tells them apart. The clusters are then determinized as in
    :func:`cssr`.
    """
    if L < 0:
        raise ValueError("L must be non-negative")
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    counts = SuffixCounts.from_sequence(seq, alphabet=alphabet, max_length=L + 1)

    histories = {history for history in counts.history_counts if len(history) <= L}
    histories.add(())
    states = list(_cluster_histories_by_morph(counts, histories, delta=delta).values())

    return _suffix_reconstruct(states, counts, seq, Lmax=L, alpha=_SUBTREE_ALPHA, test="g")


def spectral(
    sequences: Iterable[Any] | None = None,
    *,
    word_probability: Callable[[Sequence[Any]], float] | None = None,
    alphabet: Sequence[Any] | None = None,
    rank: int | None = None,
    prefix_length: int = 3,
    suffix_length: int | None = None,
    singular_value_threshold: float = 1e-3,
    min_singular_value: float = 1e-12,
    max_states: int = 10_000,
) -> EpsilonMachine:
    """Reconstruct an ε-machine by spectral learning then mixed-state extraction.

    Learns a weighted finite automaton / observable-operator model from block
    statistics :cite:`Balle2014,Hsu2012`, then extracts causal states as the
    mixed states of those operators :cite:`Ellison2009`. When the learned
    operators are non-negative this is a Mealy projection followed by
    :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm`; signed
    operators use mixed-state enumeration rather than a clustering heuristic.

    Parameters
    ----------
    sequences
        A single observed realization or an iterable of realizations. Ignored
        when ``word_probability`` is given.
    word_probability
        Optional exact block-probability function ``f(word) -> float``.
        ``alphabet`` is then required.
    alphabet
        Observation alphabet. Inferred from ``sequences`` when omitted.
    rank
        Number of latent states. When ``None`` the rank is chosen from the
        Hankel singular-value spectrum.
    prefix_length, suffix_length
        Maximum lengths of the prefix and suffix bases. ``suffix_length``
        defaults to ``prefix_length``.
    singular_value_threshold, min_singular_value
        Cutoffs for automatic rank selection; see
        :func:`~sofic.inference.spectral.learn_spectral_wfa`.
    max_states
        Safety cap on enumerated mixed states.
    """
    from sofic.inference.spectral import learn_spectral_wfa, project_to_epsilon_machine

    model = learn_spectral_wfa(
        sequences,
        word_probability=word_probability,
        alphabet=alphabet,
        rank=rank,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        singular_value_threshold=singular_value_threshold,
        min_singular_value=min_singular_value,
    )
    return project_to_epsilon_machine(model, max_states=max_states)
