"""Causal-State Splitting Reconstruction (CSSR) of ε-machines from a sample.

CSSR follows Shalizi, Shalizi & Crutchfield (arXiv:cs/0210025).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph
from sofic.inference.cssr.counts import History, SuffixCounts
from sofic.inference.cssr.significance import (
    MorphTest,
    _bonferroni_alpha,
    _observed_counts_for_morph,
    morph_test_score,
    morphs_differ,
)


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
    test: MorphTest,
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
    test: MorphTest,
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


def _recurrent_states(edges: Mapping[int, Mapping[Any, Iterable[int]]]) -> list[set[int]]:
    """Closed communicating classes (with at least one edge) of the state graph.

    ``edges[state][label]`` iterates over the successor states on ``label`` (a
    symbol here, an ``(input, output)`` pair in transCSSR).
    """
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
    test: MorphTest,
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
    test: MorphTest,
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
    test: MorphTest,
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


def suggest_lmax(
    sequence: Sequence[Any],
    *,
    alpha: float = 0.01,
    max_order: int | None = None,
    method: Literal["exact", "chi2", "aic", "bic"] = "exact",
    n_surrogates: int = 999,
    seed: int = 0,
) -> int:
    """A data-driven ``Lmax`` for :func:`cssr`: the estimated Markov order, at least 1.

    Orders ``0, 1, ...`` are tested against the next order with
    :func:`dit.inference.select_markov_order`. The default ``"exact"`` method
    compares the conditional block entropy against surrogates that preserve the
    observed ``(order + 1)``-gram counts, which is valid at any sample size, unlike
    the asymptotic chi-squared test :cite:`Pethel2014`.

    Parameters
    ----------
    sequence
        Observed symbols.
    alpha
        Significance level of each order test.
    max_order
        Largest order considered; by default the largest ``L`` whose
        ``(L + 1)``-words are seen about 5 times each on average, at most 10.
    method
        ``"exact"`` or ``"chi2"`` (sequential tests), or ``"aic"`` / ``"bic"``.
    n_surrogates
        Surrogates per test for ``"exact"``.
    seed
        Seed for the surrogates, so the suggestion is reproducible.

    Notes
    -----
    For a Markov source this recovers its order, which is the synchronization
    length CSSR needs. A strictly sofic source (such as the even process) has
    infinite Markov order, so the suggestion keeps growing with the sample; treat it
    as a lower bound on the history length the data can support, not as the source's
    synchronization length.
    """
    import dit.inference

    select_markov_order = getattr(dit.inference, "select_markov_order", None)
    if select_markov_order is None:  # pragma: no cover - depends on the installed dit
        raise ImportError("suggest_lmax requires a dit release with dit.inference.select_markov_order")
    codes: dict[Any, str] = {}
    seq = [codes.setdefault(symbol, str(len(codes))) for symbol in sequence]
    if max_order is None:
        k = max(2, len(set(seq)))
        max_order = max(1, min(10, int(np.log(max(len(seq), 1) / 5) / np.log(k)) - 1))
    order = select_markov_order(seq, max_order, method=method, alpha=alpha, n_surrogates=n_surrogates, prng=seed)
    return max(1, int(order))


def cssr(
    sequence: Sequence[Any],
    *,
    alphabet: Sequence[Any] | None = None,
    Lmax: int | Literal["auto"] | None = None,
    alpha: float = 0.01,
    test: MorphTest = "g",
    min_count: int = 5,
    correction: Literal["bonferroni"] | None = None,
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
        significance tests, and some of them split states by chance. ``"auto"``
        uses :func:`suggest_lmax`, the Markov order estimated by exact tests.
    alpha
        Significance level of each morph-equality test. The worked example of
        :cite:`Shalizi2002` uses 0.01; smaller values guard against spurious states
        when ``Lmax`` is large.
    test
        ``"g"`` (G-test), ``"chi2"``, ``"tv"`` (total-variation threshold), or
        ``"exact"`` (Monte Carlo exact G-test when expected counts are small; see
        :func:`morphs_differ`).
    min_count
        Suffixes seen fewer than this many times are not tested or placed in a state.
    correction
        ``"bonferroni"`` divides ``alpha`` by the number of suffixes eligible for
        testing, bounding the chance of any spurious split. CSSR decides each test
        in light of earlier ones, so step-up procedures that control the false
        discovery rate (Benjamini–Hochberg) do not apply directly.

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
    if Lmax == "auto":
        max_length = suggest_lmax(seq, alpha=alpha)
    else:
        max_length = Lmax if Lmax is not None else _cssr_default_lmax(len(seq), alphabet_size)
    if max_length < 0:
        raise ValueError("Lmax must be non-negative")
    counts = SuffixCounts.from_sequence(seq, alphabet=alphabet, max_length=max_length + 1)
    if correction == "bonferroni":
        alpha = _bonferroni_alpha(counts, alpha, max_length=max_length, min_count=min_count)
    elif correction is not None:
        raise ValueError(f"unknown correction {correction!r}")

    homogeneous = _suffix_homogenize(counts, Lmax=max_length, alpha=alpha, test=test, min_count=min_count)
    return _suffix_reconstruct(homogeneous, counts, seq, Lmax=max_length, alpha=alpha, test=test)
