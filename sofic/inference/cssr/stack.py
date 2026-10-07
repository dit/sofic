"""Sample-based inference for hidden Markov stack models."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Literal

from sofic.automata.learning.papni import DyckAlphabet, is_well_matched, learn_sofic_dyck_shift_papni
from sofic.exceptions import StochasticValidationError
from sofic.generators.stack_hmm import HiddenMarkovStackModel
from sofic.graph import ATTR_KIND, ATTR_SYMBOL, KIND_RETURN
from sofic.inference.cssr.counts import ConfigurationHistory, StackSuffixCounts, _push
from sofic.inference.cssr.process import _default_max_history, _recurrent_states, suggest_max_history
from sofic.inference.cssr.significance import MorphTest, _bonferroni_alpha, morph_test_score, morphs_differ
from sofic.inference.cssr.subtree import _cluster_histories_by_morph
from sofic.shifts.sofic_dyck import SoficDyckShift, TransitionRef, transition_ref

__all__ = [
    "learn_stack_hmm_mle",
    "learn_stack_hmm_papni",
    "learn_stack_hmm_cssr",
    "learn_stack_hmm_subtree",
]


def _successor_history(
    history: ConfigurationHistory,
    symbol: Any,
    *,
    alphabet: DyckAlphabet,
    length: int,
    max_stack_depth: int,
) -> ConfigurationHistory:
    suffix, stack = history
    extended = suffix + (symbol,)
    if length <= 0:
        new_suffix: tuple[Any, ...] = ()
    elif len(extended) <= length:
        new_suffix = extended
    else:
        new_suffix = extended[-length:]
    return new_suffix, _push(stack, symbol, alphabet=alphabet, max_stack_depth=max_stack_depth)


def _stack_successor_fn(
    *,
    alphabet: DyckAlphabet,
    length: int,
    max_stack_depth: int,
) -> Callable[[ConfigurationHistory, Any], ConfigurationHistory]:
    """Bind the stack-lifted successor into the ``(history, symbol)`` shape shared CSSR expects."""

    def successor(history: ConfigurationHistory, symbol: Any) -> ConfigurationHistory:
        return _successor_history(
            history,
            symbol,
            alphabet=alphabet,
            length=length,
            max_stack_depth=max_stack_depth,
        )

    return successor


_RETURN = object()


def _control_counts(counts: StackSuffixCounts, alphabet: DyckAlphabet) -> StackSuffixCounts:
    """Counts with every return symbol collapsed into one event.

    Which return symbol can follow is decided by the stack top (through matched
    call-return pairs), not by the finite control, so comparing raw morphs would
    split every control state by its stack top.
    """
    returns = alphabet.return_alphabet
    collapsed = StackSuffixCounts(
        alphabet=tuple(symbol for symbol in counts.alphabet if symbol not in returns) + (_RETURN,)
    )
    collapsed.history_counts = counts.history_counts
    for history, nxt in counts.next_counts.items():
        merged: Counter[Any] = Counter()
        for symbol, count in nxt.items():
            merged[_RETURN if symbol in returns else symbol] += count
        collapsed.next_counts[history] = merged
    return collapsed


def _stack_homogenize(
    counts: StackSuffixCounts,
    *,
    alphabet: DyckAlphabet,
    max_history: int,
    alpha: float,
    test: MorphTest,
    max_stack_depth: int,
    min_count: int = 1,
) -> tuple[dict[int, set[ConfigurationHistory]], dict[ConfigurationHistory, int]]:
    """CSSR homogenization over ``(suffix, stack)`` configurations.

    Every observed stack contributes a root ``((), stack)``; suffixes then grow one
    symbol into the past with their stack fixed, exactly as in flat CSSR. Growing
    forward from the empty configuration instead only reaches stacks of depth at
    most ``max_history``, so deeper configurations had no state and their transitions were
    dropped. Morphs are compared with return symbols collapsed (see :func:`_control_counts`).
    """
    control = _control_counts(counts, alphabet)
    states: dict[int, set[ConfigurationHistory]] = {0: {counts.empty_history}}
    history_to_state: dict[ConfigurationHistory, int] = {counts.empty_history: 0}

    def place(child: ConfigurationHistory, parent_id: int) -> None:
        target = parent_id
        if morphs_differ(control, states[parent_id], {child}, alpha=alpha, test=test):
            best_id, best_score = None, float("inf")
            for candidate_id, candidate in states.items():
                if candidate_id == parent_id or morphs_differ(control, candidate, {child}, alpha=alpha, test=test):
                    continue
                score = morph_test_score(control, candidate, {child}, test=test)
                if score < best_score:
                    best_id, best_score = candidate_id, score
            if best_id is None:
                best_id = max(states) + 1
                states[best_id] = set()
            target = best_id
        states[target].add(child)
        history_to_state[child] = target

    def observed(history: ConfigurationHistory) -> bool:
        return sum(counts.next_counts.get(history, Counter()).values()) >= max(1, min_count)

    roots = {stack for suffix, stack in counts.history_counts if not suffix and stack}
    for stack in sorted(roots, key=lambda stack: (len(stack), repr(stack))):
        if observed(((), stack)):
            place(((), stack), 0)
    for length in range(max_history):
        for state_id in sorted(states):
            for suffix, stack in sorted((h for h in states[state_id] if len(h[0]) == length), key=repr):
                for symbol in counts.alphabet:
                    child = ((symbol, *suffix), stack)
                    if child not in history_to_state and observed(child):
                        place(child, state_id)
    return states, history_to_state


def _stack_determinize(
    states: dict[int, set[ConfigurationHistory]],
    history_to_state: dict[ConfigurationHistory, int],
    counts: StackSuffixCounts,
    *,
    length: int,
    alphabet: DyckAlphabet,
    max_stack_depth: int,
) -> dict[int, set[ConfigurationHistory]]:
    """Split homogeneous states until stack-lifted transitions are unifilar.

    Updates ``history_to_state`` in place for the histories that move.
    """
    successor = _stack_successor_fn(alphabet=alphabet, length=length, max_stack_depth=max_stack_depth)
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
                buckets: dict[int, set[ConfigurationHistory]] = defaultdict(set)
                for history in histories:
                    if counts.next_counts.get(history, Counter()).get(symbol, 0) == 0:
                        continue
                    target = history_to_state.get(successor(history, symbol))
                    if target is None:
                        continue
                    buckets[target].add(history)
                if len(buckets) <= 1:
                    continue
                # Keep the largest bucket -- and every history that never emits
                # ``symbol`` -- in the original state; split the other buckets off.
                ordered = sorted(buckets.items(), key=lambda item: (-len(item[1]), min(item[1])))
                moved = set().union(*(split for _target, split in ordered[1:]))
                current[state_id] = histories - moved
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


def _stack_merge(
    states: dict[int, set[ConfigurationHistory]],
    history_to_state: dict[ConfigurationHistory, int],
    counts: StackSuffixCounts,
    *,
    alpha: float,
    test: MorphTest,
    alphabet: DyckAlphabet,
) -> dict[int, set[ConfigurationHistory]]:
    """Merge states whose pooled control morphs (see :func:`_control_counts`) are indistinguishable.

    Merging on the morph alone can fuse states with incompatible ``symbol ->
    successor`` maps, so the result need not be unifilar.
    """
    control = _control_counts(counts, alphabet)
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
                if morphs_differ(control, current[left_id], current[right_id], alpha=alpha, test=test):
                    continue
                current[left_id].update(current.pop(right_id))
                for history in current[left_id]:
                    history_to_state[history] = left_id
                changed = True
                break
            if changed:
                break
    return current


def _stack_drop_transient(
    states: dict[int, set[ConfigurationHistory]],
    history_to_state: dict[ConfigurationHistory, int],
    counts: StackSuffixCounts,
    *,
    length: int,
    alphabet: DyckAlphabet,
    max_stack_depth: int,
) -> dict[int, set[ConfigurationHistory]]:
    """Keep only states in closed communicating classes; all states if there are none.

    States left with no edge into the kept set (their successor configurations
    were too rare to be placed) are pruned repeatedly, since a generator state
    needs outgoing mass.
    """
    successor = _stack_successor_fn(alphabet=alphabet, length=length, max_stack_depth=max_stack_depth)

    def state_edges(kept: Mapping[int, set[ConfigurationHistory]]) -> dict[int, dict[Any, set[int]]]:
        edges: dict[int, dict[Any, set[int]]] = {state_id: defaultdict(set) for state_id in kept}
        for state_id, histories in kept.items():
            for history in histories:
                for symbol, count in counts.next_counts.get(history, Counter()).items():
                    target = history_to_state.get(successor(history, symbol)) if count else None
                    if target in kept:
                        edges[state_id][symbol].add(target)
        return edges

    recurrent = set().union(*_recurrent_states(state_edges(states)))
    kept = {state_id: histories for state_id, histories in states.items() if not recurrent or state_id in recurrent}
    while True:
        edges = state_edges(kept)
        dead = {state_id for state_id, by_symbol in edges.items() if not any(by_symbol.values())}
        if not dead or len(dead) == len(kept):
            return kept
        kept = {state_id: histories for state_id, histories in kept.items() if state_id not in dead}


def _counts_to_stack_hmm(
    states: dict[int, set[ConfigurationHistory]],
    counts: StackSuffixCounts,
    history_to_state: dict[ConfigurationHistory, int],
    sequence: Sequence[Any],
    *,
    alphabet: DyckAlphabet,
    length: int,
    max_stack_depth: int,
) -> HiddenMarkovStackModel:
    visits: Counter[int] = Counter()
    seq = tuple(sequence)
    stack: tuple[Any, ...] = ()
    for t in range(len(seq)):
        # Each step occupies one state: the one keyed by its longest available suffix.
        for hist_len in range(min(t, length), -1, -1):
            state = history_to_state.get((seq[t - hist_len : t], stack))
            if state is not None:
                visits[state] += 1
                break
        stack = _push(stack, seq[t], alphabet=alphabet, max_stack_depth=max_stack_depth)

    if not visits:
        raise StochasticValidationError("no empirical configuration visits")

    model = HiddenMarkovStackModel(
        call_alphabet=alphabet.call_alphabet,
        return_alphabet=alphabet.return_alphabet,
        internal_alphabet=alphabet.internal_alphabet,
    )
    state_labels = {state_id: f"s{state_id}" for state_id in states}
    for label in state_labels.values():
        model.graph.add_state(label)

    # Matched call-return pairs, as observed: a return ``r`` emitted with ``c`` on top.
    observed_pairs = {
        (stack[-1], symbol)
        for (_suffix, stack), nxt in counts.next_counts.items()
        if stack
        for symbol, count in nxt.items()
        if count > 0 and symbol in alphabet.return_alphabet
    }

    def legal(symbol: Any, stack: tuple[Any, ...]) -> bool:
        if symbol not in alphabet.return_alphabet:
            return True
        return not stack or (stack[-1], symbol) in observed_pairs

    call_refs: dict[Any, list[TransitionRef]] = defaultdict(list)
    return_refs: dict[Any, list[TransitionRef]] = defaultdict(list)
    for state_id, histories in states.items():
        source = state_labels[state_id]
        emitted: Counter[Any] = Counter()
        for history in histories:
            emitted.update(counts.next_counts.get(history, Counter()))
        # The stack model renormalizes over the moves legal in each configuration, so
        # the weights are the Luce-choice MLE over the legal sets offered by each history.
        symbols = [symbol for symbol in counts.alphabet if emitted[symbol] > 0]
        offers: Counter[tuple[Any, ...]] = Counter()
        for history in histories:
            visits_here = sum(counts.next_counts.get(history, Counter()).values())
            if visits_here:
                offers[tuple(s for s in symbols if legal(s, history[1]))] += visits_here
        weights = _luce_mle(symbols, emitted, offers, smoothing=0.0) if symbols else {}
        for symbol in counts.alphabet:
            if emitted[symbol] <= 0:
                continue
            targets: Counter[int] = Counter()
            for history in histories:
                count = counts.next_counts.get(history, Counter()).get(symbol, 0)
                if count <= 0:
                    continue
                child = _successor_history(
                    history, symbol, alphabet=alphabet, length=length, max_stack_depth=max_stack_depth
                )
                target_id = history_to_state.get(child)
                if target_id is not None:
                    targets[target_id] += count
            if not targets:
                continue
            target = state_labels[targets.most_common(1)[0][0]]
            prob = weights[symbol]
            if symbol in alphabet.call_alphabet:
                call_refs[symbol].append(model.add_call_transition(source, target, symbol, prob))
            elif symbol in alphabet.return_alphabet:
                return_refs[symbol].append(model.add_return_transition(source, target, symbol, prob))
            else:
                model.add_internal_transition(source, target, symbol, prob)

    for call_symbol, return_symbol in observed_pairs:
        for call_ref in call_refs.get(call_symbol, ()):
            for return_ref in return_refs.get(return_symbol, ()):
                model.add_matched_pair(call_ref, return_ref)

    total_visits = float(sum(visits.values()))
    initial = {state_labels[state_id]: visits[state_id] / total_visits for state_id in states if visits[state_id] > 0}
    if not initial:
        initial = {state_labels[next(iter(states))]: 1.0}
    model.initial_distribution = initial
    model.validate()
    return model


def learn_stack_hmm_cssr(
    sequence: Sequence[Any],
    *,
    alphabet: DyckAlphabet,
    max_history: int | Literal["auto"] | None = None,
    max_stack_depth: int = 8,
    alpha: float = 0.05,
    test: MorphTest = "g",
    min_count: int = 5,
    correction: Literal["bonferroni"] | None = "bonferroni",
) -> HiddenMarkovStackModel:
    """Reconstruct a stack HMM via configuration-lifted CSSR.

    ``max_history="auto"`` uses :func:`~sofic.inference.cssr.suggest_max_history`
    on the observed symbols. Stack processes generally have infinite Markov
    order, so treat it as a lower bound on the suffix length the data support.
    ``test="exact"`` and ``correction="bonferroni"`` are as in
    :func:`~sofic.inference.cssr.learn_epsilon_machine_cssr`; the correction (on by
    default; ``correction=None`` disables it) counts eligible (suffix, stack)
    configurations.
    """
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    if max_history == "auto":
        max_length = suggest_max_history(seq, alpha=alpha)
    else:
        max_length = (
            max_history if max_history is not None else _default_max_history(len(seq), len(alphabet.symbol_alphabet))
        )
    counts = StackSuffixCounts.from_sequence(
        seq,
        alphabet=alphabet,
        max_length=max_length + 1,
        max_stack_depth=max_stack_depth,
    )
    if correction == "bonferroni":
        alpha = _bonferroni_alpha(
            counts,
            alpha,
            max_length=max_length,
            min_count=min_count,
            suffix_length=lambda history: len(history[0]),
        )
    elif correction is not None:
        raise ValueError(f"unknown correction {correction!r}")
    states, history_to_state = _stack_homogenize(
        counts,
        alphabet=alphabet,
        max_history=max_length,
        alpha=alpha,
        test=test,
        max_stack_depth=max_stack_depth,
        min_count=min_count,
    )
    states = _stack_determinize(
        states,
        history_to_state,
        counts,
        length=max_length,
        alphabet=alphabet,
        max_stack_depth=max_stack_depth,
    )
    states = _stack_merge(states, history_to_state, counts, alpha=alpha, test=test, alphabet=alphabet)
    states = _stack_drop_transient(
        states, history_to_state, counts, length=max_length, alphabet=alphabet, max_stack_depth=max_stack_depth
    )
    history_to_state = {history: state_id for state_id, histories in states.items() for history in histories}
    return _counts_to_stack_hmm(
        states,
        counts,
        history_to_state,
        seq,
        alphabet=alphabet,
        length=max_length,
        max_stack_depth=max_stack_depth,
    )


def learn_stack_hmm_subtree(
    sequence: Sequence[Any],
    *,
    alphabet: DyckAlphabet,
    max_history: int,
    max_stack_depth: int = 8,
    delta: float = 0.0,
) -> HiddenMarkovStackModel:
    """Reconstruct a stack HMM by merging depth-``max_history`` configuration subtrees."""
    if max_history < 0:
        raise ValueError("max_history must be non-negative")
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    counts = StackSuffixCounts.from_sequence(
        seq,
        alphabet=alphabet,
        max_length=max_history + 1,
        max_stack_depth=max_stack_depth,
    )
    histories = {history for history in counts.history_counts if len(history[0]) <= max_history}
    histories.add(((), ()))
    proxy = counts.restricted_to(histories)
    states = _cluster_histories_by_morph(proxy, histories, delta=delta)
    history_to_state = {history: state_id for state_id, members in states.items() for history in members}
    states = _stack_determinize(
        states,
        history_to_state,
        counts,
        length=max_history,
        alphabet=alphabet,
        max_stack_depth=max_stack_depth,
    )
    history_to_state = {
        history: state_id for state_id, histories_in_state in states.items() for history in histories_in_state
    }
    states = _stack_merge(states, history_to_state, counts, alpha=0.05, test="tv", alphabet=alphabet)
    history_to_state = {
        history: state_id for state_id, histories_in_state in states.items() for history in histories_in_state
    }
    states = _stack_drop_transient(
        states, history_to_state, counts, length=max_history, alphabet=alphabet, max_stack_depth=max_stack_depth
    )
    history_to_state = {
        history: state_id for state_id, histories_in_state in states.items() for history in histories_in_state
    }
    return _counts_to_stack_hmm(
        states,
        counts,
        history_to_state,
        seq,
        alphabet=alphabet,
        length=max_history,
        max_stack_depth=max_stack_depth,
    )


def learn_stack_hmm_mle(
    shift: SoficDyckShift,
    sequence: Sequence[Any],
    *,
    smoothing: float = 1e-6,
) -> HiddenMarkovStackModel:
    """Assign MLE edge weights to a fixed Dyck topology from one sample.

    A stack HMM renormalizes a state's edge weights over the moves legal in the
    current configuration (e.g. no return on an empty stack), so the likelihood is
    a Luce choice model per state. Its maximum is found with the minorize-maximize
    iteration of :cite:`Hunter2004`, starting from each edge's frequency among the
    visits where it was legal; ``smoothing`` is added to every edge count.
    """
    from sofic.shifts.dyck_algorithms import _successors

    seq = tuple(sequence)
    states = tuple(shift.states())
    if not states:
        raise ValueError("shift has no states")
    allow_empty_returns = HiddenMarkovStackModel.from_sofic_dyck_shift(
        shift, {transition_ref(t): 1.0 for t in shift.transitions()}
    ).allow_empty_stack_returns
    state = states[0]
    stack: tuple[TransitionRef, ...] = ()
    edge_counts: Counter[TransitionRef] = Counter()
    # Per state: how often each set of legal moves was offered.
    offers: dict[Hashable, Counter[tuple[TransitionRef, ...]]] = defaultdict(Counter)

    for symbol in seq:
        legal: list[TransitionRef] = []
        chosen: tuple[TransitionRef, Hashable, tuple[TransitionRef, ...]] | None = None
        for transition in shift.graph.out_transitions(state):
            if transition.data.get(ATTR_KIND) == KIND_RETURN and not stack and not allow_empty_returns:
                continue
            for target, next_stack in _successors(shift, transition, stack):
                ref = transition_ref(transition)
                legal.append(ref)
                if chosen is None and transition.data.get(ATTR_SYMBOL) == symbol:
                    chosen = (ref, target, next_stack)
                break
        if chosen is None:
            break
        offers[state][tuple(legal)] += 1
        ref, state, stack = chosen
        edge_counts[ref] += 1

    outgoing: dict[Hashable, list[TransitionRef]] = defaultdict(list)
    for transition in shift.transitions():
        outgoing[transition.source].append(transition_ref(transition))

    probabilities: dict[TransitionRef, float] = {}
    for source, refs in outgoing.items():
        weights = _luce_mle(refs, edge_counts, offers.get(source, Counter()), smoothing=smoothing)
        probabilities.update(weights)

    return HiddenMarkovStackModel.from_sofic_dyck_shift(shift, probabilities)


def _luce_mle(
    refs: Sequence[Hashable],
    counts: Mapping[Hashable, float],
    offers: Mapping[tuple[Hashable, ...], int],
    *,
    smoothing: float,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> dict[Hashable, float]:
    """Normalized weights maximizing ``prod_t w[choice_t] / sum_{legal_t} w``.

    ``offers`` counts how often each tuple of legal moves was available.
    """
    wins = {ref: counts.get(ref, 0.0) + smoothing for ref in refs}
    exposure = {ref: sum(n for legal, n in offers.items() if ref in legal) for ref in refs}
    weights = {ref: (wins[ref] / exposure[ref] if exposure[ref] else wins[ref]) for ref in refs}
    if not any(exposure.values()):
        total = sum(weights.values())
        return {ref: (w / total if total > 0 else 1.0 / len(refs)) for ref, w in weights.items()}
    for _ in range(max_iter):
        denominators = dict.fromkeys(refs, 0.0)
        for legal, n in offers.items():
            mass = sum(weights[ref] for ref in legal)
            for ref in legal:
                denominators[ref] += n / mass
        updated = {ref: (wins[ref] / denominators[ref] if denominators[ref] else weights[ref]) for ref in refs}
        total = sum(updated.values())
        updated = {ref: w / total for ref, w in updated.items()}
        delta = max(abs(updated[ref] - weights[ref]) for ref in refs)
        weights = updated
        if delta < tol:
            break
    total = sum(weights.values())
    return {ref: w / total for ref, w in weights.items()}


def learn_stack_hmm_papni(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]] | None = None,
    *,
    alphabet: DyckAlphabet,
    sequence: Sequence[Any] | None = None,
) -> HiddenMarkovStackModel:
    """Learn stack topology via PAPNI and fit edge probabilities from ``sequence`` or positives."""
    shift = learn_sofic_dyck_shift_papni(positive, negative, alphabet=alphabet)
    fit_source: Sequence[Any]
    if sequence is not None:
        fit_source = sequence
    else:
        fit_source = max((tuple(word) for word in positive if is_well_matched(word, alphabet)), key=len, default=())
    if not fit_source:
        raise ValueError("no sequence available for parameter fitting")
    return learn_stack_hmm_mle(shift, fit_source)
