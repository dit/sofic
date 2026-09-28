"""Sample-based inference for hidden Markov stack models."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any, ClassVar, Literal

from sofic.automata.papni import DyckAlphabet, is_well_matched, learn_sofic_dyck_shift_papni
from sofic.exceptions import StochasticValidationError
from sofic.generators.epsilon_inference import (
    History,
    MorphTest,
    SuffixCounts,
    _bonferroni_alpha,
    _cluster_histories_by_morph,
    _cssr_default_lmax,
    _cssr_determinize,
    _drop_transient_states,
    _merge_similar_states,
    morph_test_score,
    morphs_differ,
    suggest_lmax,
)
from sofic.generators.stack_hmm import HiddenMarkovStackModel
from sofic.graph import ATTR_SYMBOL
from sofic.shifts.sofic_dyck import SoficDyckShift, TransitionRef, transition_ref

__all__ = [
    "ConfigurationHistory",
    "StackSuffixCounts",
    "fit_stack_hmm_mle",
    "learn_stack_hmm_papni",
    "stack_cssr",
    "stack_subtree_merge",
]

ConfigurationHistory = tuple[tuple[Any, ...], tuple[Any, ...]]


class StackSuffixCounts(SuffixCounts):
    """Empirical counts of (suffix, stack) histories and following symbols.

    Shares the morph / comparison machinery of :class:`SuffixCounts`; only the
    empty-history key and the sequence-scanning constructor differ.
    """

    empty_history: ClassVar[History] = ((), ())

    def __init__(
        self,
        alphabet: tuple[Any, ...],
        history_counts: Counter[ConfigurationHistory] | None = None,
        next_counts: dict[ConfigurationHistory, Counter[Any]] | None = None,
    ) -> None:
        super().__init__(
            alphabet=alphabet,
            history_counts=history_counts if history_counts is not None else Counter(),
            next_counts=next_counts if next_counts is not None else defaultdict(Counter),
        )

    @classmethod
    def from_sequence(  # type: ignore[override]
        cls,
        sequence: Sequence[Any],
        *,
        alphabet: DyckAlphabet,
        max_length: int | None = None,
        max_stack_depth: int = 8,
    ) -> StackSuffixCounts:
        seq = tuple(sequence)
        if not seq:
            raise ValueError("sequence must be non-empty")
        visible_alphabet = tuple(sorted(alphabet.symbol_alphabet, key=repr))
        max_len = max_length if max_length is not None else len(seq)
        counts = cls(alphabet=visible_alphabet)
        stack: list[Any] = []
        for t, symbol in enumerate(seq):
            if symbol not in alphabet.symbol_alphabet:
                raise ValueError(f"symbol {symbol!r} not in Dyck alphabet")
            for length in range(0, min(t, max_len) + 1):
                suffix = seq[t - length : t]
                history = (suffix, tuple(stack))
                counts.history_counts[history] += 1
                counts.next_counts[history][symbol] += 1
            if symbol in alphabet.call_alphabet:
                if len(stack) >= max_stack_depth:
                    stack = stack[1:]
                stack.append(symbol)
            elif symbol in alphabet.return_alphabet:
                if stack:
                    stack.pop()
        return counts


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

    stack_list = list(stack)
    if symbol in alphabet.call_alphabet:
        if len(stack_list) >= max_stack_depth:
            stack_list = stack_list[1:]
        stack_list.append(symbol)
    elif symbol in alphabet.return_alphabet and stack_list:
        stack_list.pop()
    return new_suffix, tuple(stack_list)


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
    Lmax: int,
    alpha: float,
    test: MorphTest,
    max_stack_depth: int,
    min_count: int = 1,
) -> tuple[dict[int, set[ConfigurationHistory]], dict[ConfigurationHistory, int]]:
    """CSSR homogenization over ``(suffix, stack)`` configurations.

    Every observed stack contributes a root ``((), stack)``; suffixes then grow one
    symbol into the past with their stack fixed, exactly as in flat CSSR. Growing
    forward from the empty configuration instead only reaches stacks of depth at
    most ``Lmax``, so deeper configurations had no state and their transitions were
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
    for length in range(Lmax):
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
    """Split homogeneous states until stack-lifted transitions are unifilar."""
    return _cssr_determinize(
        states,
        history_to_state,
        counts,
        length=length,
        successor_fn=_stack_successor_fn(alphabet=alphabet, length=length, max_stack_depth=max_stack_depth),
    )


def _stack_merge(
    states: dict[int, set[ConfigurationHistory]],
    history_to_state: dict[ConfigurationHistory, int],
    counts: StackSuffixCounts,
    *,
    alpha: float,
    test: MorphTest,
    alphabet: DyckAlphabet,
) -> dict[int, set[ConfigurationHistory]]:
    proxy = _control_counts(counts, alphabet).restricted_to(set(history_to_state))
    return _merge_similar_states(states, history_to_state, proxy, alpha=alpha, test=test)


def _stack_drop_transient(
    states: dict[int, set[ConfigurationHistory]],
    history_to_state: dict[ConfigurationHistory, int],
    counts: StackSuffixCounts,
    *,
    length: int,
    alphabet: DyckAlphabet,
    max_stack_depth: int,
) -> dict[int, set[ConfigurationHistory]]:
    proxy = counts.restricted_to(set(history_to_state))
    return _drop_transient_states(
        states,
        history_to_state,
        proxy,
        length=length,
        successor_fn=_stack_successor_fn(alphabet=alphabet, length=length, max_stack_depth=max_stack_depth),
    )


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
    stack: list[Any] = []
    for t in range(len(seq)):
        # Each step occupies one state: the one keyed by its longest available suffix.
        for hist_len in range(min(t, length), -1, -1):
            state = history_to_state.get((seq[t - hist_len : t], tuple(stack)))
            if state is not None:
                visits[state] += 1
                break
        symbol = seq[t]
        if symbol in alphabet.call_alphabet:
            if len(stack) >= max_stack_depth:
                stack = stack[1:]
            stack.append(symbol)
        elif symbol in alphabet.return_alphabet and stack:
            stack.pop()

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
            # The stack model renormalizes over the moves legal in each configuration, so
            # a symbol's weight is its frequency among the visits where it was legal.
            opportunities = sum(
                sum(counts.next_counts.get(history, Counter()).values())
                for history in histories
                if legal(symbol, history[1])
            )
            prob = emitted[symbol] / opportunities
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


def stack_cssr(
    sequence: Sequence[Any],
    *,
    alphabet: DyckAlphabet,
    Lmax: int | Literal["auto"] | None = None,
    max_stack_depth: int = 8,
    alpha: float = 0.05,
    test: MorphTest = "g",
    min_count: int = 5,
    correction: Literal["bonferroni"] | None = None,
) -> HiddenMarkovStackModel:
    """Reconstruct a stack HMM via configuration-lifted CSSR.

    ``Lmax="auto"`` uses :func:`~sofic.generators.epsilon_inference.suggest_lmax`
    on the observed symbols. Stack processes generally have infinite Markov
    order, so treat it as a lower bound on the suffix length the data support.
    ``test="exact"`` and ``correction="bonferroni"`` are as in
    :func:`~sofic.generators.epsilon_inference.cssr`; the correction counts
    eligible (suffix, stack) configurations.
    """
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    if Lmax == "auto":
        max_length = suggest_lmax(seq, alpha=alpha)
    else:
        max_length = Lmax if Lmax is not None else _cssr_default_lmax(len(seq), len(alphabet.symbol_alphabet))
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
        Lmax=max_length,
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


def stack_subtree_merge(
    sequence: Sequence[Any],
    *,
    alphabet: DyckAlphabet,
    L: int,
    max_stack_depth: int = 8,
    delta: float = 0.0,
) -> HiddenMarkovStackModel:
    """Reconstruct a stack HMM by merging depth-``L`` configuration subtrees."""
    if L < 0:
        raise ValueError("L must be non-negative")
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    counts = StackSuffixCounts.from_sequence(
        seq,
        alphabet=alphabet,
        max_length=L + 1,
        max_stack_depth=max_stack_depth,
    )
    histories = {history for history in counts.history_counts if len(history[0]) <= L}
    histories.add(((), ()))
    proxy = counts.restricted_to(histories)
    states = _cluster_histories_by_morph(proxy, histories, delta=delta)
    history_to_state = {history: state_id for state_id, members in states.items() for history in members}
    states = _stack_determinize(
        states,
        history_to_state,
        counts,
        length=L,
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
        states, history_to_state, counts, length=L, alphabet=alphabet, max_stack_depth=max_stack_depth
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
        length=L,
        max_stack_depth=max_stack_depth,
    )


def fit_stack_hmm_mle(
    shift: SoficDyckShift,
    sequence: Sequence[Any],
    *,
    smoothing: float = 1e-6,
) -> HiddenMarkovStackModel:
    """Assign MLE edge probabilities to a fixed Dyck topology from one sample."""
    from sofic.shifts.dyck_algorithms import _successors

    seq = tuple(sequence)
    edge_counts: Counter[TransitionRef] = Counter()
    states = tuple(shift.states())
    if not states:
        raise ValueError("shift has no states")
    state = states[0]
    stack: tuple[TransitionRef, ...] = ()

    for symbol in seq:
        matched = False
        for transition in shift.graph.out_transitions(state):
            if transition.data.get(ATTR_SYMBOL) != symbol:
                continue
            for target, next_stack in _successors(shift, transition, stack):
                edge_counts[transition_ref(transition)] += 1
                state = target
                stack = next_stack
                matched = True
                break
            if matched:
                break
        if not matched:
            break

    probabilities: dict[TransitionRef, float] = {}
    outgoing: dict[Hashable, list[TransitionRef]] = defaultdict(list)
    for transition in shift.transitions():
        ref = transition_ref(transition)
        outgoing[transition.source].append(ref)

    for refs in outgoing.values():
        total = sum(edge_counts.get(ref, 0.0) for ref in refs) + smoothing * len(refs)
        for ref in refs:
            count = edge_counts.get(ref, 0.0) + smoothing
            probabilities[ref] = count / total if total > 0 else 1.0 / len(refs)

    return HiddenMarkovStackModel.from_sofic_dyck_shift(shift, probabilities)


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
    return fit_stack_hmm_mle(shift, fit_source)
