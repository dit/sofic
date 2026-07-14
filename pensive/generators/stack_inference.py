"""Sample-based inference for hidden Markov stack models."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any, ClassVar, Literal

from pensive.automata.papni import DyckAlphabet, is_well_matched, learn_sofic_dyck_shift_papni
from pensive.exceptions import StochasticValidationError
from pensive.generators.epsilon_inference import (
    History,
    SuffixCounts,
    _cluster_histories_by_morph,
    _cssr_determinize,
    _cssr_homogenize,
    _default_lmax,
    _drop_transient_states,
    _merge_similar_states,
)
from pensive.generators.stack_hmm import HiddenMarkovStackModel
from pensive.graph import ATTR_SYMBOL
from pensive.shifts.sofic_dyck import SoficDyckShift, TransitionRef, transition_ref

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


def _stack_homogenize(
    counts: StackSuffixCounts,
    *,
    alphabet: DyckAlphabet,
    Lmax: int,
    alpha: float,
    test: Literal["g", "chi2", "tv"],
    max_stack_depth: int,
) -> tuple[dict[int, set[ConfigurationHistory]], dict[ConfigurationHistory, int]]:
    return _cssr_homogenize(
        counts,
        Lmax=Lmax,
        alpha=alpha,
        test=test,
        successor_fn=_stack_successor_fn(alphabet=alphabet, length=Lmax, max_stack_depth=max_stack_depth),
    )


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
    test: Literal["g", "chi2", "tv"],
) -> dict[int, set[ConfigurationHistory]]:
    proxy = counts.restricted_to(set(history_to_state))
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
    return _drop_transient_states(states, history_to_state, proxy, length=length)


def _representative_stack(histories: set[ConfigurationHistory]) -> tuple[Any, ...]:
    stacks = [stack for _suffix, stack in histories if stack]
    if not stacks:
        return ()
    return max(stacks, key=len)


def _counts_to_stack_hmm(
    states: dict[int, set[ConfigurationHistory]],
    counts: StackSuffixCounts,
    history_to_state: dict[ConfigurationHistory, int],
    sequence: Sequence[Any],
    *,
    alphabet: DyckAlphabet,
    length: int,
) -> HiddenMarkovStackModel:
    visits: Counter[int] = Counter()
    seq = tuple(sequence)
    stack: list[Any] = []
    for t in range(len(seq)):
        for hist_len in range(0, min(t, length) + 1):
            suffix = seq[t - hist_len : t]
            state = history_to_state.get((suffix, tuple(stack)))
            if state is not None:
                visits[state] += 1
        symbol = seq[t]
        if symbol in alphabet.call_alphabet:
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

    call_refs: dict[tuple[Hashable, Any], TransitionRef] = {}
    return_refs: dict[tuple[Hashable, Any, Any], TransitionRef] = {}

    for state_id, histories in states.items():
        source = state_labels[state_id]
        stack_repr = _representative_stack(histories)
        stack_tops = {stack[-1] for _suffix, stack in histories if stack}
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
            child_histories = {
                _successor_history(
                    history,
                    symbol,
                    alphabet=alphabet,
                    length=length,
                    max_stack_depth=max(len(stack_repr), 1),
                )
                for history in emitting
            }
            targets = {history_to_state.get(child) for child in child_histories}
            targets.discard(None)
            if not targets:
                continue
            if len(targets) > 1:
                target_counts: Counter[int] = Counter()
                for history in emitting:
                    child = _successor_history(
                        history,
                        symbol,
                        alphabet=alphabet,
                        length=length,
                        max_stack_depth=max(len(stack_repr), 1),
                    )
                    target_id = history_to_state.get(child)
                    if target_id is not None:
                        target_counts[target_id] += counts.history_counts.get(history, 0)
                target_id = target_counts.most_common(1)[0][0]
            else:
                target_id = next(iter(targets))
            target = state_labels[target_id]

            if symbol in alphabet.call_alphabet:
                key = (source, symbol, target)
                if key not in call_refs:
                    call_refs[key] = model.add_call_transition(source, target, symbol, prob)
            elif symbol in alphabet.return_alphabet:
                call_candidates = stack_tops or frozenset(alphabet.call_alphabet)
                for matched_call in call_candidates:
                    key = (source, symbol, matched_call)
                    if key not in return_refs:
                        return_refs[key] = model.add_return_transition(source, target, symbol, prob)
            else:
                model.add_internal_transition(source, target, symbol, prob)

    for (_src, call_symbol, _target), call_ref in call_refs.items():
        for (_ret_source, _return_symbol, matched_call), return_ref in return_refs.items():
            if matched_call == call_symbol:
                model.add_matched_pair(call_ref, return_ref)

    for state_id, histories in states.items():
        source = state_labels[state_id]
        for history in histories:
            _suffix, stack = history
            if not stack:
                continue
            for symbol in alphabet.return_alphabet:
                if counts.next_counts.get(history, Counter()).get(symbol, 0) <= 0:
                    continue
                matched_call = stack[-1]
                for (src, sym, _tgt), call_ref in call_refs.items():
                    if src != source or sym != matched_call:
                        continue
                    for (rsrc, rsym, mc), return_ref in return_refs.items():
                        if rsrc == source and rsym == symbol and mc == matched_call:
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
    Lmax: int | None = None,
    max_stack_depth: int = 8,
    alpha: float = 0.05,
    test: Literal["g", "chi2", "tv"] = "g",
    min_count: int = 5,
) -> HiddenMarkovStackModel:
    """Reconstruct a stack HMM via configuration-lifted CSSR."""
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    max_length = Lmax if Lmax is not None else _default_lmax(len(seq), len(alphabet.symbol_alphabet), min_count)
    counts = StackSuffixCounts.from_sequence(
        seq,
        alphabet=alphabet,
        max_length=max_length + 1,
        max_stack_depth=max_stack_depth,
    )
    states, history_to_state = _stack_homogenize(
        counts,
        alphabet=alphabet,
        Lmax=max_length,
        alpha=alpha,
        test=test,
        max_stack_depth=max_stack_depth,
    )
    states = _stack_determinize(
        states,
        history_to_state,
        counts,
        length=max_length,
        alphabet=alphabet,
        max_stack_depth=max_stack_depth,
    )
    states = _stack_merge(states, history_to_state, counts, alpha=alpha, test=test)
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
    states = _stack_merge(states, history_to_state, counts, alpha=0.05, test="tv")
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
    )


def fit_stack_hmm_mle(
    shift: SoficDyckShift,
    sequence: Sequence[Any],
    *,
    smoothing: float = 1e-6,
) -> HiddenMarkovStackModel:
    """Assign MLE edge probabilities to a fixed Dyck topology from one sample."""
    from pensive.shifts.dyck_algorithms import _successors

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
