"""Passive visibly-pushdown topology learning via PAPNI preprocessing and RPNI."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pensive.automata.dfa import DFA
from pensive.automata.rpni import learn_dfa_rpni
from pensive.graph import ATTR_KIND, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN
from pensive.shifts.sofic_dyck import SoficDyckShift, TransitionRef, transition_ref

__all__ = [
    "DyckAlphabet",
    "is_well_matched",
    "learn_sofic_dyck_shift_papni",
    "papni_encode",
    "papni_encode_samples",
    "sofic_dyck_shift_from_papni_dfa",
]


@dataclass(frozen=True, slots=True)
class DyckAlphabet:
    """Partition of symbols into call, return, and internal roles."""

    call_alphabet: frozenset[Any]
    return_alphabet: frozenset[Any]
    internal_alphabet: frozenset[Any]

    def __post_init__(self) -> None:
        union = self.call_alphabet | self.return_alphabet | self.internal_alphabet
        if len(union) != len(self.call_alphabet) + len(self.return_alphabet) + len(self.internal_alphabet):
            raise ValueError("call, return, and internal alphabets must be disjoint")

    @property
    def symbol_alphabet(self) -> frozenset[Any]:
        return self.call_alphabet | self.return_alphabet | self.internal_alphabet

    def classify(self, symbol: Any) -> str:
        if symbol in self.call_alphabet:
            return KIND_CALL
        if symbol in self.return_alphabet:
            return KIND_RETURN
        if symbol in self.internal_alphabet:
            return KIND_INTERNAL
        raise ValueError(f"symbol {symbol!r} not in Dyck alphabet")


def is_well_matched(word: Sequence[Any], alphabet: DyckAlphabet) -> bool:
    """Return whether ``word`` is well-matched under a symbol counter (PAPNI Alg. 1)."""
    counter = 0
    for symbol in word:
        if symbol in alphabet.call_alphabet:
            counter += 1
        elif symbol in alphabet.return_alphabet:
            counter -= 1
            if counter < 0:
                return False
        elif symbol not in alphabet.internal_alphabet:
            return False
    return counter == 0


def papni_encode(word: Sequence[Any], alphabet: DyckAlphabet) -> tuple[Any, ...]:
    """Convert a well-matched word to its stack-aware representation (PAPNI Alg. 2)."""
    if not is_well_matched(word, alphabet):
        raise ValueError("word is not well-matched")

    encoded: list[Any] = []
    stack: list[Any] = []
    for symbol in word:
        if symbol in alphabet.call_alphabet:
            stack.append(symbol)
            encoded.append(symbol)
        elif symbol in alphabet.return_alphabet:
            if not stack:
                raise ValueError("return with empty stack")
            call_symbol = stack.pop()
            encoded.append((symbol, call_symbol))
        else:
            encoded.append(symbol)
    return tuple(encoded)


def papni_encode_samples(
    samples: Sequence[Sequence[Any]],
    alphabet: DyckAlphabet,
    *,
    drop_non_well_matched: bool = True,
) -> list[tuple[Any, ...]]:
    """Filter and encode samples for RPNI over the stack-aware alphabet."""
    encoded: list[tuple[Any, ...]] = []
    for word in samples:
        seq = tuple(word)
        if not is_well_matched(seq, alphabet):
            if drop_non_well_matched:
                continue
            raise ValueError(f"sample {seq!r} is not well-matched")
        encoded.append(papni_encode(seq, alphabet))
    return encoded


def _stack_aware_alphabet(alphabet: DyckAlphabet) -> frozenset[Any]:
    symbols: set[Any] = set(alphabet.internal_alphabet) | set(alphabet.call_alphabet)
    for return_symbol in alphabet.return_alphabet:
        for call_symbol in alphabet.call_alphabet:
            symbols.add((return_symbol, call_symbol))
    return frozenset(symbols)


def _decode_symbol(symbol: Any, alphabet: DyckAlphabet) -> tuple[str, Any, Any | None]:
    if symbol in alphabet.call_alphabet:
        return KIND_CALL, symbol, None
    if symbol in alphabet.internal_alphabet:
        return KIND_INTERNAL, symbol, None
    if isinstance(symbol, tuple) and len(symbol) == 2:
        return_symbol, call_symbol = symbol
        if return_symbol in alphabet.return_alphabet and call_symbol in alphabet.call_alphabet:
            return KIND_RETURN, return_symbol, call_symbol
    raise ValueError(f"unknown stack-aware symbol {symbol!r}")


def sofic_dyck_shift_from_papni_dfa(
    dfa: DFA,
    alphabet: DyckAlphabet,
    *,
    positive_traces: Sequence[Sequence[Any]] | None = None,
) -> SoficDyckShift:
    """Convert a PAPNI-learned DFA over stack-aware symbols into a ``SoficDyckShift``."""
    if not dfa.initial_states:
        raise ValueError("DFA requires an initial state")
    initial = next(iter(dfa.initial_states))
    reachable = _reachable_dfa_states(dfa, initial)

    shift = SoficDyckShift(
        call_alphabet=alphabet.call_alphabet,
        return_alphabet=alphabet.return_alphabet,
        internal_alphabet=alphabet.internal_alphabet,
    )

    for state in sorted(reachable, key=repr):
        shift.graph.add_state(state)

    call_by_symbol: dict[Any, list[TransitionRef]] = {symbol: [] for symbol in alphabet.call_alphabet}
    return_by_pair: dict[tuple[Any, Any], list[TransitionRef]] = {}

    for source in reachable:
        for transition in dfa.graph.out_transitions(source):
            symbol = transition.data.get(ATTR_SYMBOL)
            if symbol is None:
                continue
            kind, visible, matched_call = _decode_symbol(symbol, alphabet)
            target = transition.target
            if kind == KIND_CALL:
                ref = shift.add_call_transition(source, target, visible)
                call_by_symbol[visible].append(ref)
            elif kind == KIND_INTERNAL:
                shift.add_internal_transition(source, target, visible)
            else:
                assert matched_call is not None
                ref = shift.add_return_transition(source, target, visible)
                return_by_pair.setdefault((visible, matched_call), []).append(ref)

    if positive_traces:
        _infer_matched_edges_from_traces(shift, dfa, alphabet, positive_traces)
    else:
        for (_return_symbol, call_symbol), return_refs in return_by_pair.items():
            for return_ref in return_refs:
                for call_ref in call_by_symbol.get(call_symbol, ()):
                    shift.add_matched_pair(call_ref, return_ref)

    shift = _trim_shift_to_reachable(shift, initial)
    shift.validate()
    return shift


def _trim_shift_to_reachable(shift: SoficDyckShift, initial: Hashable) -> SoficDyckShift:
    """Drop unreachable states and zero-outdegree control states."""
    if not shift.graph.has_state(initial):
        return shift
    active: set[Hashable] = {initial}
    queue = [initial]
    while queue:
        state = queue.pop(0)
        for transition in shift.graph.out_transitions(state):
            if transition.target not in active:
                active.add(transition.target)
                queue.append(transition.target)

    dead = {state for state in active if not any(True for _ in shift.graph.out_transitions(state))}
    active -= dead

    trimmed = SoficDyckShift(
        call_alphabet=shift.call_alphabet,
        return_alphabet=shift.return_alphabet,
        internal_alphabet=shift.internal_alphabet,
        matched_edges=shift.matched_edges,
        symbol_alphabet=shift.symbol_alphabet,
    )
    for state in active:
        trimmed.graph.add_state(state, **shift.graph.state_attrs(state))

    edge_map: dict[TransitionRef, TransitionRef] = {}
    for transition in shift.transitions():
        if transition.source not in active:
            continue
        target = transition.target if transition.target in active else transition.source
        data = dict(transition.data)
        key = trimmed.graph.add_transition(transition.source, target, **data)
        edge_map[transition_ref(transition)] = (transition.source, target, key)

    trimmed.matched_edges = frozenset(
        (edge_map[call_ref], edge_map[return_ref])
        for call_ref, return_ref in shift.matched_edges
        if call_ref in edge_map and return_ref in edge_map
    )
    return trimmed


def _infer_matched_edges_from_traces(
    shift: SoficDyckShift,
    dfa: DFA,
    alphabet: DyckAlphabet,
    traces: Sequence[Sequence[Any]],
) -> None:
    """Record matched call-return pairs observed when replaying positive traces."""
    if not dfa.initial_states:
        return
    initial = next(iter(dfa.initial_states))

    for word in traces:
        if not is_well_matched(word, alphabet):
            continue
        encoded = papni_encode(word, alphabet)
        dfa_state = initial
        config_state = initial
        stack: list[TransitionRef] = []

        for symbol in encoded:
            kind, visible, matched_call = _decode_symbol(symbol, alphabet)
            successors = list(dfa.graph.out_transitions(dfa_state))
            dfa_transition = next(
                (transition for transition in successors if transition.data.get(ATTR_SYMBOL) == symbol),
                None,
            )
            if dfa_transition is None:
                break
            dfa_state = dfa_transition.target

            if kind == KIND_CALL:
                ref = _select_transition(shift, config_state, dfa_state, KIND_CALL, visible)
                if ref is None:
                    break
                stack.append(ref)
                config_state = dfa_state
            elif kind == KIND_INTERNAL:
                ref = _select_transition(shift, config_state, dfa_state, KIND_INTERNAL, visible)
                if ref is None:
                    break
                config_state = dfa_state
            else:
                assert matched_call is not None
                ref = _select_transition(shift, config_state, dfa_state, KIND_RETURN, visible)
                if ref is None or not stack:
                    break
                call_ref = stack.pop()
                shift.add_matched_pair(call_ref, ref)
                config_state = dfa_state


def _select_transition(
    shift: SoficDyckShift,
    source: Hashable,
    target: Hashable,
    kind: str,
    symbol: Any,
) -> TransitionRef | None:
    for transition in shift.graph.out_transitions(source):
        if transition.target != target:
            continue
        if transition.data.get(ATTR_KIND) != kind:
            continue
        if transition.data.get(ATTR_SYMBOL) != symbol:
            continue
        return transition_ref(transition)
    return None


def _reachable_dfa_states(dfa: DFA, initial: Hashable) -> set[Hashable]:
    seen = {initial}
    queue = [initial]
    while queue:
        state = queue.pop(0)
        for transition in dfa.graph.out_transitions(state):
            if transition.target not in seen:
                seen.add(transition.target)
                queue.append(transition.target)
    return seen


def learn_sofic_dyck_shift_papni(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]] | None = None,
    *,
    alphabet: DyckAlphabet,
) -> SoficDyckShift:
    """Learn a ``SoficDyckShift`` topology from labeled samples via PAPNI + RPNI."""
    encoded_positive = papni_encode_samples(positive, alphabet)
    if not encoded_positive:
        raise ValueError("no well-matched positive samples remain after PAPNI filtering")

    encoded_negative: list[tuple[Any, ...]] = []
    if negative:
        for word in negative:
            seq = tuple(word)
            if not is_well_matched(seq, alphabet):
                continue
            encoded_negative.append(papni_encode(seq, alphabet))

    dfa = learn_dfa_rpni(encoded_positive, encoded_negative)
    well_matched_positive = [tuple(word) for word in positive if is_well_matched(word, alphabet)]
    return sofic_dyck_shift_from_papni_dfa(
        dfa,
        alphabet,
        positive_traces=well_matched_positive,
    )
