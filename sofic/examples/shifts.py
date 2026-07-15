"""Constructors for canonical symbolic-shift examples."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from sofic.shifts.sofic_dyck import SoficDyckShift


def dyck_shift_order(
    k: int = 2,
    *,
    call_symbols: Sequence[Hashable] | None = None,
    return_symbols: Sequence[Hashable] | None = None,
) -> SoficDyckShift:
    """Dyck shift of order ``k`` from Beal, Blockelet & Dima, Example 1.

    The default symbols follow the paper's notation:
    ``A_c = {a1, ..., ak}`` and ``A_r = {b1, ..., bk}``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    calls = tuple(call_symbols) if call_symbols is not None else tuple(f"a{i}" for i in range(1, k + 1))
    returns = tuple(return_symbols) if return_symbols is not None else tuple(f"b{i}" for i in range(1, k + 1))
    if len(calls) != k or len(returns) != k:
        raise ValueError("call_symbols and return_symbols must have length k")

    return _one_state_dyck_shift(calls, returns, ())


def motzkin_shift(
    call_symbols: Sequence[Hashable] = ("(", "["),
    return_symbols: Sequence[Hashable] = (")", "]"),
    internal_symbols: Sequence[Hashable] = ("i",),
) -> SoficDyckShift:
    """Motzkin shift shown on the left of Beal, Blockelet & Dima, Fig. 1."""
    return _one_state_dyck_shift(tuple(call_symbols), tuple(return_symbols), tuple(internal_symbols))


def sofic_dyck_fig1_shift() -> SoficDyckShift:
    """Two-state sofic-Dyck shift shown on the right of Beal, Blockelet & Dima, Fig. 1."""
    shift = SoficDyckShift(
        call_alphabet=frozenset({"(", "["}),
        return_alphabet=frozenset({")", "]"}),
        internal_alphabet=frozenset({"i"}),
    )
    shift.graph.add_state("1")
    shift.graph.add_state("2")

    left_paren = shift.add_call_transition("1", "1", "(")
    left_bracket = shift.add_call_transition("1", "1", "[")
    right_paren = shift.add_return_transition("1", "1", ")")
    right_bracket = shift.add_return_transition("1", "1", "]")
    shift.add_internal_transition("1", "2", "i")
    shift.add_internal_transition("2", "1", "i")

    shift.add_matched_pair(left_paren, right_paren)
    shift.add_matched_pair(left_bracket, right_bracket)
    return shift


def sofic_dyck_nondeterminizable_shift() -> SoficDyckShift:
    """Sofic-Dyck shift with no deterministic presentation from Beal et al., Fig. 3."""
    shift = SoficDyckShift(
        call_alphabet=frozenset({"a"}),
        return_alphabet=frozenset({"b"}),
        internal_alphabet=frozenset({"i", "j", "k"}),
    )
    for state in ("1", "2", "3"):
        shift.graph.add_state(state)

    call = shift.add_call_transition("1", "1", "a")
    return_2 = shift.add_return_transition("2", "2", "b")
    shift.add_return_transition("3", "3", "b")
    shift.add_internal_transition("1", "2", "i")
    shift.add_internal_transition("2", "1", "j")
    shift.add_internal_transition("1", "3", "i")
    shift.add_internal_transition("3", "1", "k")

    shift.add_matched_pair(call, return_2)
    return shift


def sofic_dyck_zeta_example_shift() -> SoficDyckShift:
    """Sofic-Dyck shift used in Beal, Blockelet & Dima's zeta-function example."""
    shift = SoficDyckShift(
        call_alphabet=frozenset({"a", "a'"}),
        return_alphabet=frozenset({"b", "b'"}),
        internal_alphabet=frozenset({"i"}),
    )
    shift.graph.add_state("1")
    shift.graph.add_state("2")

    return_b = shift.add_return_transition("1", "1", "b")
    call_a = shift.add_call_transition("1", "1", "a")
    call_a_prime = shift.add_call_transition("1", "1", "a'")
    return_b_prime = shift.add_return_transition("1", "1", "b'")
    shift.add_internal_transition("1", "2", "i")
    shift.add_internal_transition("2", "1", "i")

    shift.add_matched_pair(call_a, return_b)
    shift.add_matched_pair(call_a_prime, return_b_prime)
    return shift


def _one_state_dyck_shift(
    call_symbols: tuple[Hashable, ...],
    return_symbols: tuple[Hashable, ...],
    internal_symbols: tuple[Hashable, ...],
) -> SoficDyckShift:
    if len(call_symbols) != len(return_symbols):
        raise ValueError("call_symbols and return_symbols must have the same length")
    _require_disjoint(call_symbols, return_symbols, internal_symbols)

    shift = SoficDyckShift(
        call_alphabet=frozenset(call_symbols),
        return_alphabet=frozenset(return_symbols),
        internal_alphabet=frozenset(internal_symbols),
    )
    state = "1"
    shift.graph.add_state(state)

    call_refs = {symbol: shift.add_call_transition(state, state, symbol) for symbol in call_symbols}
    return_refs = {symbol: shift.add_return_transition(state, state, symbol) for symbol in return_symbols}
    for symbol in internal_symbols:
        shift.add_internal_transition(state, state, symbol)
    for call_symbol, return_symbol in zip(call_symbols, return_symbols, strict=True):
        shift.add_matched_pair(call_refs[call_symbol], return_refs[return_symbol])
    return shift


def _require_disjoint(*symbol_groups: tuple[Hashable, ...]) -> None:
    symbols = tuple(symbol for group in symbol_groups for symbol in group)
    if len(frozenset(symbols)) != len(symbols):
        raise ValueError("call, return, and internal symbols must be distinct")


__all__ = [
    "dyck_shift_order",
    "motzkin_shift",
    "sofic_dyck_fig1_shift",
    "sofic_dyck_nondeterminizable_shift",
    "sofic_dyck_zeta_example_shift",
]
