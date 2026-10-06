"""Tests for Fischer and Krieger covers."""

import networkx as nx
import pytest

from sofic.exceptions import SoficValidationError
from sofic.graph import ATTR_SYMBOL
from sofic.shifts.covers import LeftFischerCover, LeftKriegerCover, RightFischerCover, RightKriegerCover
from sofic.shifts.sofic import SoficShift


def _shift(edges, alphabet=("0", "1")) -> SoficShift:
    shift = SoficShift(symbol_alphabet=frozenset(alphabet))
    for source, target, symbol in edges:
        shift.graph.add_state(source)
        shift.graph.add_state(target)
        shift.graph.add_transition(source, target, **{ATTR_SYMBOL: symbol})
    return shift


def _golden_mean() -> SoficShift:
    return _shift([("A", "B", "1"), ("B", "A", "0"), ("B", "B", "0"), ("A", "A", "0")])


def _even_shift() -> SoficShift:
    """Runs of 1s between 0s have even length; Krieger cover has three vertices."""
    return _shift([("A", "A", "0"), ("A", "B", "1"), ("B", "A", "1")])


def _nondeterministic_even_shift() -> SoficShift:
    # Two copies of the even-shift presentation glued nondeterministically.
    return _shift(
        [
            ("A", "A", "0"),
            ("A", "B", "1"),
            ("B", "A", "1"),
            ("A", "C", "0"),
            ("C", "D", "1"),
            ("D", "C", "1"),
            ("C", "A", "0"),
        ]
    )


def _language(shift: SoficShift, max_length: int = 8) -> set[tuple]:
    return {word for n in range(max_length + 1) for word in shift.factor_language(n)}


def _is_left_resolving(shift: SoficShift) -> bool:
    return SoficShift(graph=shift.graph.reverse(), symbol_alphabet=shift.symbol_alphabet).is_unifilar()


@pytest.mark.parametrize("builder", [_golden_mean, _even_shift, _nondeterministic_even_shift])
def test_right_fischer_cover_is_minimal_right_resolving_and_presents_shift(builder):
    shift = builder()
    cover = RightFischerCover.from_sofic(shift)
    cover.validate()
    assert cover.is_unifilar()
    assert nx.is_strongly_connected(cover.graph.nx)
    assert _language(cover) == _language(shift)
    assert len(list(cover.states())) == 2


@pytest.mark.parametrize("builder", [_golden_mean, _even_shift, _nondeterministic_even_shift])
def test_left_fischer_cover_is_left_resolving_and_presents_shift(builder):
    shift = builder()
    cover = LeftFischerCover.from_sofic(shift)
    assert _is_left_resolving(cover)
    assert _language(cover) == _language(shift)


def test_even_shift_krieger_cover_has_three_vertices_and_contains_fischer_cover():
    shift = _even_shift()
    krieger = RightKriegerCover.from_sofic(shift)
    fischer = RightFischerCover.from_sofic(shift)
    assert krieger.is_unifilar()
    assert len(list(krieger.states())) == 3
    assert _language(krieger) == _language(shift)
    condensation = nx.condensation(krieger.graph.nx)
    terminal = [n for n in condensation.nodes if condensation.out_degree(n) == 0]
    assert len(terminal) == 1
    assert len(condensation.nodes[terminal[0]]["members"]) == len(list(fischer.states()))


def test_golden_mean_krieger_cover_equals_fischer_cover_size():
    shift = _golden_mean()
    assert len(list(RightKriegerCover.from_sofic(shift).states())) == 2
    assert len(list(LeftKriegerCover.from_sofic(shift).states())) == 2


@pytest.mark.parametrize("builder", [_golden_mean, _even_shift, _nondeterministic_even_shift])
def test_left_krieger_cover_is_left_resolving_and_presents_shift(builder):
    shift = builder()
    cover = LeftKriegerCover.from_sofic(shift)
    assert _is_left_resolving(cover)
    assert _language(cover) == _language(shift)


def test_fischer_cover_rejects_reducible_shift():
    reducible = _shift([("A", "A", "0"), ("B", "B", "1")])
    with pytest.raises(SoficValidationError, match="irreducible"):
        RightFischerCover.from_sofic(reducible)
    krieger = RightKriegerCover.from_sofic(reducible)
    assert _language(krieger) == _language(reducible)
