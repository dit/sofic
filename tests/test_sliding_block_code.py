"""Tests for sliding block codes (factor maps)."""

from itertools import product

import pytest

from sofic import SoficShift
from sofic.shifts.sliding_block_code import SlidingBlockCode, full_shift


def _nor() -> SlidingBlockCode:
    return SlidingBlockCode(
        {("0", "0"): "1", ("0", "1"): "0", ("1", "0"): "0", ("1", "1"): "0"},
        memory=1,
        anticipation=0,
    )


def _flip() -> SlidingBlockCode:
    return SlidingBlockCode({("0",): "1", ("1",): "0"}, memory=0, anticipation=0)


def test_apply_word():
    assert _nor().apply_word(list("0010")) == ("1", "0", "0")
    assert _flip().apply_word(list("011")) == ("1", "0", "0")


def test_bad_window_key_rejected():
    with pytest.raises(ValueError):
        SlidingBlockCode({("0", "0"): "1"}, memory=0, anticipation=0)


def test_apply_produces_sofic_shift():
    image = _nor().apply(full_shift({"0", "1"}))
    assert isinstance(image, SoficShift)
    assert image.symbol_alphabet <= frozenset({"0", "1"})


def test_apply_matches_pointwise_image():
    code = _nor()
    shift = full_shift({"0", "1"})
    image = code.apply(shift)
    length = 3
    expected = {code.apply_word(word) for word in product("01", repeat=length + code.window - 1)}
    observed = set(image.factor_language(length))
    assert observed == expected


def test_compose_is_function_composition():
    flip = _flip()
    identity = flip.compose(flip)
    assert identity.apply_word(["0"]) == ("0",)
    assert identity.apply_word(["1"]) == ("1",)


def test_to_transducer_realizes_code():
    flip = _flip()
    machine = flip.to_transducer()
    machine.validate()
    assert machine.transduce(("0", "1")) == {("1", "0")}


def test_memoryless_transducer_round_trip():
    from sofic.examples.processes import BitFlip

    code = BitFlip().to_sliding_block_code()
    assert code.memory == 0
    assert code.apply_word(["0"]) == ("1",)


def _golden_mean_shift() -> SoficShift:
    shift = SoficShift(symbol_alphabet=frozenset({"0", "1"}))
    shift.graph.add_state("A")
    shift.graph.add_state("B")
    shift.add_transition("A", "A", "0")
    shift.add_transition("A", "B", "1")
    shift.add_transition("B", "A", "0")
    return shift


def test_apply_keeps_constraints_longer_than_the_window():
    identity = SlidingBlockCode({("0",): "0", ("1",): "1"})
    shift = _golden_mean_shift()
    image = identity.apply(shift)
    for length in range(1, 7):
        assert set(image.factor_language(length)) == set(shift.factor_language(length))


def test_apply_rejects_partial_block_map():
    partial = SlidingBlockCode({("0",): "0"}, input_alphabet={"0", "1"})
    with pytest.raises(ValueError, match="not in block_map"):
        partial.apply(_golden_mean_shift())
