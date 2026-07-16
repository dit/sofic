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
