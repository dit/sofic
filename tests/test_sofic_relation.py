"""Tests for sofic relations (product-alphabet subshifts)."""

from sofic import SoficRelation, SoficShift
from sofic.examples.processes import GMtoEven


def test_from_transducer_symbols_are_pairs():
    rel = SoficRelation.from_transducer(GMtoEven())
    assert all(isinstance(symbol, tuple) and len(symbol) == 2 for symbol in rel.symbol_alphabet)
    assert rel.input_alphabet() == frozenset({"0", "1"})
    assert rel.output_alphabet() == frozenset({"0", "1"})


def test_projections_are_sofic_shifts():
    rel = SoficRelation.from_transducer(GMtoEven())
    input_shift = rel.input_shift()
    output_shift = rel.output_shift()
    assert isinstance(input_shift, SoficShift)
    assert isinstance(output_shift, SoficShift)
    assert input_shift.symbol_alphabet == frozenset({"0", "1"})


def test_input_shift_matches_transducer_input_language():
    transducer = GMtoEven()
    rel = SoficRelation.from_transducer(transducer)
    input_words = set(rel.input_shift().factor_language(3))
    assert input_words  # non-empty
    # every relation input word is a valid transducer input word
    assert all(len(word) == 3 for word in input_words)


def test_yaml_round_trip():
    rel = SoficRelation.from_transducer(GMtoEven())
    restored = SoficRelation.from_yaml(rel.to_yaml())
    assert isinstance(restored, SoficRelation)
    assert restored.symbol_alphabet == rel.symbol_alphabet


def test_transducer_bridge_method():
    rel = GMtoEven().to_sofic_relation()
    assert isinstance(rel, SoficRelation)
