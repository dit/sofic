"""Tests for tetromino randomizer ε-machines."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from sofic.examples import (
    TETROMINOES,
    tetris_bag,
    tetris_gameboy,
    tetris_history,
    tetris_iid,
    tetris_nes,
    tetris_tgm,
    tetris_tgm2,
)
from sofic.examples.tetris import _reroll_emission_probs
from sofic.generators.epsilon_machine import EpsilonMachine

_ALPHABET = frozenset(TETROMINOES)


def _history_order1() -> EpsilonMachine:
    return tetris_history(1, 2)


@pytest.mark.parametrize(
    "constructor, n_states",
    [
        (tetris_iid, 1),
        (tetris_nes, 7),
        (tetris_bag, 127),
        (tetris_gameboy, 49),
        (_history_order1, 7),
    ],
)
def test_small_randomizers_validate(constructor, n_states):
    machine = constructor()
    machine.validate()
    assert machine.is_unifilar()
    assert machine.observation_alphabet == _ALPHABET
    assert len(list(machine.states())) == n_states


def test_history_parameter_validation():
    with pytest.raises(ValueError, match="window"):
        tetris_history(window=0)
    with pytest.raises(ValueError, match="rolls"):
        tetris_history(rolls=0)


def test_iid_entropy_rate():
    pytest.importorskip("dit")
    assert tetris_iid().entropy_rate() == pytest.approx(math.log2(7))


def test_nes_transition_probs():
    nes = tetris_nes()
    edges = {(t.source, t.data["emission"], t.target): t.data["prob"] for t in nes.transitions()}
    for prev in TETROMINOES:
        assert edges[(prev, prev, prev)] == pytest.approx(1.0 / 28.0)
        others = [piece for piece in TETROMINOES if piece != prev]
        for piece in others:
            assert edges[(prev, piece, piece)] == pytest.approx(9.0 / 56.0)


def test_bag_singleton_refills():
    bag = tetris_bag()
    full = TETROMINOES
    edges = {(t.source, t.data["emission"], t.target): t.data["prob"] for t in bag.transitions()}
    singleton = ("I",)
    assert edges[(singleton, "I", full)] == pytest.approx(1.0)
    six = tuple(piece for piece in TETROMINOES if piece != "I")
    assert edges[(full, "I", six)] == pytest.approx(1.0 / 7.0)


def test_bag_entropy_rate_and_complexity():
    pytest.importorskip("dit")
    bag = tetris_bag()
    assert bag.entropy_rate() == pytest.approx(math.log2(math.factorial(7)) / 7.0)
    expected_cmu = math.log2(7) + (1.0 / 7.0) * sum(math.log2(math.comb(7, k)) for k in range(1, 8))
    assert bag.statistical_complexity() == pytest.approx(expected_cmu)


def test_history_avoids_window_pieces():
    history = ("I", "J", "L", "O")
    forbidden = set(history)
    tgm_probs = _reroll_emission_probs(TETROMINOES, 4, forbidden)
    tgm2_probs = _reroll_emission_probs(TETROMINOES, 6, forbidden)
    uniform = 1.0 / 7.0
    for piece in forbidden:
        assert tgm_probs[piece] < uniform
        assert tgm2_probs[piece] < tgm_probs[piece]


@pytest.fixture(scope="module")
def tgm_machine():
    return tetris_tgm()


@pytest.fixture(scope="module")
def tgm2_machine():
    return tetris_tgm2()


def test_tgm_validates(tgm_machine):
    tgm_machine.validate()
    assert tgm_machine.is_unifilar()
    assert tgm_machine.observation_alphabet == _ALPHABET
    assert len(list(tgm_machine.states())) == 7**4


def test_tgm_history_emissions_match_formula(tgm_machine, tgm2_machine):
    history = ("I", "J", "L", "O")
    forbidden = set(history)
    tgm_edges = {t.data["emission"]: t.data["prob"] for t in tgm_machine.transitions() if t.source == history}
    tgm2_edges = {t.data["emission"]: t.data["prob"] for t in tgm2_machine.transitions() if t.source == history}
    expected_tgm = _reroll_emission_probs(TETROMINOES, 4, forbidden)
    expected_tgm2 = _reroll_emission_probs(TETROMINOES, 6, forbidden)
    for piece in TETROMINOES:
        assert tgm_edges[piece] == pytest.approx(expected_tgm[piece])
        assert tgm2_edges[piece] == pytest.approx(expected_tgm2[piece])
        if piece in forbidden:
            assert tgm_edges[piece] < 1.0 / 7.0
            assert tgm2_edges[piece] < tgm_edges[piece]


def test_gameboy_l_is_rarest():
    gb = tetris_gameboy()
    gb.validate()
    mass: Counter[str] = Counter()
    pi = gb.initial_distribution
    for transition in gb.transitions():
        mass[transition.data["emission"]] += pi[transition.source] * transition.data["prob"]
    total = sum(mass.values())
    freqs = {piece: mass[piece] / total for piece in TETROMINOES}
    assert min(freqs, key=freqs.get) == "L"
    for common in ("O", "S", "T"):
        assert freqs[common] > freqs["L"]
