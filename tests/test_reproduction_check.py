"""Word-process verification of candidate generators beyond a fixed word length."""

from __future__ import annotations

import numpy as np

from sofic.examples import golden_mean
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators.mealy import MealyHMM
from sofic.generators.minimal_generative_model import _equivalence_error, _reproduction_error


def _fair_coin():
    return from_symbol_matrices((0,), (0, 1), {0: np.array([[0.5]]), 1: np.array([[0.5]])})


def _alternating():
    return from_symbol_matrices(
        (0, 1), (0, 1), {0: np.array([[0.0, 1.0], [0.0, 0.0]]), 1: np.array([[0.0, 0.0], [1.0, 0.0]])}
    )


def test_processes_equal_on_single_symbols_are_told_apart() -> None:
    # both emit each symbol with probability 1/2; they differ first on words of length 2
    coin, alternating = _fair_coin(), _alternating()
    assert _equivalence_error(coin, alternating) >= 0.25 - 1e-12
    assert _reproduction_error(coin, alternating, max_length=1) >= 0.25 - 1e-12


def test_presentations_of_one_process_agree() -> None:
    g = golden_mean(0.5)
    assert _equivalence_error(g, g) <= 1e-12
    # a non-unifilar presentation of the golden mean: state A split into two copies
    split = MealyHMM(
        observation_alphabet=frozenset({"0", "1"}), initial_distribution={"A1": 1 / 3, "A2": 1 / 3, "B": 1 / 3}
    )
    for state in ("A1", "A2", "B"):
        split.graph.add_state(state)
    for source in ("A1", "A2"):
        split.add_transition(source, "A1", "0", 0.25)
        split.add_transition(source, "A2", "0", 0.25)
        split.add_transition(source, "B", "1", 0.5)
    split.add_transition("B", "A1", "0", 0.5)
    split.add_transition("B", "A2", "0", 0.5)
    assert _equivalence_error(split, g) <= 1e-12
