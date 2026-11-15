"""ε-machines of processes whose hidden state no finite word reveals."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.minimal_generative_model import _equivalence_error
from sofic.generators.reversal import time_reverse_stochastic


def _paired_bits(e: float):
    """Period 2: a fair bit, then a copy of it flipped with probability ``e``; the phase is never certain."""
    mats = {
        0: np.array([[0, 0.5, 0], [1 - e, 0, 0], [e, 0, 0]]),
        1: np.array([[0, 0, 0.5], [e, 0, 0], [1 - e, 0, 0]]),
    }
    return from_symbol_matrices(("A", "B0", "B1"), (0, 1), mats)


# 0.4: rounding closes the mixed-state presentation on hundreds of near-limit beliefs;
# 0.1: the mixed-state presentation exceeds ``max_states`` first.
@pytest.mark.parametrize("e", [0.4, 0.1])
def test_reverse_epsilon_machine_of_asymptotically_synchronizing_process(e):
    forward = _paired_bits(e)
    reversed_hmm = time_reverse_stochastic(forward)

    reverse = EpsilonMachine.from_hmm(reversed_hmm)

    assert len(list(reverse.states())) == 3
    assert float(reverse.statistical_complexity()) == pytest.approx(1.5)
    assert _equivalence_error(reverse, reversed_hmm) < 1e-12
