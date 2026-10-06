"""Erasure-entropy identities checked against the exact ephemeral information ``r_mu``.

``r_mu = H[X_0 | X_{:0}, X_{1:}]`` is the erasure entropy rate ``h^-`` of Verdu &
Weissman (2008). The identities come from van Enter & Verbitskiy, *Erasure entropies
and Gibbs measures* (arXiv:1001.3122):

- Theorem 2: ``h^- = lim_n H[X_0 | X_{-n:0}, X_{1:n+1}] <= h``.
- Eq. (5): for a ``k``-step Markov chain, ``(k + 1) h = h^- + H[X_{1:k+1} | X_{-k:0}]``.
- Eq. (8): for a Gibbs measure, ``h^- = -E log gamma(X_0 | X_{-1}, X_1)`` with
  ``gamma`` the single-site specification.
- Section 2: ``h^-`` is not an isomorphism invariant. The 2-block presentation
  ``Y_n = (X_{n-1}, X_n)`` of the fair coin has ``h = 1`` but ``h^- = 0``.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sofic.examples import nemo_process
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.shifts.sofic import SoficShift

pytest.importorskip("dit")


def _entropy(dist):
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def _marginal(dist, keep):
    out = defaultdict(float)
    for word, p in dist.items():
        out[tuple(word[i] for i in keep)] += p
    return out


def _window_erasure(machine, length, context):
    """``H[X_{c:c+L} | X_{0:c}, X_{c+L:2c+L}]`` from exact word probabilities."""
    words = machine.words_of_length(2 * context + length)
    outside = list(range(context)) + list(range(context + length, 2 * context + length))
    return _entropy(words) - _entropy(_marginal(words, outside))


def _two_sided(machine, n):
    """``H[X_0 | X_{-n:0}, X_{1:n+1}]``."""
    return _window_erasure(machine, 1, n)


def _markov_machine(transition):
    """Order-1 Markov chain on symbols ``0..m-1``: the state is the last symbol."""
    m = transition.shape[0]
    matrices = {}
    for x in range(m):
        matrix = np.zeros((m, m))
        matrix[:, x] = transition[:, x]
        matrices[x] = matrix
    return from_symbol_matrices(tuple(range(m)), tuple(range(m)), matrices)


def _order_two_machine(conditional):
    """Binary order-2 Markov chain; ``conditional[(a, b)]`` is ``P(X_t = 1 | a, b)``."""
    pairs = tuple(itertools.product((0, 1), repeat=2))
    states = tuple(f"{a}{b}" for a, b in pairs)
    matrices = {x: np.zeros((4, 4)) for x in (0, 1)}
    for a, b in pairs:
        p1 = conditional[(a, b)]
        matrices[0][states.index(f"{a}{b}"), states.index(f"{b}0")] = 1.0 - p1
        matrices[1][states.index(f"{a}{b}"), states.index(f"{b}1")] = p1
    return from_symbol_matrices(states, (0, 1), matrices)


def _two_block_coin():
    """``Y_n = (X_{n-1}, X_n)`` for a fair coin ``X``: an isomorphic copy of the coin."""
    matrices = {}
    for s, x in itertools.product((0, 1), repeat=2):
        matrix = np.zeros((2, 2))
        matrix[s, x] = 0.5
        matrices[f"{s}{x}"] = matrix
    return from_symbol_matrices((0, 1), tuple(matrices), matrices)


def _gap_conditional(machine, k):
    """``H[X_{1:k+1} | X_{-k:0}]``: the next ``k`` symbols given the past ``k``, across the gap at 0."""
    words = machine.words_of_length(2 * k + 1)
    outside = _marginal(words, list(range(k)) + list(range(k + 1, 2 * k + 1)))
    return _entropy(outside) - _entropy(_marginal(words, range(k)))


positive_probs = st.floats(min_value=0.05, max_value=0.95)


@st.composite
def binary_transitions(draw):
    p, q = draw(positive_probs), draw(positive_probs)
    return np.array([[1.0 - p, p], [q, 1.0 - q]])


@given(binary_transitions())
def test_eq5_order_one_markov(transition):
    machine = _markov_machine(transition)
    h, r = machine.entropy_rate(), machine.ephemeral_information()
    assert 2 * h == pytest.approx(r + _gap_conditional(machine, 1), abs=1e-9)


@given(st.tuples(*[positive_probs] * 4))
def test_eq5_order_two_markov(probs):
    conditional = dict(zip(itertools.product((0, 1), repeat=2), probs, strict=True))
    machine = _order_two_machine(conditional)
    h, r = machine.entropy_rate(), machine.ephemeral_information()
    assert 3 * h == pytest.approx(r + _gap_conditional(machine, 2), abs=1e-9)


@given(binary_transitions())
def test_eq8_gibbs_specification(transition):
    """``h^- = -E log2 gamma(X_0 | X_{-1}, X_1)``, ``gamma(b | a, c) ~ P(a, b) P(b, c)``."""
    machine = _markov_machine(transition)
    pi = machine.stationary_distribution()
    expected = 0.0
    for a, b, c in itertools.product(range(2), repeat=3):
        weight = pi[a] * transition[a, b] * transition[b, c]
        gamma = transition[a, b] * transition[b, c] / sum(transition[a, d] * transition[d, c] for d in range(2))
        expected -= weight * math.log2(gamma)
    assert machine.ephemeral_information() == pytest.approx(expected, abs=1e-9)


@given(binary_transitions())
def test_theorem2_markov_two_sided_window_is_exact(transition):
    """For an order-1 chain the nearest neighbours already screen off the rest."""
    machine = _markov_machine(transition)
    r = machine.ephemeral_information()
    assert r <= machine.entropy_rate() + 1e-12
    for n in (1, 2, 3):
        assert _two_sided(machine, n) == pytest.approx(r, abs=1e-9)


def test_theorem2_hidden_markov_two_sided_limit():
    """Non-Markov process: two-sided conditionals decrease toward ``h^-`` from above."""
    machine = nemo_process()
    r = machine.ephemeral_information()
    approximants = [_two_sided(machine, n) for n in range(1, 7)]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(approximants))
    assert all(a >= r - 1e-9 for a in approximants)
    assert approximants[-1] - r < approximants[0] - r
    assert r <= machine.entropy_rate() + 1e-12


def test_two_block_coin_has_zero_erasure_entropy():
    machine = _two_block_coin()
    assert machine.entropy_rate() == pytest.approx(1.0, abs=1e-9)
    assert machine.ephemeral_information() == pytest.approx(0.0, abs=1e-9)
    assert 2 * machine.entropy_rate() == pytest.approx(
        machine.ephemeral_information() + _gap_conditional(machine, 1), abs=1e-9
    )


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_two_block_coin_is_not_bilaterally_deterministic(length):
    """Erasing ``L`` consecutive pairs loses ``L - 1`` coin bits, so only single erasures are free."""
    assert _window_erasure(_two_block_coin(), length, 2) == pytest.approx(length - 1, abs=1e-9)


def test_erasure_rate_is_not_a_conjugacy_invariant():
    """The full 2-shift and its 2-block presentation are conjugate but have different ``r_top``."""
    two_block = SoficShift(symbol_alphabet=frozenset({"00", "01", "10", "11"}))
    for state in (0, 1):
        two_block.graph.add_state(state)
    for s, x in itertools.product((0, 1), repeat=2):
        two_block.add_transition(s, x, f"{s}{x}")
    full = SoficShift(symbol_alphabet=frozenset({0, 1}))
    full.graph.add_state("S")
    full.add_transition("S", "S", 0)
    full.add_transition("S", "S", 1)

    full_anatomy, block_anatomy = full.topological_anatomy(), two_block.topological_anatomy()
    assert block_anatomy["h_top"] == pytest.approx(full_anatomy["h_top"], abs=1e-9)
    assert full_anatomy["r_top"] == pytest.approx(1.0, abs=1e-9)
    assert block_anatomy["r_top"] == pytest.approx(0.0, abs=1e-9)
