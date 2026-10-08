"""Statistical complexity dimension (Jurgens & Crutchfield, Chaos 2021)."""

import math

import numpy as np
import pytest

from sofic.examples import even_process, golden_mean, nonunifilar_golden_mean, sns
from sofic.generators.complexity_dimension import ifs_lyapunov_dimension, statistical_complexity_dimension
from sofic.generators.mealy import MealyHMM


def _from_matrices(matrices: dict[str, np.ndarray]) -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset(matrices))
    for symbol, matrix in matrices.items():
        for i in range(matrix.shape[0]):
            hmm.graph.add_state(i)
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    hmm.add_transition(i, j, symbol, float(matrix[i, j]))
    return hmm


def cantor_machine(s: float) -> MealyHMM:
    """Fig. 1/2: the mixed states form the middle-1/s Cantor set; both maps contract by 1/s."""
    return _from_matrices(
        {
            "0": np.array([[0.5, 0.0], [(s - 1) / (2 * s), 1 / (2 * s)]]),
            "1": np.array([[1 / (2 * s), (s - 1) / (2 * s)], [0.0, 0.5]]),
        }
    )


def sierpinski_machine(s: float = 2.0, a: float = 1 / 6) -> MealyHMM:
    """Appendix D, Eq. (S1): the mixed-state attractor is the Sierpinski triangle."""
    b = 1 - a * s
    return _from_matrices(
        {
            "0": np.array([[a, 0, a * (s - 1)], [0, a, a * (s - 1)], [0, 0, a * s]]),
            "1": np.array(
                [[b / 2, 0, 0], [b * (s - 1) / (2 * s), b / (2 * s), 0], [b * (s - 1) / (2 * s), 0, b / (2 * s)]]
            ),
            "2": np.array(
                [[b / (2 * s), b * (s - 1) / (2 * s), 0], [0, b / 2, 0], [0, b * (s - 1) / (2 * s), b / (2 * s)]]
            ),
        }
    )


@pytest.mark.parametrize("s", [3.0, 4.0])
def test_cantor_machine_dimension(s):
    result = statistical_complexity_dimension(cantor_machine(s), n_samples=5_000, seed=0)
    assert not result.finite
    assert result.entropy_rate == pytest.approx(1.0)
    np.testing.assert_allclose(result.lyapunov_exponents, [-math.log2(s)], rtol=1e-9)
    assert result.dimension == pytest.approx(math.log(2) / math.log(s), rel=1e-9)


def test_sierpinski_machine_dimension():
    result = statistical_complexity_dimension(sierpinski_machine(), n_samples=5_000, seed=0)
    assert result.entropy_rate == pytest.approx(math.log2(3))
    np.testing.assert_allclose(result.lyapunov_exponents, [-1.0, -1.0], rtol=1e-9)
    assert result.dimension == pytest.approx(math.log2(3), rel=1e-9)


@pytest.mark.parametrize("factory", [golden_mean, even_process])
def test_unifilar_processes_have_zero_dimension(factory):
    result = statistical_complexity_dimension(factory(), n_samples=2_000, seed=0)
    assert result.finite
    assert result.dimension == 0.0
    assert result.entropy_rate == pytest.approx(factory().entropy_rate())


def test_nonunifilar_presentation_with_finite_epsilon_machine_has_zero_dimension():
    result = statistical_complexity_dimension(nonunifilar_golden_mean(), n_samples=2_000, seed=0)
    assert result.finite
    assert result.dimension == 0.0
    assert result.entropy_rate == pytest.approx(2 / 3)


def test_countable_mixed_states_have_zero_dimension():
    result = statistical_complexity_dimension(sns(), n_samples=5_000, seed=0, max_states=200)
    assert not result.finite
    assert result.lyapunov_exponents[0] == -math.inf
    assert result.dimension == 0.0


def test_seed_determinism():
    hmm = cantor_machine(3.0)
    first = statistical_complexity_dimension(hmm, n_samples=1_000, seed=5, max_states=None)
    second = statistical_complexity_dimension(hmm, n_samples=1_000, seed=np.random.default_rng(5), max_states=None)
    assert first.dimension == second.dimension
    np.testing.assert_array_equal(first.lyapunov_exponents, second.lyapunov_exponents)


@pytest.mark.parametrize(
    ("rate", "exponents", "expected"),
    [
        (1.0, [-2.0], 0.5),
        (math.log2(3), [-1.0, -1.0], math.log2(3)),
        (3.0, [-1.0, -1.0], 2.0),
        (0.0, [-1.0], 0.0),
        (1.0, [-math.inf], 0.0),
        (1.5, [-1.0, -math.inf], 1.0),
        (1.0, [], 0.0),
    ],
)
def test_ifs_lyapunov_dimension(rate, exponents, expected):
    assert ifs_lyapunov_dimension(rate, np.array(exponents)) == pytest.approx(expected)
