"""Tests for GeneratorProblemSpec snapshots and future-window specs."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.epsilon_machines import golden_mean_bidirectional
from sofic.generators.coupled_generators import _GeneratorProblem, _kernel_lp_beliefs
from sofic.generators.generator_caps import generator_coverage_bound, generator_weight_cap
from sofic.generators.generator_problem_spec import spec_from_generator_problem, window_spec
from tests.test_coupled_generators import _rll
from tests.test_generator_caps import _SURVEY_UPPER, _survey_machine


def test_spec_from_generator_problem_roundtrip_shapes() -> None:
    problem = _GeneratorProblem(golden_mean_bidirectional(0.5))
    spec = spec_from_generator_problem(problem)
    assert spec.Pe.shape == problem.Pe.shape
    assert spec.n_sym == problem.n_sym


def test_problem_from_spec_reproduces_caps_and_kernel_lp() -> None:
    problem = _GeneratorProblem(_survey_machine())
    rebuilt = _GeneratorProblem.from_spec(spec_from_generator_problem(problem))
    assert generator_weight_cap(rebuilt, 1, max_nodes=40) == generator_weight_cap(problem, 1, max_nodes=40)
    seed = problem.epsilon_seed()
    assert _kernel_lp_beliefs(rebuilt, *seed[:3])[0] == pytest.approx(_kernel_lp_beliefs(problem, *seed[:3])[0])


@pytest.mark.parametrize("length", [1, 2, 3])
def test_window_spec_is_a_consistent_edge_law(length: int) -> None:
    spec = window_spec(_survey_machine().forward_machine, length)
    assert spec.Pe.sum() == pytest.approx(1.0)
    # S+_0 marginal is stationary; Y_1 = X_{1:L+1} has the same law as Y_0 = X_{0:L}
    assert spec.Pe.sum(axis=(1, 2)) == pytest.approx(spec.p_plus)
    assert spec.Pe.sum(axis=(0, 1)) == pytest.approx(spec.p_minus)
    # Y_0 = g(X_0, Y_1): each (x, y') with mass maps to exactly one window
    live = spec.Pe.sum(axis=0) > 0
    assert np.all(spec.G.sum(axis=2)[live] == 1)
    # every generator satisfies the window equations, in particular the forward epsilon-machine
    problem = _GeneratorProblem.from_spec(spec)
    pi, Q, R, _K = problem.epsilon_seed()
    assert _kernel_lp_beliefs(problem, pi, Q, R)[0] < 1e-9


def test_two_symbol_window_gives_three_halves_for_rll() -> None:
    problem = _GeneratorProblem.from_spec(window_spec(_rll().forward_machine, 2))
    assert problem.n_minus == 3
    assert generator_coverage_bound(problem, explicit=2, max_nodes=(300, 600)).value == pytest.approx(1.5, abs=1e-6)


def test_window_bounds_stay_below_the_known_upper_bound() -> None:
    problem = _GeneratorProblem.from_spec(window_spec(_survey_machine().forward_machine, 2))
    value = generator_coverage_bound(problem, explicit=2, max_nodes=(200, 400)).value
    assert 0.9 < value <= _SURVEY_UPPER
