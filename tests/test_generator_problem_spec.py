"""Tests for GeneratorProblemSpec snapshots."""

from __future__ import annotations

from sofic.examples.epsilon_machines import golden_mean_bidirectional
from sofic.generators.coupled_generators import _GeneratorProblem
from sofic.generators.generator_problem_spec import spec_from_generator_problem


def test_spec_from_generator_problem_roundtrip_shapes() -> None:
    problem = _GeneratorProblem(golden_mean_bidirectional(0.5))
    spec = spec_from_generator_problem(problem)
    assert spec.Pe.shape == problem.Pe.shape
    assert spec.n_sym == problem.n_sym
