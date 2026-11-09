"""Line-split coupled seeds: broad decomposition terms plus point-mass states."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators import coupled_generators as cg
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


def _survey_machine() -> BidirectionalEpsilonMachine:
    """Survey process ((0, 1), (2, 1), (0, 1)): bracket [1.234, 1.397] before line-split seeds."""
    a, b, c = 0.23098961132583967, 0.4322639772906034, 0.7484936101088153
    mats = {
        0: np.array([[a, 0.0, 0.0], [0.0, 0.0, b], [c, 0.0, 0.0]]),
        1: np.array([[0.0, 1 - a, 0.0], [0.0, 1 - b, 0.0], [0.0, 1 - c, 0.0]]),
    }
    return BidirectionalEpsilonMachine.from_forward(from_symbol_matrices((0, 1, 2), (0, 1), mats))


def _widest_survey_machine() -> BidirectionalEpsilonMachine:
    """Survey process ((0, 1), (0, 2), (0, 1)): bracket [0.906, 1.371] before the targeted search."""
    a, b, c = 0.06761675137492751, 0.5575272429346271, 0.45789456421998476
    mats = {
        0: np.array([[a, 0.0, 0.0], [0.44247275706537287, 0.0, 0.0], [c, 0.0, 0.0]]),
        1: np.array([[0.0, 1 - a, 0.0], [0.0, 0.0, b], [0.0, 1 - c, 0.0]]),
    }
    return BidirectionalEpsilonMachine.from_forward(from_symbol_matrices((0, 1, 2), (0, 1), mats))


def test_targeted_search_finds_low_entropy_generator() -> None:
    problem = cg._GeneratorProblem(_widest_survey_machine())
    _best, optima = cg.separating_optima(problem, starts=10, rng=np.random.default_rng(0))
    seeds = cg.targeted_line_split_seeds(problem, optima, configs=((1, 1), (2, 0)), rng=np.random.default_rng(0))
    assert 0 < len(seeds) <= 3
    assert all(problem.residual(*seed) <= 1e-7 for _name, seed in seeds)
    # the pipeline's other stages reach 1.371 here and a 0.970-bit generator is known; COBYLA alone
    # stops at 0.98 to 1.10 depending on the seed, and the local-search hops reach 0.970 to 0.975
    assert min(cg._entropy(seed[0]) for _name, seed in seeds) < 0.98


def test_line_split_seeds_are_exact_decompositions_with_point_mass_states() -> None:
    problem = cg._GeneratorProblem(_survey_machine())
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    _best, optima = cg.separating_optima(problem, starts=10, rng=np.random.default_rng(0))
    seeds = cg.line_split_seeds(problem, optima)
    assert seeds
    for _name, (pi, Q, R, K) in seeds:
        assert np.einsum("w,wa,wb->ab", pi, Q, R) == pytest.approx(P, abs=1e-9)
        point = np.isclose(Q.max(axis=1), 1.0) | np.isclose(R.max(axis=1), 1.0)
        assert point.sum() >= len(pi) - 2
        assert K.reshape(len(pi), -1).sum(axis=1) == pytest.approx(1.0)


def test_pipeline_offers_line_split_candidates() -> None:
    from sofic.generators.generative_search import generative_model_from_bidirectional

    result = generative_model_from_bidirectional(
        _survey_machine(),
        stages=("machines", "exact"),
        exact_starts=10,
        entropy_sweep_starts=10,
        bilevel=False,
        generator_caps=0,
        targeted=False,
        rng=np.random.default_rng(0),
    )
    # Which exact-CI optima the multistart finds varies with platform floating point, so the
    # line-split candidates need not win (1.2367 bits locally, epsilon machine 1.4557 on some CI).
    line_split = [s for s in result.stages if s["stage"].startswith("line split")]
    assert line_split
    epsilon = next(s for s in result.stages if s["stage"] == "epsilon machine")
    assert result.upper <= epsilon["value"] + 1e-9
