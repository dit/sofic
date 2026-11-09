"""Wyner and functional variants of the generator search."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples.epsilon_machines import golden_mean_bidirectional
from sofic.generators import coupled_generators as cg
from sofic.generators.functional_generators import (
    functional_cover_search,
    minimal_functional_generator,
    product_rectangles,
)
from sofic.generators.generative_search import _wyner_information, minimal_wyner_model
from sofic.generators.lifted_common_information import lifted_partition_channel
from sofic.generators.minimal_generative_model import _normalized_joint_distribution
from tests.test_coupled_generators import _interleaved, _rll

_RLL_G = 2 - 3 / 8 * np.log2(3)


def _h(*p: float) -> float:
    q = np.array([x for x in p if x > 0])
    return float(-(q * np.log2(q)).sum())


def _excess(problem: cg._GeneratorProblem) -> float:
    return _h(*problem.p_plus) + _h(*problem.p_minus) - _h(*problem.p_pair)


def test_product_rectangles_of_dsbs_are_lines_and_cells() -> None:
    eps = 9 / 20
    pairs = [(a, b) for a in "01" for b in "01"]
    p = np.array([1 - eps, eps, eps, 1 - eps]) / 2
    blocks = product_rectangles(pairs, p)
    assert len(blocks) == 8
    assert all(len(b) <= 2 for b in blocks)


def test_product_rectangles_include_whole_product_support() -> None:
    pairs = [(a, b) for a in "01" for b in "xyz"]
    p = np.outer([0.3, 0.7], [0.2, 0.3, 0.5]).ravel()
    assert frozenset(range(6)) in product_rectangles(pairs, p)


@pytest.mark.parametrize("bidir", [golden_mean_bidirectional(0.5), _rll()], ids=["golden", "rll"])
def test_functional_generator_matches_exhaustive_partition_search(bidir) -> None:
    result = minimal_functional_generator(bidir)
    assert result.objective == "functional"
    assert result.certified
    _, exhaustive = lifted_partition_channel(bidir, _normalized_joint_distribution(bidir, cutoff=1e-10))
    assert result.upper == pytest.approx(exhaustive, abs=1e-9)


def test_functional_generator_of_rll_rejects_the_functional_common_information() -> None:
    result = minimal_functional_generator(_rll())
    search = result.stages[2]
    assert result.stages[1]["value"] == pytest.approx(_RLL_G, abs=1e-9)
    assert search["rejected covers"] > 0
    assert result.lower == pytest.approx(1.5, abs=1e-9)
    assert result.upper == pytest.approx(1.5, abs=1e-9)
    assert not result.separating_variables_generate


def test_functional_cover_search_lower_bound_when_truncated() -> None:
    problem = cg._GeneratorProblem(_rll())
    search = functional_cover_search(problem, max_covers=1)
    assert not search.exact
    assert search.lower == pytest.approx(_RLL_G, abs=1e-9)
    assert search.lower <= 1.5 <= search.value + 1e-12


def test_functional_generator_certified_on_sixteen_pair_interleaved() -> None:
    result = minimal_functional_generator(_interleaved(0.1))
    assert result.certified
    assert result.upper >= result.lower - 1e-12


def test_generative_model_uses_certified_coverage_on_full_support() -> None:
    from sofic.generators.generative_search import generative_model_from_bidirectional

    result = generative_model_from_bidirectional(_interleaved(9 / 20), stages=("machines",), generator_caps=0)
    assert result.lower_source == "coverage bound (certified)"
    assert result.lower_certified
    assert result.lower == pytest.approx(0.78730, abs=1e-4)


def _near_degenerate_survey_machine():
    """Survey process ((0, 1), (2, 0), (0, None)): dit's exact-CI channel misses exact separation."""
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

    a, b = 0.9928125105284444, 0.8050849845480182
    mats = {
        0: np.array([[a, 0.0, 0.0], [0.0, 0.0, b], [1.0, 0.0, 0.0]]),
        1: np.array([[0.0, 1 - a, 0.0], [1 - b, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
    return BidirectionalEpsilonMachine.from_forward(from_symbol_matrices((0, 1, 2), (0, 1), mats))


@pytest.mark.parametrize("seed", [4, 7])
def test_exact_common_information_estimate_is_never_below_certified_lower(seed: int) -> None:
    from sofic.generators.generative_search import generative_model_from_bidirectional

    result = generative_model_from_bidirectional(
        _near_degenerate_survey_machine(),
        stages=("machines", "exact"),
        exact_starts=6,
        entropy_sweep_starts=6,
        bilevel=False,
        # generator caps bound C_g and may exceed G; this test is about bounds on G
        generator_caps=0,
        targeted=False,
        rng=np.random.default_rng(seed),
    )
    assert result.lower_certified
    assert result.exact_common_information_estimate is not None
    assert result.exact_common_information_estimate >= result.lower - 1e-9


def test_generative_model_offers_minimal_functional_generator() -> None:
    from sofic.generators.generative_search import generative_model_from_bidirectional

    result = generative_model_from_bidirectional(
        _interleaved(0.1), stages=("functional",), generator_caps=0, rng=np.random.default_rng(0)
    )
    stage = next(s for s in result.stages if s["stage"] == "functional cover search")
    assert stage["exact"]
    assert result.upper_source == "minimal functional generator"
    assert result.upper == pytest.approx(stage["value"], abs=1e-9)


_FAST_WYNER = {"max_states": 3, "starts": 4, "exact_starts": 8, "entropy_sweep_starts": 10, "memory_starts": 4}


def test_wyner_model_on_golden_mean_meets_wyner_common_information() -> None:
    bidir = golden_mean_bidirectional(0.5)
    result = minimal_wyner_model(bidir, rng=np.random.default_rng(0), **_FAST_WYNER)
    problem = cg._GeneratorProblem(bidir)
    assert result.objective == "wyner"
    assert result.lower == pytest.approx(_excess(problem), abs=1e-9)
    # both persistent crypticities vanish, so the minimum over generators is C[S+ : S-] = 2/3
    assert result.upper == pytest.approx(2 / 3, abs=1e-3)
    assert result.upper <= result.model.state_entropy() + 1e-9


def test_wyner_model_of_rll_is_at_most_its_generative_complexity() -> None:
    bidir = _rll()
    result = minimal_wyner_model(bidir, rng=np.random.default_rng(0), **_FAST_WYNER)
    assert result.lower == pytest.approx(_excess(cg._GeneratorProblem(bidir)), abs=1e-9)
    assert result.lower <= result.upper <= 1.5 + 1e-6


def test_wyner_information_of_epsilon_machine_is_statistical_complexity_minus_crypticity() -> None:
    problem = cg._GeneratorProblem(_rll())
    seed = problem.epsilon_seed()
    # W = S+ is a function of the pair: I[S+ : S+, S-] = H[S+]
    assert _wyner_information(problem, seed) == pytest.approx(_h(*problem.p_plus), abs=1e-9)
