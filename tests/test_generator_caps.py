"""Certified caps on the heaviest generator states and the coverage bound they give."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from sofic.examples.epsilon_machines import from_symbol_matrices, golden_mean_bidirectional
from sofic.generators import coupled_generators as cg
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coverage_bound import mu1_vertex
from sofic.generators.functional_generators import functional_cover_search
from sofic.generators.generator_caps import (
    certified_generator_weight_cap,
    generator_coverage_bound,
    generator_weight_cap,
    verify_weight_cap_certificate,
)
from tests.test_coupled_generators import _rll

_SURVEY_UPPER = 1.50195  # minimal functional generator of the survey process below


def _survey_machine() -> BidirectionalEpsilonMachine:
    """Survey process ((1, 1), (2, 0), (1, 0)): full-support joint, bracket [G ~ 1.097, 1.502] before these caps."""
    a, b, c = 0.6638466315183409, 0.7235161688504436, 0.5389064686238092
    mats = {
        0: np.array([[0.0, a, 0.0], [0.0, 0.0, b], [0.0, c, 0.0]]),
        1: np.array([[0.0, 1 - a, 0.0], [1 - b, 0.0, 0.0], [1 - c, 0.0, 0.0]]),
    }
    return BidirectionalEpsilonMachine.from_forward(from_symbol_matrices((0, 1, 2), (0, 1), mats))


def _joint(problem: cg._GeneratorProblem) -> np.ndarray:
    return problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)


def test_cap_meets_known_minimal_generators() -> None:
    golden = cg._GeneratorProblem(golden_mean_bidirectional(0.5))
    assert generator_weight_cap(golden, 1, max_nodes=200) == pytest.approx(2 / 3, abs=1e-6)
    rll = cg._GeneratorProblem(_rll())
    # the minimal (2, inf) generator has state weights (1/2, 1/4, 1/4)
    assert generator_weight_cap(rll, 1, max_nodes=200) >= 0.5 - 1e-9
    assert generator_weight_cap(rll, 2, max_nodes=200) >= 0.75 - 1e-9


def test_generator_coverage_meets_three_halves_on_rll() -> None:
    cov = generator_coverage_bound(_rll(), explicit=2, max_nodes=(200, 400))
    # caps (1/2, 3/4, 1) give H(1/2, 1/4, 1/4) = 3/2, the entropy of the known minimal generator
    assert cov.caps[:2] == pytest.approx((0.5, 0.75), abs=1e-6)
    assert cov.value == pytest.approx(1.5, abs=1e-6)


def test_forward_joint_stationarity_tightens_the_two_state_cap() -> None:
    problem = cg._GeneratorProblem(_survey_machine())
    cover = functional_cover_search(problem)
    weights = np.sort(np.bincount(cover.labels, weights=problem.p_pair))[::-1]
    cap2 = generator_weight_cap(problem, 2, max_nodes=600)
    # 0.918 without the forward joint-moment rows at this budget (0.907 with them on macOS)
    assert weights[:2].sum() <= cap2 + 1e-9 < 0.915


def test_rll_caps_are_exactly_one_half_and_three_quarters() -> None:
    rll = cg._GeneratorProblem(_rll())
    for k, nodes, cap in ((1, 200, Fraction(1, 2)), (2, 400, Fraction(3, 4))):
        cert = certified_generator_weight_cap(rll, k, max_nodes=nodes, data_denominator=10**6)
        # the true (2, inf) edge law has denominators <= 16, the unique such rationals this close
        assert cert.data_error < 1 / (2 * 10**12)
        assert cert.cap == cap and cert.failed == 0
        assert verify_weight_cap_certificate(rll, cert)


def test_verifier_rejects_tampered_certificates() -> None:
    rll = cg._GeneratorProblem(_rll())
    cert = certified_generator_weight_cap(rll, 1, max_nodes=60, data_denominator=10**6)
    assert verify_weight_cap_certificate(rll, cert)
    assert not verify_weight_cap_certificate(rll, replace(cert, cap=Fraction(49, 100)))
    assert not verify_weight_cap_certificate(rll, replace(cert, records=cert.records[1:]))


def test_caps_dominate_the_functional_generator_and_improve_on_mu1() -> None:
    problem = cg._GeneratorProblem(_survey_machine())
    cover = functional_cover_search(problem)
    weights = np.sort(np.bincount(cover.labels, weights=problem.p_pair))[::-1]
    cap1 = generator_weight_cap(problem, 1, max_nodes=300)
    cap2 = generator_weight_cap(problem, 2, max_nodes=300)
    assert weights[0] <= cap1 + 1e-9
    assert weights[:2].sum() <= cap2 + 1e-9
    assert cap1 < mu1_vertex(_joint(problem)).value - 0.05


def test_generator_coverage_exceeds_exact_common_information_on_survey_process() -> None:
    problem = cg._GeneratorProblem(_survey_machine())
    cov = generator_coverage_bound(problem, explicit=2, max_nodes=(300, 600))
    g_estimate, _ = cg.separating_optima(problem, starts=10, rng=np.random.default_rng(0))
    assert cov.certified
    assert g_estimate + 0.1 < cov.value <= _SURVEY_UPPER + 1e-9


def test_generative_model_reports_generator_coverage() -> None:
    from sofic.generators.generative_search import generative_model_from_bidirectional

    result = generative_model_from_bidirectional(
        _survey_machine(), stages=("machines", "functional"), generator_caps=1, rng=np.random.default_rng(0)
    )
    assert result.lower_source == "generator coverage bound (certified)"
    assert result.lower_certified
    assert 1.15 < result.lower <= result.upper + 1e-9


def test_expired_deadline_still_gives_a_valid_cap() -> None:
    import time

    rll = cg._GeneratorProblem(_rll())
    cap = generator_weight_cap(rll, 1, max_nodes=10_000, deadline=time.monotonic())
    assert 0.5 - 1e-9 <= cap <= 1.0
    cov = generator_coverage_bound(rll, explicit=2, deadline=time.monotonic())
    assert cov.value <= 1.5 + 1e-6
