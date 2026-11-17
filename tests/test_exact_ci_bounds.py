"""Certified heaviest-term lower bound on the exact common information."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.generators.coupled_generators import _decompose
from sofic.generators.exact_ci_bounds import capped_coverage_entropy, heaviest_term_bound


def _h(p: float) -> float:
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def test_capped_coverage_is_below_a_feasible_vector_the_greedy_misses() -> None:
    # caps (0.5, 0.6, 1.0): the sorted vector (0.34, 0.26, 0.26, 0.14) is feasible with S_3 = 0.86 > 0.7
    y = np.array([0.34, 0.26, 0.26, 0.14])
    assert capped_coverage_entropy([0.5, 0.6, 1.0], 1.0) <= float(-(y * np.log2(y)).sum())


def test_dsbs_reaches_its_exact_common_information() -> None:
    eps = 0.45
    D = np.array([[1 - eps, eps], [eps, 1 - eps]]) / 2
    G = _h(1 / 11)  # Kumar-Li-El Gamal Prop. 8 at eps = 9/20: weights (10/11, 1/11)
    result = heaviest_term_bound(D, target=G - 1e-4, max_nodes=50_000)
    assert result.reached
    assert result.value <= G + 1e-9


@settings(max_examples=8, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_bound_is_below_every_exact_decomposition(seed: int) -> None:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(9)).reshape(3, 3)
    result = heaviest_term_bound(P, max_nodes=2_000)
    for k in (3, 4, 5):
        lam, A, B, residual = _decompose(
            P, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(3), size=k), rng.dirichlet(np.ones(3), size=k)
        )
        if residual <= 1e-10:
            p = lam[lam > 1e-12]
            assert result.value <= float(-(p * np.log2(p)).sum()) + 1e-9


def test_survey_case_closes_against_its_exact_common_information() -> None:
    """((0,1),(2,·),(0,1)): the bound rises to within 0.01 of the best exact decomposition (0.93349 bits)."""
    P = np.array([[0.107, 0.3019, 0.1449], [0.0781, 0.1449, 0.0], [0.0379, 0.107, 0.0781]])
    P = P / P.sum()
    best = np.inf
    rng = np.random.default_rng(0)
    for k in (2, 3, 4):
        for _ in range(20):
            lam, A, B, residual = _decompose(
                P, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(3), size=k), rng.dirichlet(np.ones(3), size=k)
            )
            if residual <= 1e-10:
                p = lam[lam > 1e-12]
                best = min(best, float(-(p * np.log2(p)).sum()))
    result = heaviest_term_bound(P, target=best - 1e-2, max_nodes=100_000)
    assert result.reached
    assert best - 1e-2 <= result.value <= best + 1e-9


def test_weight_cap_that_admits_no_term_prunes_every_box() -> None:
    D = np.array([[0.3, 0.2], [0.2, 0.3]])
    result = heaviest_term_bound(D, max_nodes=10, weight_cap=lambda box: None)
    assert result.value == np.inf
    assert result.open_boxes == 0


def _survey_case_a():
    """Survey process ``((0,1),(0,2),(0,1))`` (v11 seed 54), where C_g >= 0.928 > G = 0.92433."""
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.coupled_generators import _GeneratorProblem
    from sofic.generators.epsilon_machine import EpsilonMachine

    a, b, c = 0.06761675137492751, 0.44247275706537287, 0.45789456421998476
    mats = {
        0: np.array([[a, 0, 0], [b, 0, 0], [c, 0, 0]]),
        1: np.array([[0, 1 - a, 0], [0, 0, 1 - b], [0, 1 - c, 0]]),
    }
    machine = from_symbol_matrices((0, 1, 2), (0, 1), mats)
    forward = machine if isinstance(machine, EpsilonMachine) else EpsilonMachine.from_hmm(machine)
    return _GeneratorProblem(BidirectionalEpsilonMachine.from_forward(forward))


def test_generator_equations_exclude_the_g_optimal_heavy_term() -> None:
    from sofic.generators.generator_caps import _HeaviestStatesRelaxation

    problem = _survey_case_a()
    # heaviest term of the G-optimal decomposition (entropy 0.92433)
    t = 0.6943850024064206
    q = np.array([0.06761675137400137, 0.5912539107207103, 0.3411293379052883])
    r = np.array([0.18172612379896527, 0.46738361190932987, 0.3508902642917049])
    relaxation = _HeaviestStatesRelaxation(problem, 1)
    wide = relaxation.bound(
        ((np.clip(q - 1e-2, 0, 1), np.clip(q + 1e-2, 0, 1), np.clip(r - 1e-2, 0, 1), np.clip(r + 1e-2, 0, 1)),)
    )
    tight = relaxation.bound(
        ((np.clip(q - 1e-4, 0, 1), np.clip(q + 1e-4, 0, 1), np.clip(r - 1e-4, 0, 1), np.clip(r + 1e-4, 0, 1)),)
    )
    assert wide is not None and wide > t
    assert tight is not None and tight < t - 1e-3


def test_generator_heaviest_term_bound_is_below_a_known_generator() -> None:
    from sofic.generators.exact_ci_bounds import generator_heaviest_term_bound

    result = generator_heaviest_term_bound(_survey_case_a(), max_nodes=300)
    assert 0.85 < result.value <= 0.9697134 + 1e-9  # entropy of a word-verified generator


def test_generator_heaviest_term_bound_stops_at_its_deadline() -> None:
    import time

    from sofic.generators.exact_ci_bounds import generator_heaviest_term_bound

    start = time.monotonic()
    generator_heaviest_term_bound(_survey_case_a(), deadline=start + 2.0)
    assert time.monotonic() - start < 10.0


def _dataset_case_t006():
    """Dataset process ``n2-t006-d0`` (code ((0,1),(0,0))): C_g = 0.8551484, coverage bound 0.8379."""
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.coupled_generators import _GeneratorProblem
    from sofic.generators.epsilon_machine import EpsilonMachine

    a, b = 0.6115387860108349, 0.2662431875066565
    mats = {0: np.array([[a, 0], [b, 0]]), 1: np.array([[0, 1 - a], [1 - b, 0]])}
    machine = from_symbol_matrices((0, 1), (0, 1), mats)
    forward = machine if isinstance(machine, EpsilonMachine) else EpsilonMachine.from_hmm(machine)
    return _GeneratorProblem(BidirectionalEpsilonMachine.from_forward(forward))


@pytest.mark.slow
def test_generator_heaviest_term_bound_closes_a_two_state_bracket() -> None:
    from sofic.generators.exact_ci_bounds import generator_heaviest_term_bound

    upper = 0.8551483886521953  # entropy of a word-verified generator
    result = generator_heaviest_term_bound(_dataset_case_t006(), target=upper - 5e-5)
    assert result.reached
    assert result.value <= upper + 1e-9


def _dataset_case_t055():
    """Dataset process ``n3-t055-d2`` (code ((0,1),(0,2),(0,2))): C_g = G = 1.42254, minimal lifted model 1.4815+."""
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.coupled_generators import _GeneratorProblem
    from sofic.generators.epsilon_machine import EpsilonMachine

    a, b, c = 0.29501936538151036, 0.7629418805405772, 0.1416369489322688
    mats = {
        0: np.array([[a, 0, 0], [b, 0, 0], [c, 0, 0]]),
        1: np.array([[0, 1 - a, 0], [0, 0, 1 - b], [0, 0, 1 - c]]),
    }
    machine = from_symbol_matrices((0, 1, 2), (0, 1), mats)
    forward = machine if isinstance(machine, EpsilonMachine) else EpsilonMachine.from_hmm(machine)
    return _GeneratorProblem(BidirectionalEpsilonMachine.from_forward(forward))


def test_epsilon_machine_states_pass_the_lifted_term_test() -> None:
    from sofic.generators.exact_ci_bounds import _LiftedTermTest

    problem = _dataset_case_t055()
    test = _LiftedTermTest(problem.lifted)
    _, Q, R, _ = problem.epsilon_seed()
    assert all(test.violation(q, r) < 1e-12 for q, r in zip(Q, R, strict=True))


def test_lifted_bound_separates_lifted_generators_from_g() -> None:
    from sofic.generators.exact_ci_bounds import lifted_heaviest_term_bound

    problem = _dataset_case_t055()
    P = problem.lifted.plus.T @ (problem.p_pair[:, None] * problem.lifted.minus)
    lifted = lifted_heaviest_term_bound(problem, target=1.43, max_nodes=50_000)
    static = heaviest_term_bound(P, target=1.43, max_nodes=50_000)
    assert lifted.reached  # every lifted generator has entropy >= 1.43
    assert static.value <= 1.42255 + 1e-6  # a coupled generator of entropy 1.4225448 exists, so G <= 1.42255
    c_mu = float(-(problem.p_plus * np.log2(problem.p_plus)).sum())
    assert lifted.value <= c_mu  # the epsilon-machine is a lifted generator
