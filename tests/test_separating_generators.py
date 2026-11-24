"""Exact minimal generators for cryptic order at most 1, the support bound, and lookahead generators."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from sofic.generators import separating_generators as sg


def _h(p):
    p = np.asarray(p, float).ravel()
    p = p[p > 1e-300]
    return float(-(p * np.log2(p)).sum())


def _problem(machine):
    from sofic.generators.coupled_generators import _GeneratorProblem
    from sofic.generators.generative_search import resolve_bidirectional

    return _GeneratorProblem(resolve_bidirectional(machine)[0])


def _joint(problem):
    lif = problem.lifted
    return np.einsum("p,pa,pb->ab", lif.p, lif.plus, lif.minus)


def _topologies():
    from sofic.examples.epsilon_machines import even_process, golden_mean, restricted_golden_mean
    from sofic.examples.processes import rip, stretched_gm

    return {
        "golden mean": golden_mean(),
        "even": even_process(),
        "rip": rip(),
        "stretched gm 2": stretched_gm(2),
        "rgm 2": restricted_golden_mean(2),
    }


def _random_machine(name: str, seed: int):
    from sofic.examples.processes import uniform_mealyhmm
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.graph import ATTR_PROB

    m = uniform_mealyhmm(_topologies()[name], create_using=EpsilonMachine, prng=np.random.default_rng(seed))
    m = m if isinstance(m, EpsilonMachine) else EpsilonMachine.from_hmm(m)
    # Below this the reverse ε-machine can come out non-minimal: on the (2, inf) topology a self-loop of 0.0010739
    # gives 4 reverse states, two of them predictively equivalent up to 3e-10, and a reverse cryptic order of inf.
    assume(min(float(t.data[ATTR_PROB]) for t in m.transitions()) >= 0.02)
    return m


# Kumar, Li and El Gamal's Example 2 (an erasure joint), and a 3 x 3 joint where OBBT cuts off the optimum.
KUMAR_2 = np.array([[0.05, 0.0, 0.45], [0.0, 0.05, 0.45]])
JOINT_52 = np.array(
    [
        [0.25510874950017903, 0.0, 0.18459086280725023],
        [0.0, 0.0, 0.41334737027755303],
        [0.0, 0.14247245903801908, 0.004480558376998535],
    ]
)


@pytest.mark.parametrize(
    ("P", "K"),
    [(np.eye(3) / 3, 3), (np.full((2, 2), 0.25), 3), (KUMAR_2, 3), (JOINT_52, 4), (np.ones((1, 4)) / 4, 2)],
)
def test_support_bound_states(P, K) -> None:
    assert sg.support_bound_states(P) == K


@st.composite
def _decompositions(draw):
    rows, cols, terms = draw(st.integers(2, 4)), draw(st.integers(2, 4)), draw(st.integers(1, 8))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    A = rng.random((terms, rows)) * (rng.random((terms, rows)) < 0.7)
    A[A.sum(axis=1) == 0, rng.integers(rows)] = 1.0
    R = rng.dirichlet(np.ones(cols), size=terms) * (rng.random((terms, cols)) < 0.6)
    R[R.sum(axis=1) == 0, rng.integers(cols)] = 1.0
    return A / A.sum(), R / R.sum(axis=1, keepdims=True)


@settings(max_examples=60, deadline=None)
@given(_decompositions())
def test_reduction_keeps_the_joint_lowers_the_entropy_and_reaches_the_support_bound(dec) -> None:
    A, R = dec
    P = A.T @ R
    A2, R2, moves = sg.reduce_decomposition(P, A, R)
    assert np.abs(A2.T @ R2 - P).max() <= 1e-9
    assert (A2 >= 0).all() and np.allclose(R2.sum(axis=1), 1.0)
    assert _h(A2.sum(axis=1)) <= _h(A.sum(axis=1)) + 1e-9
    assert len(A2) <= sg.support_bound_states(P)
    assert sum(moves.values()) >= len(A) - len(A2)


_KAPPA_LE_1 = ("golden mean", "even", "rip", "stretched gm 2")


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(sorted(_topologies())), st.integers(0, 2**32 - 1))
def test_theorem_class_identity_and_equal_cryptic_orders(name, seed) -> None:
    """p.chi+ = chi+ - H[S+_0 | X_0, S+_1], zero exactly at cryptic order at most 1; kappa+ = kappa-."""
    from sofic.generators.generative_search import _cryptic_orders, _persistent_criteria, resolve_bidirectional

    machine = _random_machine(name, seed)
    bidir = resolve_bidirectional(machine)[0]
    problem = _problem(machine)
    lif = problem.lifted
    P = _joint(problem)
    chi_plus = _h(P) - _h(P.sum(axis=0))
    F = np.einsum("pa,pxq,qc->axc", lif.plus, lif.edges, lif.plus)
    p_chi_plus, _ = _persistent_criteria(problem)
    assert p_chi_plus == pytest.approx(chi_plus - (_h(F) - _h(F.sum(axis=0))), abs=1e-10)
    kappa_plus, kappa_minus = _cryptic_orders(bidir)
    assert kappa_plus == kappa_minus
    assert (p_chi_plus < 1e-10) == (kappa_plus <= 1)
    ladder = sg._block_persistent_crypticities(problem)
    assert all((max(plus, minus) < 1e-10) == (length >= kappa_plus) for length, plus, minus in ladder)


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(_KAPPA_LE_1), st.integers(0, 2**32 - 1))
def test_theorem_generator_generates(name, seed) -> None:
    from sofic.generators.coupled_generators import CoupledSearchResult, _model_from_kernel
    from sofic.generators.minimal_generative_model import _equivalence_error

    machine = _random_machine(name, seed)
    assume(machine.cryptic_order() <= 1)
    problem = _problem(machine)
    P = _joint(problem)
    dec = sg.separating_decomposition(P, time_limit=20)
    A, R, _ = sg.reduce_decomposition(P, dec.pi[:, None] * dec.Q, dec.R)
    assert len(A) <= sg.support_bound_states(P, tol=1e-12)
    pi = A.sum(axis=1)
    pi, Q, R, K = sg.theorem_generator(problem, pi, A / pi[:, None], R)
    assert problem.residual(pi, Q, R, K) <= 1e-9
    model = _model_from_kernel(problem, CoupledSearchResult(pi, Q, R, K, _h(pi), 0.0, "theorem"), 1e-14)
    assert _equivalence_error(model, machine) <= 1e-8
    assert dec.lower <= dec.value + 1e-12


def test_theorem_generator_needs_cryptic_order_at_most_1() -> None:
    from sofic.examples.epsilon_machines import restricted_golden_mean

    problem = _problem(restricted_golden_mean(2))
    with pytest.raises(ValueError, match="cryptic order at most 1"):
        sg.theorem_generator(problem, np.ones(1), np.ones((1, problem.n_plus)), np.ones((1, problem.n_minus)))


def test_rectangle_free_components_are_exact() -> None:
    from sofic.generators.generator_bounds import one_wide_exact_common_information

    for P in (KUMAR_2, JOINT_52):
        dec = sg.separating_decomposition(P)
        value = one_wide_exact_common_information(P, atol=1e-13).value
        assert dec.exact and dec.value == pytest.approx(value, abs=1e-12) and dec.lower == dec.value
        assert dec.residual <= 1e-15


@pytest.fixture
def scip():
    return pytest.importorskip("pyscipopt")


def test_scip_second_opinion(scip) -> None:
    """Presolve can declare a feasible model infeasible, and OBBT can cut off the optimum; with both off SCIP is
    right on these joints, and the exact path keeps the smaller lower bound."""
    from sofic.generators.generator_bounds import one_wide_exact_common_information
    from sofic.generators.global_bound import separating_global_bound

    second = separating_global_bound(KUMAR_2, 2, presolve=False, obbt=False)
    assert second.status == "optimal" and second.primal == pytest.approx(1.0, abs=1e-5)
    assert separating_global_bound(KUMAR_2, 2).status in ("optimal", "infeasible")
    _, lower, _ = sg._scip_component(KUMAR_2, 2, 60.0)
    assert lower <= 1.0 + 1e-9

    value = one_wide_exact_common_information(JOINT_52, atol=1e-13).value
    second = separating_global_bound(JOINT_52, 3, presolve=False, obbt=False)
    assert second.status == "optimal" and second.primal == pytest.approx(value, abs=1e-5)
    assert value - 1e-4 <= second.lower <= value
    candidates, lower, _ = sg._scip_component(JOINT_52, 3, 60.0)
    assert lower <= value + 1e-9
    A, R, residual = sg._polish(JOINT_52, candidates, 3, 2, np.random.default_rng(0))
    assert residual <= 1e-12 and _h(A.sum(axis=1)) == pytest.approx(value, abs=1e-9)


def test_full_support_component_uses_scip(scip) -> None:
    P = np.array([[0.261439, 0.063794], [0.088097, 0.261439]])
    P = P / P.sum()
    dec = sg.separating_decomposition(P)
    assert not dec.exact and [c["method"] for c in dec.components] == ["scip"]
    assert dec.residual <= 1e-12 and dec.value - 1e-4 <= dec.lower <= dec.value
    assert len(dec.pi) <= sg.support_bound_states(P)


def test_missing_pyscipopt_falls_back_to_multistart(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pyscipopt", None)
    P = np.array([[0.3, 0.1], [0.1, 0.5]])
    dec = sg.separating_decomposition(P)
    assert [c["method"] for c in dec.components] == ["multistart"]
    assert dec.residual <= 1e-12 and dec.lower == 0.0 and 0.0 < dec.value <= 1.0 + 1e-12


def test_lookahead_generator_on_the_two_infinity_process() -> None:
    """The (2, inf) run-length-limited process (``restricted_golden_mean(2)``) has cryptic order 2 and
    C_g = 3/2; the lookahead generator reaches it with 3 states."""
    from sofic.examples.epsilon_machines import restricted_golden_mean

    machine = restricted_golden_mean(2)
    look = sg.lookahead_generator(machine)
    assert look.L == 2 and look.states == 3 and look.exact
    assert look.value == pytest.approx(1.5, abs=1e-12) and look.lower == pytest.approx(1.5, abs=1e-12)
    assert look.word_error <= 1e-12
    with pytest.raises(ValueError, match="p.chi"):
        sg.lookahead_generator(machine, 1)


# A 2-state binary process of cryptic order 1 with a full-support causal-state joint, where C_mu = 0.930387 > G
# and the support-type bounds are far below G.
SCIP_KAPPA_1 = {
    0: [[0.8620162340426342, 0.0], [0.26088622236842307, 0.0]],
    1: [[0.0, 0.1379837659573659], [0.0, 0.7391137776315769]],
}
SCIP_KAPPA_1_G = 0.7970965391094695


def test_pipeline_exact_path_with_a_scip_component(scip) -> None:
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.generators.generative_search import generative_model

    m = from_symbol_matrices((0, 1), (0, 1), {x: np.array(M) for x, M in SCIP_KAPPA_1.items()})
    machine = m if isinstance(m, EpsilonMachine) else EpsilonMachine.from_hmm(m)
    numerical = generative_model(machine, coverage="off", rng=np.random.default_rng(0))
    stages = {s["stage"]: s for s in numerical.stages}
    assert stages["exact path"]["components"] == ["scip"] and stages["exact path"]["states"] == 2
    assert numerical.lower_source == "separating global bound (numerical)" and not numerical.lower_certified
    assert numerical.upper == pytest.approx(SCIP_KAPPA_1_G, abs=1e-7)
    assert numerical.upper - 1e-4 <= numerical.lower <= numerical.upper
    certified = generative_model(machine, rng=np.random.default_rng(0))
    assert certified.certified and certified.lower_certified
    assert certified.upper == pytest.approx(SCIP_KAPPA_1_G, abs=1e-7)


@pytest.mark.parametrize("name", ["golden mean", "rip", "stretched gm 2"])
def test_pipeline_certifies_the_theorem_class(name) -> None:
    from sofic.generators.generative_search import generative_model

    machine = _topologies()[name]
    value = sg.separating_decomposition(_joint(_problem(machine))).value
    result = generative_model(machine, coverage="off", generator_heaviest_term=False, rng=np.random.default_rng(0))
    stages = {s["stage"]: s for s in result.stages}
    assert result.persistent_crypticity_theorem_applies and result.certified and result.lower_certified
    assert stages["persistent criteria"]["cryptic orders"][0] <= 1
    assert result.upper == pytest.approx(value, abs=1e-9)
    if "exact path" in stages:
        assert stages["exact path"]["states"] <= stages["exact path"]["support bound states"]
