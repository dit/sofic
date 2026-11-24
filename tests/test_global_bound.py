"""Restricted global lower bound (SCIP) on the state entropy of generators with at most ``k`` states."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from sofic.generators import global_bound as gb
from sofic.generators import hmm_search as hs


@pytest.fixture
def scip():
    return pytest.importorskip("pyscipopt")


def _two_state(p, q):
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.epsilon_machine import EpsilonMachine

    m = from_symbol_matrices((0, 1), (0, 1), {0: np.array([[p, 0], [q, 0]]), 1: np.array([[0, 1 - p], [1 - q, 0]])})
    return m if isinstance(m, EpsilonMachine) else EpsilonMachine.from_hmm(m)


def _bidir(forward):
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

    return BidirectionalEpsilonMachine.from_forward(forward)


def _h(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


GOLDEN_MEAN = (0.5, 1.0)
C_MU_GOLDEN = _h([2 / 3, 1 / 3])


def test_golden_mean_bound_is_the_statistical_complexity(scip) -> None:
    bidir = _bidir(_two_state(*GOLDEN_MEAN))
    one = gb.generator_global_bound(bidir, 1)
    assert one.status == "infeasible" and one.lower == math.inf  # the golden mean is not i.i.d.
    two = gb.generator_global_bound(bidir, 2)
    assert two.status == "optimal" and two.two_step
    assert C_MU_GOLDEN - 1e-4 <= two.lower <= C_MU_GOLDEN + 1e-9
    assert two.lower == pytest.approx(two.dual_bound - two.slack)


@pytest.mark.parametrize("presolve", [True, False])
def test_presolve_off_also_turns_off_symmetry(scip, presolve) -> None:
    m = scip.Model()
    default = m.getParam("misc/usesymmetry")
    gb._configure(m, scip, presolve=presolve, obbt=True, params=None)
    assert m.getParam("misc/usesymmetry") == (default if presolve else 0)
    assert m.getParam("propagating/obbt/freq") != -1


def test_objective_limit_semantics(scip) -> None:
    bidir = _bidir(_two_state(*GOLDEN_MEAN))
    below = gb.generator_global_bound(bidir, 2, objective_limit=C_MU_GOLDEN - 0.01)
    assert below.status == "infeasible" and below.seed is None
    assert below.lower == pytest.approx(C_MU_GOLDEN - 0.01 - below.slack)
    above = gb.generator_global_bound(bidir, 2, objective_limit=C_MU_GOLDEN + 0.03)
    assert above.status == "optimal" and above.seed is not None
    assert C_MU_GOLDEN - 1e-4 <= above.lower <= C_MU_GOLDEN + 1e-9
    assert above.primal == pytest.approx(C_MU_GOLDEN, abs=1e-4)


def test_pattern_c_point_bound_and_polish(scip) -> None:
    """At (p, q) = (0.836, 0.788) a 3-state generator of entropy H(1 - 2y, y, y) beats C_mu by 0.43 bits."""
    from sofic.generators.coupled_generators import CoupledSearchResult, _GeneratorProblem, _model_from_kernel
    from sofic.generators.minimal_generative_model import _equivalence_error

    p, q = 0.836, 0.788
    y = (1 - p) * (p - q) / ((2 - p) * (p * q + p - q))
    h_c = _h([1 - 2 * y, y, y])
    forward = _two_state(p, q)
    problem = _GeneratorProblem(_bidir(forward))
    res = gb.generator_global_bound(problem, 3, time_limit=300)
    assert res.status == "optimal"
    assert h_c - 2e-4 <= res.lower <= h_c
    pi, _, _, K = res.seed
    T = [K[:, x, :] for x in range(K.shape[1])]
    polished = hs.hmm_generator_search(forward, 3, starts=0, initial=[T])
    assert polished and polished[0].entropy == pytest.approx(h_c, abs=1e-5)
    seed = hs.coupled_seed_from_hmm(problem, polished[0].matrices)
    model = _model_from_kernel(problem, CoupledSearchResult(*seed, polished[0].entropy, 0.0, "t"), 1e-12)
    assert _equivalence_error(model, forward) < 1e-6


def _pipeline(bidir, **kwargs):
    from sofic.generators.generative_search import generative_model_from_bidirectional

    return generative_model_from_bidirectional(
        bidir,
        stages=("machines",),
        coverage="off",
        generator_heaviest_term=False,
        early_certified=False,
        rng=np.random.default_rng(0),
        **kwargs,
    )


def test_pipeline_records_the_restricted_bound_only(scip) -> None:
    bidir = _bidir(_two_state(*GOLDEN_MEAN))
    off = _pipeline(bidir)
    on = _pipeline(bidir, global_bound=True, global_bound_states=2)
    assert off.restricted_lower is None and off.restricted_states is None
    assert on.restricted_states == 2
    assert C_MU_GOLDEN - 1e-4 <= on.restricted_lower <= on.upper + 1e-9
    assert (on.lower, on.lower_source, on.certified) == (off.lower, off.lower_source, off.certified)
    summary = on.stages[-1]
    assert summary["stage"] == "reliability summary" and summary["restricted_states"] == 2
    assert any(s["stage"] == "restricted global bound" and "value" in s for s in on.stages)


def test_missing_pyscipopt(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pyscipopt", None)
    bidir = _bidir(_two_state(*GOLDEN_MEAN))
    with pytest.raises(ImportError, match=r"sofic\[global\]"):
        gb.generator_global_bound(bidir, 2)
    result = _pipeline(bidir, global_bound=True)
    assert result.restricted_lower is None
    assert any(s["stage"] == "restricted global bound" and "skipped" in s for s in result.stages)
