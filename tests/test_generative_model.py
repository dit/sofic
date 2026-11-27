"""Tests for generative_model entry point."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sofic.examples import golden_mean, golden_mean_bidirectional
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators import generative_search as gs
from sofic.generators.bidirectional_build import try_build_bidirectional
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.generative_search import generative_model, generative_model_from_bidirectional

# Lower bounds of the truncation that must not reach the result of a truncated build.
_TRUNCATION_BOUNDS = {
    "excess entropy",
    "exact rectangle partition",
    "support bound",
    "coverage bound (certified)",
    "generator coverage bound (certified)",
    "heaviest-term bound (certified)",
    "generator heaviest-term bound (certified)",
}


def _truncated_forward():
    """A 2-state ternary process whose reverse ε-machine is infinite (s3-n2-t0022-d2 of the ternary survey)."""
    mats = {
        0: np.array([[0.0, 0.0], [0.8718949197791982, 0.0]]),
        1: np.array([[0.2089291369568694, 0.0], [0.0, 0.05375082082018049]]),
        2: np.array([[0.0, 0.7910708630431306], [0.0, 0.07435425940062132]]),
    }
    return from_symbol_matrices((0, 1), (0, 1, 2), mats)


def test_a_prebuilt_truncation_is_classed_truncated() -> None:
    built = try_build_bidirectional(_truncated_forward())
    assert built.preflight.bidir_class == "truncated"
    bidir, meta = gs.resolve_bidirectional(built.bidir)
    assert bidir is built.bidir
    assert meta == {"bidir_class": "truncated", "source": "bidirectional input"}


def test_a_window_lower_bound_replaces_the_lower_bounds_of_the_machine() -> None:
    result = gs._bracket_search(
        golden_mean_bidirectional(),
        objective="entropy",
        stages=("machines",),
        window_lower=(0.5, "a bound"),
        rng=np.random.default_rng(0),
    )
    assert (result.lower, result.lower_source, result.lower_certified) == (0.5, "a bound", True)
    assert not result.certified
    assert not {s["stage"] for s in result.stages} & _TRUNCATION_BOUNDS


@pytest.mark.slow
def test_a_truncated_build_reports_the_future_window_bound() -> None:
    start = time.monotonic()
    result = generative_model(_truncated_forward(), time_budget=60.0, rng=np.random.default_rng(0))
    assert time.monotonic() - start < 66.0
    assert result.stages[0]["bidir_class"] == "truncated"
    assert result.lower_source.startswith("future-window generator coverage bound (L=")
    # without a deadline the window bounds are 0.9978545 (L = 2) and 0.9981857 (L = 3); the best generator known
    # has entropy 0.9982954
    assert 0.99 < result.lower < result.upper
    assert not result.certified
    assert not {s["stage"] for s in result.stages} & _TRUNCATION_BOUNDS
    wyner = gs.minimal_wyner_model(_truncated_forward(), time_budget=20.0, rng=np.random.default_rng(0))
    assert (wyner.lower, wyner.lower_source) == (0.0, "trivial lower bound")


def test_generative_model_from_forward_epsilon_machine_certifies_golden_mean() -> None:
    forward = golden_mean(0.5)
    rng = np.random.default_rng(0)
    result = generative_model(
        forward,
        max_states=3,
        starts=8,
        exact_starts=15,
        entropy_sweep_starts=20,
        rng=rng,
    )
    assert result.stages[0]["stage"] == "bidirectional build"
    assert result.stages[-1]["stage"] == "reliability summary"
    assert result.stages[-1]["certified"] is True
    assert result.certified
    target = float(-(1 / 3) * np.log2(1 / 3) - (2 / 3) * np.log2(2 / 3))
    assert abs(result.upper - target) < 1e-5
    assert "universal coupled" not in result.upper_source
    assert result.reproduction_error is not None and result.reproduction_error < 1e-9
    assert result.structure is not None
    assert result.structure["n_states"] == 2
    assert result.structure["functional"] and result.structure["lifted"]
    assert result.stages[-1]["structure"] == result.structure
    dec = result.decomposition
    assert dec is not None and len(dec["states"]) == 2
    assert np.allclose(np.sum(dec["pi"]), 1.0)
    assert np.allclose(np.sum(dec["K"], axis=(1, 2)), 1.0)
    # weights, beliefs and the model agree: pi Q is the forward causal-state distribution
    assert np.allclose(np.array(dec["pi"]) @ np.array(dec["Q"]), [2 / 3, 1 / 3]) or np.allclose(
        np.array(dec["pi"]) @ np.array(dec["Q"]), [1 / 3, 2 / 3]
    )
    assert sorted(dec["pi"]) == pytest.approx(sorted(result.structure["weights"]))


def test_certified_requires_the_model_to_reproduce_the_source() -> None:
    from sofic.generators.generative_search import GenerativeModelResult

    def result(error):
        return GenerativeModelResult(
            model=None,
            upper=1.0,
            lower=1.0,
            upper_source="test",
            lower_source="test",
            exact_common_information_estimate=None,
            separating_variables_generate=False,
            reproduction_error=error,
        )

    assert result(None).certified
    assert result(1e-9).certified
    assert not result(1e-4).certified


def test_upper_is_the_entropy_of_the_verified_start_vector() -> None:
    """A near-zero-weight absorbing state makes the model reducible; its other stationary vector must not set the upper."""
    from sofic.generators.coupled_generators import _GeneratorProblem
    from sofic.generators.generative_search import _try_best_candidate
    from sofic.generators.hmm_search import coupled_seed_from_hmm
    from sofic.generators.matrices import symbol_matrices

    bidir = golden_mean_bidirectional(0.5)
    problem = _GeneratorProblem(bidir)
    mats = symbol_matrices(bidir.forward_machine)
    pi, Q, R, K = coupled_seed_from_hmm(problem, tuple(np.asarray(mats[x], float) for x in problem.symbols))
    eps, k = 1e-7, len(pi)
    pi2 = np.append((1 - eps) * pi, eps)
    K2 = np.zeros((k + 1, K.shape[1], k + 1))
    K2[:k, :, :k] = K
    loop = next(i for i, x in enumerate(problem.symbols) if np.trace(np.asarray(mats[x], float)) > 0)
    K2[k, loop, k] = 1.0  # absorbing, repeats a symbol the golden mean can repeat forever
    seed = (pi2, np.vstack([Q, Q[0]]), np.vstack([R, R[0]]), K2)
    result = _try_best_candidate(
        problem,
        bidir.forward_machine,
        6,
        [(0.0, "test", seed)],
        lower=0.0,
        lower_source="test",
        lower_certified=False,
        g_estimate=None,
        theorem=False,
        n_g_optima=0,
        n_generating=0,
        cg_exceeds_g=None,
        log=[],
        reproduction_atol=1e-5,
    )
    assert result is not None
    target = float(-(pi2 * np.log2(pi2)).sum())
    assert result.upper == pytest.approx(target, abs=1e-9)
    assert result.upper >= result.lower


def test_generative_model_accepts_bidirectional_input() -> None:
    bidir = golden_mean_bidirectional(0.5)
    assert isinstance(bidir, BidirectionalEpsilonMachine)
    result = generative_model(bidir, max_states=3, starts=6, rng=np.random.default_rng(1))
    assert result.stages[0]["source"] == "bidirectional input"
    assert result.certified


def test_generative_model_from_bidirectional_matches_pipeline_on_bidir() -> None:
    bidir = golden_mean_bidirectional(0.5)
    full = generative_model(bidir, max_states=3, starts=4, rng=np.random.default_rng(2))
    direct = generative_model_from_bidirectional(
        bidir,
        max_states=3,
        starts=4,
        universal_coupled=True,
        early_theorem_g_exit=True,
        rng=np.random.default_rng(2),
    )
    assert full.certified and direct.certified
    assert abs(full.upper - direct.upper) < 1e-5


def test_zero_time_budget_returns_a_valid_bracket_from_the_machines() -> None:
    from tests.test_generator_caps import _survey_machine

    result = generative_model_from_bidirectional(_survey_machine(), time_budget=0.0, rng=np.random.default_rng(0))
    assert result.budget_exhausted
    machines = {
        "epsilon machine",
        "reverse epsilon machine",
        "bidirectional machine",
        "canonical RFSA generator",
        "maximized prime átomaton generator",
    }
    assert result.upper_source in machines
    assert result.lower <= result.upper + 1e-9
    skipped = {s["stage"] for s in result.stages if s.get("skipped") == "time budget exhausted"}
    assert {"generator coverage bound", "exact common information", "coupled search"} <= skipped
    assert all("elapsed" in s for s in result.stages[:-1])
    assert result.stages[-1]["budget_exhausted"] is True


def test_generator_heaviest_term_stage_runs_last_on_open_small_brackets() -> None:
    from tests.test_exact_ci_bounds import _dataset_case_t006

    bidir = _dataset_case_t006().bidir
    result = generative_model_from_bidirectional(bidir, time_budget=0.0, rng=np.random.default_rng(0))
    names = [s["stage"] for s in result.stages]
    assert names[-2] == "generator heaviest-term bound"
    assert result.stages[-2]["skipped"] == "time budget exhausted"
    off = generative_model_from_bidirectional(
        bidir, time_budget=0.0, generator_heaviest_term=False, rng=np.random.default_rng(0)
    )
    assert "generator heaviest-term bound" not in {s["stage"] for s in off.stages}


def test_generator_heaviest_term_stage_keeps_its_share_of_the_budget(monkeypatch) -> None:
    import time

    import sofic.generators.generative_search as gs
    from tests.test_exact_ci_bounds import _dataset_case_t006

    bidir = _dataset_case_t006().bidir
    start, left = time.monotonic(), []
    real = gs.generator_heaviest_term_bound

    def slow_chains(problem, **kwargs):
        time.sleep(max(0.0, start + 4.5 - time.monotonic()))
        return []

    def final(problem, **kwargs):
        left.append(kwargs["deadline"] - time.monotonic())
        return real(problem, **kwargs)

    monkeypatch.setattr(gs, "simplex_chain_search", slow_chains)
    monkeypatch.setattr(gs, "generator_heaviest_term_bound", final)
    result = generative_model_from_bidirectional(bidir, time_budget=5.0, rng=np.random.default_rng(0))
    skipped = {s["stage"] for s in result.stages if s.get("skipped") == "time budget exhausted"}
    assert "memory lifts" in skipped and "generator heaviest-term bound" not in skipped
    assert left and left[0] > 0.0


def test_lower_bounds_before_the_searches_share_the_whole_budget(monkeypatch) -> None:
    import time

    import sofic.generators.generative_search as gs
    import sofic.generators.generator_caps as caps
    from tests.test_exact_ci_bounds import _dataset_case_t006

    given = []
    real = caps.generator_coverage_bound

    def coverage(problem, **kwargs):
        given.append(kwargs["deadline"] - time.monotonic())
        return real(problem, **kwargs)

    monkeypatch.setattr(gs, "_GENERATOR_HEAVIEST_TERM_SHARE", 0.9)
    monkeypatch.setattr(caps, "generator_coverage_bound", coverage)
    generative_model_from_bidirectional(_dataset_case_t006().bidir, time_budget=5.0, rng=np.random.default_rng(0))
    assert given and given[0] > 1.0


def test_unlimited_budget_is_not_reported_as_exhausted() -> None:
    result = generative_model(golden_mean(0.5), rng=np.random.default_rng(0))
    assert not result.budget_exhausted


def test_settled_bracket_skips_the_coverage_stages() -> None:
    import time

    from sofic.examples.epsilon_machines import butterfly_process

    start = time.monotonic()
    result = generative_model(butterfly_process(), time_budget=60.0, rng=np.random.default_rng(0))
    assert time.monotonic() - start < 60.0  # took about 1900 s when words were enumerated over all 8**8
    stages = {s["stage"]: s for s in result.stages}
    assert result.certified and result.upper == pytest.approx(np.log2(5), abs=1e-9)
    assert stages["coverage bounds"]["skipped"] == "the lower bound meets an ε-machine's entropy"
    assert "generator coverage bound (certified)" not in stages


def test_generator_coverage_waits_for_the_exact_path_in_the_theorem_class() -> None:
    from sofic.examples.epsilon_machines import from_symbol_matrices
    from sofic.generators.epsilon_machine import EpsilonMachine
    from tests.test_separating_generators import SCIP_KAPPA_1, SCIP_KAPPA_1_G

    m = from_symbol_matrices((0, 1), (0, 1), {x: np.array(M) for x, M in SCIP_KAPPA_1.items()})
    machine = m if isinstance(m, EpsilonMachine) else EpsilonMachine.from_hmm(m)
    result = generative_model(machine, time_budget=60.0, rng=np.random.default_rng(0))
    names = [s["stage"] for s in result.stages]
    assert result.persistent_crypticity_theorem_applies and result.certified
    assert result.upper == pytest.approx(SCIP_KAPPA_1_G, abs=1e-7)
    assert "exact path" in names and "generator coverage bound (certified)" not in names


def test_an_inconclusive_exact_path_is_followed_by_the_heaviest_term_bound() -> None:
    from sofic.examples.epsilon_machines import from_symbol_matrices

    mats = {
        0: np.array([[0.0, 0.0], [0.28489650228026714, 0.0]]),
        1: np.array([[0.19631079625699752, 0.0], [0.008208977634958007, 0.0]]),
        2: np.array([[0.0, 0.8036892037430026], [0.0, 0.7068945200847747]]),
    }
    machine = from_symbol_matrices((0, 1), (0, 1, 2), mats)
    result = generative_model(machine, time_budget=60.0, rng=np.random.default_rng(0))
    names = [s["stage"] for s in result.stages]
    exact = result.stages[names.index("exact path")]
    assert result.persistent_crypticity_theorem_applies and result.certified
    assert result.upper - exact["lower"] > 1e-6
    assert result.lower_source == "heaviest-term bound (certified)"
    assert names.index("heaviest-term bound (certified)") == names.index("exact path") + 2
    assert "exact common information" not in names


def test_dit_fallbacks_are_skipped_on_large_joints(monkeypatch) -> None:
    import sofic.generators.generative_search as gs
    import sofic.generators.lifted_common_information as lci
    from tests.test_generator_caps import _survey_machine

    def fail(*args, **kwargs):
        raise AssertionError("dit fallback ran")

    monkeypatch.setattr(gs, "_DIT_MAX_PAIRS", 0)
    monkeypatch.setattr(lci, "_unconstrained_channel", fail)
    result = generative_model_from_bidirectional(
        _survey_machine(),
        stages=("machines", "exact"),
        exact_starts=2,
        coverage="off",
        generator_caps=0,
        early_certified=False,
        time_budget=5.0,
        rng=np.random.default_rng(0),
    )
    skipped = [s for s in result.stages if s["stage"] == "dit fallbacks"]
    assert skipped and skipped[0]["skipped"] == "more than 0 causal-state pairs"
