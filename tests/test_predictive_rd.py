"""Tests for predictive rate-distortion (causal information bottleneck) curves."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import even_process, golden_mean, iid, nemo_process, nonunifilar_golden_mean, periodic
from sofic.generators import EpsilonMachine
from sofic.generators.predictive_rd import PredictiveRateDistortionCurve, predictive_rate_distortion

PROCESSES = {
    "golden_mean": lambda: golden_mean(0.5),
    "even": even_process,
    "period3": lambda: periodic("001"),
    "nemo": nemo_process,
}


@pytest.fixture(scope="module", params=sorted(PROCESSES))
def curve(request) -> tuple[EpsilonMachine, PredictiveRateDistortionCurve]:
    machine = EpsilonMachine.from_hmm(PROCESSES[request.param]())
    return machine, predictive_rate_distortion(machine, 30, beta_range=(0.25, 2000.0))


def test_endpoints(curve) -> None:
    machine, result = curve
    assert result.statistical_complexity == pytest.approx(machine.statistical_complexity())
    assert result.predictive_information <= machine.excess_entropy() + 1e-9
    assert result.predictive_information == pytest.approx(machine.excess_entropy(), abs=1e-6)
    small = result.beta <= 1.0
    np.testing.assert_allclose(result.rate[small], 0.0, atol=1e-9)
    np.testing.assert_allclose(result.relevant_information[small], 0.0, atol=1e-9)
    assert result.rate[-1] == pytest.approx(result.statistical_complexity, abs=1e-4)
    assert result.relevant_information[-1] == pytest.approx(result.predictive_information, abs=1e-6)
    np.testing.assert_allclose(result.distortion, result.predictive_information - result.relevant_information)


def test_monotone_and_bounded(curve) -> None:
    _machine, result = curve
    assert np.all(np.diff(result.beta) > 0)
    assert np.all(np.diff(result.rate) >= -1e-7)
    assert np.all(np.diff(result.relevant_information) >= -1e-7)
    assert np.all(result.relevant_information <= result.rate + 1e-9)
    assert np.all(result.rate <= result.statistical_complexity + 1e-9)
    assert np.all(result.distortion >= -1e-12)


def test_each_point_optimizes_its_lagrangian(curve) -> None:
    """Every computed point is feasible, so the one chosen at ``β`` must minimize ``I[S;R] − β I[R;Y]``."""
    _machine, result = curve
    for beta, rate, info in zip(result.beta, result.rate, result.relevant_information, strict=True):
        lagrangians = result.rate - beta * result.relevant_information
        assert rate - beta * info <= lagrangians.min() + 1e-6


def test_curve_is_concave(curve) -> None:
    _machine, result = curve
    rate, order = np.unique(np.round(result.rate, 6), return_index=True)
    info = result.relevant_information[order]
    if len(rate) < 3:
        return
    slopes = np.diff(info) / np.diff(rate)
    assert np.all(np.diff(slopes) <= 1e-3)
    assert np.all(slopes <= 1.0 + 1e-6)


def test_iid_curve_is_zero() -> None:
    result = predictive_rate_distortion(iid(3), 10)
    assert result.statistical_complexity == 0.0
    assert result.future_length == 0
    for values in (result.rate, result.relevant_information, result.distortion):
        np.testing.assert_allclose(values, 0.0, atol=1e-12)


def test_hmm_is_converted_to_epsilon_machine() -> None:
    from_hmm = predictive_rate_distortion(nonunifilar_golden_mean(), 20)
    direct = predictive_rate_distortion(golden_mean(0.5), 20)
    np.testing.assert_allclose(from_hmm.rate, direct.rate, atol=1e-8)
    np.testing.assert_allclose(from_hmm.relevant_information, direct.relevant_information, atol=1e-8)


def test_seed_determinism() -> None:
    machine = nemo_process()
    first = predictive_rate_distortion(machine, 12, seed=7)
    second = predictive_rate_distortion(machine, 12, seed=7)
    np.testing.assert_array_equal(first.rate, second.rate)
    np.testing.assert_array_equal(first.relevant_information, second.relevant_information)


def test_explicit_betas_and_future_length() -> None:
    result = predictive_rate_distortion(golden_mean(0.5), [100.0, 0.5, 5.0], future_length=3)
    np.testing.assert_array_equal(result.beta, [0.5, 5.0, 100.0])
    assert result.future_length == 3
    with pytest.raises(ValueError, match="nonnegative"):
        predictive_rate_distortion(golden_mean(0.5), [-1.0])
    with pytest.raises(ValueError, match="future_length"):
        predictive_rate_distortion(golden_mean(0.5), 3, future_length=-1)


def test_short_future_captures_less_than_excess_entropy() -> None:
    machine = even_process()
    short = predictive_rate_distortion(machine, [1000.0], future_length=2)
    assert short.predictive_information < machine.excess_entropy() - 1e-3


def test_warns_when_future_length_cap_is_hit() -> None:
    with pytest.warns(RuntimeWarning, match="did not reach E"):
        result = predictive_rate_distortion(even_process(), 3, max_future_length=2)
    assert result.future_length == 2
