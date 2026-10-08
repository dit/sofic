"""Entropy rates of non-unifilar HMM presentations: bounds, Blackwell estimates, and dispatch."""

import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings

from sofic.examples import golden_mean, nonunifilar_golden_mean, sns
from sofic.exceptions import MixedStateExplosionError
from sofic.generators.measures import entropy_rate, entropy_rate_blackwell, entropy_rate_bounds
from sofic.testing.strategies import markov_chains, mealy_hmms

pytestmark = pytest.mark.filterwarnings("ignore:stationary distribution is not unique:RuntimeWarning")


def _sns_entropy_rate() -> float:
    """SNS is a renewal process: ``0`` resynchronizes, and ``P(L = n) = (n - 1) / 2^n`` for ``n >= 2``."""
    n = np.arange(2, 400)
    p = (n - 1) / 2.0**n
    return float(-(p * np.log2(p)).sum() / (n * p).sum())


SNS_RATE = _sns_entropy_rate()


def test_sns_reference_value():
    assert pytest.approx(0.677867181055153, abs=1e-12) == SNS_RATE


@pytest.mark.parametrize("n", [0, 1, 3, 8, 14])
def test_bounds_bracket_sns_rate(n):
    lower, upper = entropy_rate_bounds(sns(), n)
    assert lower - 1e-12 <= SNS_RATE <= upper + 1e-12


def test_bounds_are_monotone_and_converge_for_sns():
    pairs = [entropy_rate_bounds(sns(), n) for n in range(16)]
    lowers, uppers = np.array(pairs).T
    assert np.all(np.diff(lowers) >= -1e-12)
    assert np.all(np.diff(uppers) <= 1e-12)
    assert uppers[-1] - lowers[-1] < 1e-7


def test_bounds_for_finite_epsilon_machine_presentation():
    lower, upper = entropy_rate_bounds(nonunifilar_golden_mean(), 10)
    assert lower == pytest.approx(2 / 3, abs=1e-9)
    assert upper == pytest.approx(2 / 3, abs=1e-9)


def test_bounds_reject_negative_length():
    with pytest.raises(ValueError, match="nonnegative"):
        entropy_rate_bounds(sns(), -1)


def test_exact_method_uses_finite_mixed_states():
    hmm = nonunifilar_golden_mean()
    assert not hmm.is_unifilar()
    assert hmm.entropy_rate(method="exact") == pytest.approx(golden_mean(0.5).entropy_rate())
    assert hmm.entropy_rate() == pytest.approx(2 / 3)


def test_exact_method_raises_when_mixed_states_explode():
    with pytest.raises(MixedStateExplosionError):
        sns().entropy_rate(method="exact", max_states=200)


def test_auto_falls_back_to_converged_bounds():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = sns().entropy_rate(max_states=200)
    assert value == pytest.approx(SNS_RATE, abs=1e-6)


def test_bounds_method_warns_when_not_converged():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        value = entropy_rate(sns(), "bounds", max_length=3)
    lower, upper = entropy_rate_bounds(sns(), 3)
    assert value == pytest.approx(0.5 * (lower + upper))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown entropy-rate method"):
        sns().entropy_rate(method="nope")


def test_unifilar_dispatch_is_unchanged():
    eps = golden_mean(0.5)
    assert eps.entropy_rate() == eps.entropy_rate(method="exact") == pytest.approx(2 / 3)


def test_blackwell_estimate_within_standard_errors():
    estimate, stderr = entropy_rate_blackwell(sns(), n_samples=40_000, burn_in=500, seed=1)
    assert stderr > 0.0
    assert abs(estimate - SNS_RATE) < 4 * stderr


def test_blackwell_is_seed_deterministic():
    first = entropy_rate_blackwell(sns(), n_samples=2_000, seed=7)
    second = entropy_rate_blackwell(sns(), n_samples=2_000, seed=np.random.default_rng(7))
    assert first == second


def test_blackwell_method_returns_estimate():
    value = sns().entropy_rate(method="blackwell", n_samples=2_000, seed=3)
    assert value == entropy_rate_blackwell(sns(), n_samples=2_000, seed=3).estimate


def test_blackwell_on_unifilar_process_is_exact_per_step():
    estimate, _stderr = entropy_rate_blackwell(golden_mean(0.5), n_samples=20_000, burn_in=100, seed=0)
    assert estimate == pytest.approx(2 / 3, abs=0.02)


@settings(max_examples=40, deadline=None)
@given(mealy_hmms(max_states=3))
def test_bounds_sandwich_property(hmm):
    pairs = [entropy_rate_bounds(hmm, n) for n in range(6)]
    lowers, uppers = np.array(pairs).T
    assert np.all(lowers <= uppers + 1e-9)
    assert np.all(np.diff(lowers) >= -1e-9)
    assert np.all(np.diff(uppers) <= 1e-9)


@settings(max_examples=30, deadline=None)
@given(markov_chains(max_states=3))
def test_markov_chain_bounds_match_exact_rate(chain):
    exact = chain.entropy_rate()
    lower, upper = entropy_rate_bounds(chain, 2)
    assert lower == pytest.approx(exact, abs=1e-9)
    assert upper == pytest.approx(exact, abs=1e-9)
    assert not math.isnan(entropy_rate(chain))
