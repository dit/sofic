"""Tests for spectral (Hankel-WFA) learning of stochastic processes."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sofic.examples import bernoulli, even_process, golden_mean
from sofic.generators.mealy import MealyHMM
from sofic.generators.nmachine import NMachine
from sofic.generators.quasi_realization import QuasiRealization
from sofic.inference.spectral import (
    SpectralInferenceError,
    learn_spectral_wfa,
    project_to_epsilon_machine,
    project_to_mealy,
    project_to_nmachine,
    spectral_singular_values,
)


def _all_words(alphabet, max_length):
    for length in range(1, max_length + 1):
        yield from itertools.product(alphabet, repeat=length)


def _max_word_error(model, learned, alphabet, max_length=6):
    return max(
        abs(model.word_probability(word) - learned.word_probability(word)) for word in _all_words(alphabet, max_length)
    )


# --- Exact-moment recovery -------------------------------------------------


def test_recovers_golden_mean_word_probabilities_exactly():
    model = golden_mean(0.4)
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3)
    assert isinstance(learned, QuasiRealization)
    assert learned.pi.shape[0] == 2  # golden mean has two causal states
    assert learned.pi.sum() == pytest.approx(1.0, abs=1e-9)
    assert _max_word_error(model, learned, alphabet) < 1e-9


def test_recovers_even_process():
    model = even_process(0.5)
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3)
    assert learned.pi.shape[0] == 2
    assert _max_word_error(model, learned, alphabet) < 1e-9


def test_fair_coin_is_rank_one():
    model = bernoulli()
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=2)
    assert learned.pi.shape[0] == 1
    for length in range(1, 5):
        for word in itertools.product(alphabet, repeat=length):
            assert learned.word_probability(word) == pytest.approx(0.5**length, abs=1e-9)


# --- Model-order selection -------------------------------------------------


def test_singular_value_spectrum_exposes_rank():
    model = golden_mean(0.4)
    alphabet = sorted(model.observation_alphabet, key=repr)
    spectrum = spectral_singular_values(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3)
    assert spectrum[0] >= spectrum[1] > 1e-6
    assert spectrum[2] < 1e-9  # exactly rank two


def test_explicit_rank_truncation():
    model = golden_mean(0.4)
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3, rank=1)
    assert learned.pi.shape[0] == 1  # forced rank-one approximation


# --- Empirical recovery ----------------------------------------------------


def test_recovers_golden_mean_from_samples():
    model = golden_mean(0.4)
    alphabet = sorted(model.observation_alphabet, key=repr)
    rng = np.random.default_rng(0)
    observations, _states = model.sample(40000, rng=rng)
    learned = learn_spectral_wfa(observations, prefix_length=3, rank=2)
    assert learned.pi.shape[0] == 2
    assert _max_word_error(model, learned, alphabet, max_length=4) < 0.02


def test_accepts_multiple_sequences():
    model = even_process(0.5)
    alphabet = sorted(model.observation_alphabet, key=repr)
    rng = np.random.default_rng(1)
    sequences = [model.sample(4000, rng=rng)[0] for _ in range(10)]
    learned = learn_spectral_wfa(sequences, alphabet=alphabet, prefix_length=3, rank=2)
    assert _max_word_error(model, learned, alphabet, max_length=4) < 0.03


# --- Projection back to generators -----------------------------------------


def test_project_to_nmachine_reproduces_word_probabilities():
    model = golden_mean(0.4)
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3)
    machine = project_to_nmachine(learned)
    assert isinstance(machine, NMachine)
    machine.validate()
    for word in _all_words(alphabet, 5):
        assert machine.word_probability(word) == pytest.approx(learned.word_probability(word), abs=1e-8)


def test_project_to_mealy_on_nonnegative_process():
    model = bernoulli()
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=2)
    machine = project_to_mealy(learned)
    assert isinstance(machine, MealyHMM)
    machine.validate()
    assert machine.word_probability(("0", "0")) == pytest.approx(0.25, abs=1e-9)


def test_project_to_mealy_rejects_signed_realization():
    model = golden_mean(0.4)
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3)
    with pytest.raises(SpectralInferenceError):
        project_to_mealy(learned)


def test_project_to_epsilon_machine_from_signed_golden_mean():
    from sofic.generators.epsilon_machine import EpsilonMachine

    model = golden_mean(0.5)
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3, rank=2)
    machine = project_to_epsilon_machine(learned)
    assert isinstance(machine, EpsilonMachine)
    machine.validate()
    assert len(list(machine.states())) == 2
    assert machine.entropy_rate() == pytest.approx(model.entropy_rate(), abs=1e-6)


def test_project_to_epsilon_machine_respects_max_states():
    model = golden_mean(0.5)
    alphabet = sorted(model.observation_alphabet, key=repr)
    learned = learn_spectral_wfa(word_probability=model.word_probability, alphabet=alphabet, prefix_length=3, rank=2)
    with pytest.raises(SpectralInferenceError, match="max_states"):
        project_to_epsilon_machine(learned, max_states=1)


# --- Input validation ------------------------------------------------------


def test_requires_sequences_or_word_probability():
    with pytest.raises(SpectralInferenceError):
        learn_spectral_wfa()


def test_alphabet_required_without_sequences():
    model = golden_mean(0.4)
    with pytest.raises(SpectralInferenceError):
        learn_spectral_wfa(word_probability=model.word_probability)


# --- Regressions -------------------------------------------------------------


@pytest.mark.parametrize("make", [golden_mean, even_process])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_epsilon_machine_spectral_from_samples_recovers_two_states_quickly(make, seed):
    """Regression: a relative rank cutoff kept sampling-noise singular values, and
    mixed-state extraction then enumerated noise states for seconds before failing."""
    import time

    from sofic.inference.spectral import learn_epsilon_machine_spectral

    model = make()
    observations, _ = model.sample(50_000, rng=np.random.default_rng(seed))
    assert spectral_singular_values(observations, prefix_length=3)[2] < 0.01  # noise, not structure
    start = time.perf_counter()
    machine = learn_epsilon_machine_spectral(observations)
    assert time.perf_counter() - start < 5.0
    assert len(list(machine.states())) == 2
    assert machine.entropy_rate() == pytest.approx(model.entropy_rate(), abs=0.02)


def test_sampled_rank_selection_uses_noise_floor():
    observations, _ = golden_mean().sample(50_000, rng=np.random.default_rng(0))
    assert learn_spectral_wfa(observations, prefix_length=3).pi.shape[0] == 2
    # Without the noise floor the relative threshold alone keeps the noise.
    assert learn_spectral_wfa(observations, prefix_length=3, noise_scale=0.0).pi.shape[0] > 2


def test_mixed_state_extraction_fails_fast_on_noise():
    import time

    observations, _ = even_process().sample(20_000, rng=np.random.default_rng(0))
    learned = learn_spectral_wfa(observations, prefix_length=3, rank=4)
    start = time.perf_counter()
    with pytest.raises(SpectralInferenceError, match="max_states"):
        project_to_epsilon_machine(learned, max_states=200)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(("make", "states"), [("period3", 3), ("golden", 2), ("even", 2)])
def test_epsilon_machine_spectral_exact_has_no_negligible_states(make, states):
    """Regression: 6-decimal belief rounding leaked ~1e-8 emission masses that seeded a
    spurious mixed state (period-3 came out with 4 states, one of mass 2.8e-8)."""
    from sofic.examples.processes import period
    from sofic.inference.spectral import learn_epsilon_machine_spectral

    model = {"period3": lambda: period(3), "golden": golden_mean, "even": even_process}[make]()
    alphabet = sorted(model.observation_alphabet, key=repr)
    machine = learn_epsilon_machine_spectral(word_probability=model.word_probability, alphabet=alphabet)
    assert len(list(machine.states())) == states
    assert min(machine.stationary_distribution()) > 1e-3
