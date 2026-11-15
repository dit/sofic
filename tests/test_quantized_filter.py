"""Rounded filters and loss guarantees of Feng & Meng, arXiv:2611.13760."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from sofic import MooreHMM, QuantizedFilter, QuantizedFilterBounds, quantized_filter
from sofic.exceptions import StochasticValidationError
from sofic.generators.matrices import emission_tensors


def _moore(transition=None, emissions=None, initial=(0.5, 0.5)):
    transition = np.array([[0.8, 0.2], [0.2, 0.8]] if transition is None else transition, dtype=float)
    emissions = np.array([[0.9, 0.1], [0.1, 0.9]] if emissions is None else emissions, dtype=float)
    symbols = tuple(str(i) for i in range(emissions.shape[1]))
    model = MooreHMM(observation_alphabet=frozenset(symbols))
    if initial is not None:
        model.initial_distribution = dict(enumerate(initial))
    for i, row in enumerate(transition):
        for j, probability in enumerate(row):
            if probability:
                model.add_transition(i, j, float(probability))
        model.set_emission_distribution(i, dict(zip(symbols, emissions[i], strict=True)))
    return model


def _words(max_length):
    for length in range(max_length + 1):
        yield from product("01", repeat=length)


def _likelihood(matrices, word):
    vector = np.ones(len(next(iter(matrices.values()))))
    for symbol in reversed(word):
        vector = matrices[symbol] @ vector
    return vector


def _forecast_laws(model, observer, history, horizon):
    """Independent enumeration of the two laws and the source-prefix KL chain rule."""
    _, matrices = emission_tensors(model)
    initial = model.forward(history, normalize=True)[-1]
    nodes = [(initial, 1.0, observer.state_after(history))]
    chain_kl = 0.0
    for _ in range(horizon):
        following = []
        for mass, q_mass, state in nodes:
            p_mass = float(mass.sum())
            conditional = np.array([(mass @ matrices[y]).sum() / p_mass for y in observer.symbols])
            q_row = observer.output_probabilities[state]
            chain_kl += p_mass * float(np.sum(conditional * np.log2(conditional / q_row)))
            for column, symbol in enumerate(observer.symbols):
                following.append((mass @ matrices[symbol], q_mass * q_row[column], observer.successors[state, column]))
        nodes = following
    p = np.array([mass.sum() for mass, _, _ in nodes])
    q = np.array([mass for _, mass, _ in nodes])
    return p, q, chain_kl


def test_public_exports():
    from sofic.generators import QuantizedFilter as Filter
    from sofic.generators import QuantizedFilterBounds as Bounds
    from sofic.generators import quantized_filter as construct

    assert (Filter, Bounds, construct) == (QuantizedFilter, QuantizedFilterBounds, quantized_filter)


@pytest.mark.parametrize(
    ("loss", "horizon", "grid_states", "reachable"),
    [("event", None, 72, 29), ("kl", 100, 225, 51), ("kl", 10000, 2198, 183)],
)
def test_paper_table_3_log_grid_counts(loss, horizon, grid_states, reachable):
    # The paper's epsilon=0.1 is in nats; the public API uses bits.
    observer = quantized_filter(_moore(), 0.1 / math.log(2), loss=loss, horizon=horizon)
    assert observer.grid_states == grid_states
    assert observer.n_states == reachable
    assert observer.memory_bits == (reachable - 1).bit_length()
    assert observer.bounds.error_bound <= observer.bounds.tolerance * (1 + 1e-12)
    assert observer.bounds.max_rounding_error <= observer.bounds.rounding_error
    assert observer.bounds.contraction == pytest.approx(0.6)
    assert not observer.bounds.rigorous


@pytest.mark.parametrize("loss", ["event", "word", "kl"])
def test_finite_horizon_guarantees_and_kl_chain_rule(loss):
    model = _moore()
    horizon = 8
    observer = quantized_filter(model, 0.1, loss=loss, horizon=None if loss == "event" else horizon)
    for history in [(), ("1",), tuple("011010")]:
        p, q, chain_kl = _forecast_laws(model, observer, history, horizon)
        log_ratios = np.log2(p / q)
        direct_kl = float(p @ log_ratios)
        assert direct_kl == pytest.approx(chain_kl, abs=2e-14)
        assert direct_kl <= observer.bounds.total_kl_bound(horizon) + 2e-14
        assert np.max(np.abs(log_ratios)) <= observer.bounds.word_log_error_bound(horizon) + 2e-14
    assert observer.bounds.error_bound <= 0.1 * (1 + 1e-12)


@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_hilbert_and_event_bounds_after_all_short_histories(n_states):
    rng = np.random.default_rng(n_states)
    transition = rng.uniform(0.2, 1.0, (n_states, n_states))
    transition /= transition.sum(axis=1, keepdims=True)
    emissions = rng.uniform(0.2, 1.0, (n_states, 2))
    emissions /= emissions.sum(axis=1, keepdims=True)
    model = _moore(transition, emissions, initial=np.arange(1, n_states + 1) / sum(range(1, n_states + 1)))
    observer = quantized_filter(model, 0.5)
    _, matrices = emission_tensors(model)
    likelihoods = [_likelihood(matrices, word) for word in product("01", repeat=5)]
    for history in _words(5):
        exact = model.forward(history, normalize=True)[-1]
        state = observer.state_after(history)
        retained = observer.beliefs[state]
        distance = np.ptp(np.log(exact) - np.log(retained))
        assert distance <= observer.bounds.hilbert_error + 2e-14
        for likelihood in likelihoods:
            actual = observer.event_probability(likelihood, state=state)
            expected = float(exact @ likelihood)
            assert abs(math.log2(actual / expected)) <= observer.bounds.event_log_error_bound + 2e-14


def test_query_and_private_simulation_do_not_accumulate_rounding():
    model = _moore()
    observer = quantized_filter(model, 0.5)
    history = tuple("0110100111")
    state = observer.state_after(history)
    exact = model.forward(history, normalize=True)[-1]
    private = observer.private_generator(state=state)
    rounded = observer.to_generator()
    word = tuple("1" * 1024)
    true_log = model.log_word_probability(word, start=exact)
    private_error = abs(private.log_word_probability(word) - true_log)
    rounded_error = abs(rounded.log_word_probability(word, start=state) - true_log)
    assert private_error <= observer.bounds.event_log_error_bound + 1e-11
    assert rounded_error > 10 * observer.bounds.event_log_error_bound
    _, matrices = emission_tensors(model)
    likelihood = _likelihood(matrices, tuple("10110"))
    assert private.word_probability(tuple("10110")) == pytest.approx(
        observer.event_probability(likelihood, state=state)
    )
    observations, hidden = private.sample(20, np.random.default_rng(7))
    assert len(observations) == len(hidden) == 20


def test_generator_is_fixed_start_unifilar_and_serializable():
    model = _moore()
    observer = quantized_filter(model, 0.2, loss="word", horizon=20)
    generator = observer.to_generator()
    generator.validate()
    assert generator.is_unifilar()
    assert generator.initial_distribution == {0: 1.0}
    restored = type(generator).from_yaml(generator.to_yaml())
    for word in _words(4):
        state, probability = 0, 1.0
        for symbol in word:
            probability *= observer.output_probabilities[state, observer.symbols.index(symbol)]
            state = observer.step(state, symbol)
        assert generator.word_probability(word) == pytest.approx(probability)
        assert restored.word_probability(word) == pytest.approx(probability)


def test_initial_law_is_exact_and_explicit_start_overrides_model():
    model = _moore(initial=(0.1, 0.9))
    observer = quantized_filter(model, 0.1)
    np.testing.assert_array_equal(observer.beliefs[0], [0.1, 0.9])
    np.testing.assert_array_equal(observer.private_generator().forward([], normalize=True)[0], [0.1, 0.9])
    boundary = quantized_filter(model, 0.1, start=0)
    np.testing.assert_array_equal(boundary.beliefs[0], [1.0, 0.0])
    vector = quantized_filter(model, 0.1, start={0: 0.7, 1: 0.3})
    np.testing.assert_array_equal(vector.beliefs[0], [0.7, 0.3])
    assert observer.state_after("0110") == observer.state_after("10", state=observer.state_after("01"))


def test_unspecified_initial_uses_stationary_law():
    model = _moore([[0.7, 0.3], [0.1, 0.9]], initial=None)
    observer = quantized_filter(model, 0.5)
    np.testing.assert_allclose(observer.beliefs[0], [0.25, 0.75])


@pytest.mark.parametrize("initial", [(0.25, 0.75), (1.0, 0.0)])
def test_identical_transition_rows_are_exact(initial):
    model = _moore([[0.25, 0.75], [0.25, 0.75]], initial=initial)
    observer = quantized_filter(model, 0.1, loss="word", horizon=100)
    expected_states = 1 if initial == (0.25, 0.75) else 2
    assert observer.n_states == observer.grid_states == expected_states
    assert observer.bounds.error_bound == observer.bounds.hilbert_error == 0
    for word in _words(4):
        assert observer.to_generator().word_probability(word) == pytest.approx(model.word_probability(word))


def test_single_hidden_state():
    observer = quantized_filter(_moore([[1.0]], [[0.3, 0.7]], initial=(1.0,)), 0.1, max_states=1)
    assert observer.n_states == observer.grid_states == 1
    assert observer.memory_bits == 0
    assert observer.bounds.error_bound == 0


def test_zero_emissions_and_impossible_observations():
    model = _moore(emissions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], initial=(1.0, 0.0))
    observer = quantized_filter(model, 0.2)
    assert observer.event_probability([0.0, 0.0]) == 0
    with pytest.raises(ValueError, match="impossible"):
        observer.state_after("1")
    for label in range(observer.n_states):
        with pytest.raises(ValueError, match="impossible"):
            observer.step(label, "2")
    assert observer.state_after("010") >= 0
    generator = observer.to_generator()
    generator.validate()
    assert generator.word_probability("1") == 0
    assert generator.word_probability("02") == 0


def test_globally_rare_symbol_does_not_underflow_the_update():
    observer = quantized_filter(_moore(emissions=[[1 - 1e-300, 1e-300], [1 - 2e-300, 2e-300]]), 0.2)
    state = observer.state_after("11111")
    assert np.all(np.isfinite(observer.beliefs[state]))
    assert np.all(observer.beliefs[state] > 0)


def test_per_symbol_kl_allowance_has_horizon_independent_grid():
    short = quantized_filter(_moore(), 0.001 * 10, loss="kl", horizon=10)
    long = quantized_filter(_moore(), 0.001 * 1000, loss="kl", horizon=1000)
    assert short.grid_states == long.grid_states
    np.testing.assert_array_equal(short.successors, long.successors)
    np.testing.assert_allclose(short.beliefs, long.beliefs)


def test_large_tolerance_uses_finite_linear_kl_bound():
    observer = quantized_filter(_moore(), 10.0)
    assert observer.bounds.hilbert_error > 1.0
    assert observer.bounds.total_kl_bound(10) == observer.bounds.word_log_error_bound(10)
    assert observer.bounds.error_bound <= 10.0


def test_underflowing_positive_observation_is_rejected():
    model = _moore(emissions=[[1e-300, 1.0], [0.0, 1.0]], initial=(1e-300, 1.0))
    with pytest.raises(ValueError, match="underflows"):
        quantized_filter(model, 0.2)


def test_source_and_returned_models_do_not_mutate_observer():
    model = _moore()
    observer = quantized_filter(model, 0.2)
    expected = observer.private_generator().word_probability("011")
    model.set_emission_distribution(0, {"0": 1.0, "1": 0.0})
    private = observer.private_generator()
    private.set_emission_distribution(1, {"0": 1.0, "1": 0.0})
    observer.to_generator().initial_distribution = {1: 1.0}
    assert observer.private_generator().word_probability("011") == expected
    for array in (observer.beliefs, observer.successors, observer.output_probabilities):
        assert not array.flags.writeable


@pytest.mark.parametrize("tolerance", [0.0, -1.0, math.nan, math.inf, True])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        quantized_filter(_moore(), tolerance)


@pytest.mark.parametrize(
    "options",
    [
        {"loss": "unknown"},
        {"horizon": 1},
        {"loss": "kl"},
        {"loss": "word", "horizon": 0},
        {"loss": "kl", "horizon": True},
        {"loss": "kl", "horizon": 1.5},
        {"max_states": 0},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        quantized_filter(_moore(), 0.1, **options)


def test_reachable_cap_does_not_allocate_full_grid():
    transition = 0.4 * np.eye(5) + 0.6 / 5
    emissions = np.column_stack((np.linspace(0.1, 0.9, 5), np.linspace(0.9, 0.1, 5)))
    with pytest.raises(ValueError, match="max_states=3"):
        quantized_filter(_moore(transition, emissions, initial=np.ones(5) / 5), 0.01, max_states=3)


def test_general_mealy_is_rejected_even_with_positive_total_transition():
    with pytest.raises(TypeError, match="MooreHMM"):
        quantized_filter(_moore().to_mealy(), 0.1)


@pytest.mark.parametrize("transition", [[[1.0, 0.0], [0.2, 0.8]], [[math.nan, 0.2], [0.2, 0.8]]])
def test_invalid_hidden_transitions(transition):
    with pytest.raises((ValueError, StochasticValidationError)):
        quantized_filter(_moore(transition), 0.1)


@pytest.mark.parametrize("emissions", [[[0.7, 0.2], [0.2, 0.8]], [[math.nan, 0.1], [0.2, 0.8]]])
def test_invalid_emissions(emissions):
    with pytest.raises((ValueError, StochasticValidationError)):
        quantized_filter(_moore(emissions=emissions), 0.1)


def test_empty_model_and_invalid_start():
    with pytest.raises(ValueError, match="nonempty"):
        quantized_filter(MooreHMM(), 0.1)
    with pytest.raises(ValueError, match="initial"):
        quantized_filter(_moore(), 0.1, start=[0.2, 0.3])


@pytest.mark.parametrize("state", [-1, 100000, 1.5, True])
def test_invalid_state(state):
    observer = quantized_filter(_moore(), 0.5)
    with pytest.raises(ValueError, match="state"):
        observer.state_after([], state=state)
    with pytest.raises(ValueError, match="state"):
        observer.private_generator(state=state)


@pytest.mark.parametrize("likelihood", [[0.3], [0.2, math.inf], [-0.1, 0.5], [1.1, 0.5]])
def test_invalid_event_likelihood(likelihood):
    with pytest.raises(ValueError, match="likelihood"):
        quantized_filter(_moore(), 0.5).event_probability(likelihood)


def test_unknown_symbol_and_invalid_bound_horizon():
    observer = quantized_filter(_moore(), 0.5)
    with pytest.raises(ValueError, match="unknown"):
        observer.step(0, "unknown")
    for method in (observer.bounds.word_log_error_bound, observer.bounds.total_kl_bound):
        with pytest.raises(ValueError, match="horizon"):
            method(0)


@pytest.mark.parametrize("options", [{}, {"loss": "word", "horizon": 10**100}, {"loss": "kl", "horizon": 10**100}])
def test_unresolvable_precision_is_rejected(options):
    with pytest.raises(ValueError, match="float64"):
        quantized_filter(_moore(), 1e-20, **options)


def test_numerically_unresolvable_contraction_is_rejected():
    with pytest.raises(ValueError, match="float64"):
        quantized_filter(_moore([[1.0, 1e-100], [1e-100, 1.0]]), 0.1)
