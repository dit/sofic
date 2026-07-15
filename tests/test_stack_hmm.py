"""Tests for hidden Markov stack models."""

import numpy as np
import pytest

from sofic.exceptions import StochasticValidationError
from sofic.generators.stack_hmm import HiddenMarkovStackModel
from sofic.shifts.sofic_dyck import SoficDyckShift


def _dyck2(*, allow_empty_stack_returns: bool = True) -> HiddenMarkovStackModel:
    model = HiddenMarkovStackModel(
        initial_distribution={"q": 1.0},
        call_alphabet=frozenset({"a", "b"}),
        return_alphabet=frozenset({"A", "B"}),
        allow_empty_stack_returns=allow_empty_stack_returns,
    )
    model.graph.add_state("q")
    call_a = model.add_call_transition("q", "q", "a", 1.0)
    call_b = model.add_call_transition("q", "q", "b", 1.0)
    return_a = model.add_return_transition("q", "q", "A", 1.0)
    return_b = model.add_return_transition("q", "q", "B", 1.0)
    model.add_matched_pair(call_a, return_a)
    model.add_matched_pair(call_b, return_b)
    return model


def test_balanced_dyck_word_has_positive_probability():
    model = _dyck2()

    model.validate()
    assert model.word_probability(("a", "A")) > 0.0
    assert model.word_probability(("b", "B")) > 0.0


def test_mismatched_return_after_call_has_zero_probability():
    model = _dyck2()

    assert model.word_probability(("a", "B")) == 0.0
    assert model.word_probability(("b", "A")) == 0.0


def test_pending_calls_and_empty_stack_returns_follow_default_factor_convention():
    model = _dyck2()

    assert model.word_probability(("a",)) > 0.0
    assert model.word_probability(("A",)) > 0.0

    strict = _dyck2(allow_empty_stack_returns=False)
    assert strict.word_probability(("A",)) == 0.0


def test_sample_returns_observations_and_configurations():
    model = _dyck2()
    rng = np.random.default_rng(0)

    observations, configurations = model.sample(20, rng=rng)

    assert len(observations) == 20
    assert len(configurations) == 20
    assert set(observations) <= {"a", "b", "A", "B"}
    assert all(state == "q" for state, _stack in configurations)


def test_word_probability_sums_nondeterministic_paths():
    model = HiddenMarkovStackModel(
        initial_distribution={"q": 1.0},
        internal_alphabet=frozenset({"i", "j"}),
    )
    model.graph.add_state("q")
    model.add_internal_transition("q", "q", "i", 0.25)
    model.add_internal_transition("q", "q", "i", 0.25)
    model.add_internal_transition("q", "q", "j", 0.5)

    assert model.word_probability(("i",)) == pytest.approx(0.5)
    assert model.word_probability(("j",)) == pytest.approx(0.5)


def test_words_of_length_include_probabilities():
    model = HiddenMarkovStackModel(
        initial_distribution={"q": 1.0},
        internal_alphabet=frozenset({"0", "1"}),
    )
    model.graph.add_state("q")
    model.add_internal_transition("q", "q", "0", 0.25)
    model.add_internal_transition("q", "q", "1", 0.75)

    assert model.words_of_length(1) == {
        ("0",): pytest.approx(0.25),
        ("1",): pytest.approx(0.75),
    }


def test_stationary_control_marginal_for_finite_stack_truncation():
    model = HiddenMarkovStackModel(
        initial_distribution={"q0": 1.0},
        internal_alphabet=frozenset({"0", "1"}),
    )
    model.graph.add_state("q0")
    model.graph.add_state("q1")
    model.add_internal_transition("q0", "q1", "0", 1.0)
    model.add_internal_transition("q1", "q0", "1", 1.0)

    matrix = model.configuration_transition_matrix(max_stack_depth=0)
    pi = model.stationary_distribution(max_stack_depth=0)
    config_pi = model.stationary_distribution(max_stack_depth=0, marginal="configuration")

    assert matrix.shape == (2, 2)
    assert matrix.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert pi == pytest.approx([0.5, 0.5])
    assert config_pi.sum() == pytest.approx(1.0)


def test_conversion_to_and_from_sofic_dyck_shift_preserves_support():
    shift = SoficDyckShift(
        call_alphabet=frozenset({"a"}),
        return_alphabet=frozenset({"A"}),
    )
    shift.graph.add_state("q")
    call = shift.add_call_transition("q", "q", "a")
    ret = shift.add_return_transition("q", "q", "A")
    shift.add_matched_pair(call, ret)

    model = HiddenMarkovStackModel.from_sofic_dyck_shift(
        shift,
        probabilities={call: 0.6, ret: 0.4},
        initial_distribution={"q": 1.0},
    )
    support = model.to_sofic_dyck_shift()

    model.validate()
    support.validate()
    assert set(support.factor_language(2)) == set(shift.factor_language(2))
    assert model.word_probability(("a", "A")) > 0.0


def test_validate_rejects_negative_probability():
    model = HiddenMarkovStackModel(
        initial_distribution={"q": 1.0},
        internal_alphabet=frozenset({"x"}),
    )
    model.graph.add_state("q")
    model.add_internal_transition("q", "q", "x", -0.1)

    with pytest.raises(StochasticValidationError):
        model.validate()
