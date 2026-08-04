"""Tests for channel information measures on transducers."""

import pytest

from sofic import EpsilonTransducer, MealyHMM
from sofic.examples.processes import BinaryChannel, Delay
from sofic.generators.channel_measures import (
    channel_statistical_complexity,
    directed_information,
    driven_entropy_rate,
    transfer_entropy,
)


def _iid_input() -> MealyHMM:
    inp = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"S": 1.0})
    inp.graph.add_state("S")
    inp.add_transition("S", "S", "0", 0.5)
    inp.add_transition("S", "S", "1", 0.5)
    inp.validate()
    return inp


def test_memoryless_channel_zero_complexity():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    assert channel_statistical_complexity(eps, _iid_input()) == pytest.approx(0.0, abs=1e-9)


def test_memoryless_channel_zero_transfer_entropy():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    assert transfer_entropy(eps, _iid_input()) == pytest.approx(0.0, abs=1e-9)


def test_memory_channel_positive_complexity():
    eps = EpsilonTransducer.from_channel(Delay(1))
    assert channel_statistical_complexity(eps, _iid_input()) > 0.5


def test_driven_entropy_rate_bounds():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    rate = driven_entropy_rate(eps, _iid_input())
    assert 0.0 <= rate <= 1.0 + 1e-9


def test_directed_information_positive():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    assert directed_information(eps, _iid_input(), length=2) > 0.0


def test_complete_skips_unused_reject_sink():
    """Already-total channels must not gain an unused absorbing ``?`` state."""
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    completed = eps.complete(frozenset({"0", "1"}))
    assert "?" not in completed.states()
    assert list(completed.states()) == list(eps.states())


def test_directed_information_ignores_unreachable_error_class():
    """Reducible joints with an unused error sink must keep positive DI.

    ``compose_tg(..., complete=True)`` used to always attach an absorbing ``?``
    component; the eigenvector stationary law is then non-unique and can put all
    mass on ``?`` (PYTHONHASHSEED-dependent), zeroing directed information.
    """
    from sofic.automata.transducer_operations import compose_tg
    from sofic.generators.directional_flow import directed_information as di_flow
    from sofic.generators.hmm_inference import _stationary_emission_tensors

    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    # Force the historical reducible joint even after complete() stops adding an
    # unused sink: compose with an explicit completed copy that includes ``?``.
    completed = eps.copy()
    completed.graph.add_state("?")
    completed.add_transition("?", "?", "0", "?", prob=1.0)
    completed.add_transition("?", "?", "1", "?", prob=1.0)
    joint = compose_tg(completed, _iid_input(), joint=True, complete=False)
    assert ("S", "?") in list(joint.states())
    pi, _tensors = _stationary_emission_tensors(joint)
    idx = joint.reindex()
    error_index = idx.index(("S", "?"))
    assert pi[error_index] == pytest.approx(0.0, abs=1e-12)
    assert di_flow(joint, source="x", target="y", length=2) > 0.0


def test_method_dispatch_matches_functions():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    inp = _iid_input()
    assert eps.statistical_complexity(inp) == pytest.approx(channel_statistical_complexity(eps, inp))
    assert eps.transfer_entropy(inp) == pytest.approx(transfer_entropy(eps, inp))
