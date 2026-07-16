"""Tests for the computational-mechanics epsilon-transducer."""

import pytest

from sofic import EpsilonTransducer, MealyHMM
from sofic.automata.transducer_operations import compose_tg
from sofic.examples.processes import GME, RCT, BinaryChannel, GMtoEven
from sofic.exceptions import StochasticValidationError, UnifilarityError
from sofic.graph import ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL


def _iid_input() -> MealyHMM:
    inp = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"S": 1.0})
    inp.graph.add_state("S")
    inp.add_transition("S", "S", "0", 0.5)
    inp.add_transition("S", "S", "1", 0.5)
    inp.validate()
    return inp


@pytest.mark.parametrize(
    ("channel", "expected_states"),
    [
        (BinaryChannel(0.1, 0.2), 1),
        (GMtoEven(), 2),
        (RCT(0.5), 3),
        (GME(), 2),
    ],
)
def test_from_channel_minimizes(channel, expected_states):
    eps = EpsilonTransducer.from_channel(channel)
    eps.validate()
    assert len(list(eps.states())) == expected_states
    assert eps.is_unifilar()


def test_memoryless_channel_is_single_causal_state():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    assert len(eps.causal_states()) == 1
    rows = {}
    for transition in eps.transitions():
        key = (transition.data[ATTR_SYMBOL], transition.data[ATTR_OUTPUT])
        rows[key] = float(transition.data[ATTR_PROB])
    assert rows[("0", "0")] == pytest.approx(0.9)
    assert rows[("0", "1")] == pytest.approx(0.1)
    assert rows[("1", "1")] == pytest.approx(0.8)
    assert rows[("1", "0")] == pytest.approx(0.2)


def test_initial_distribution_normalized():
    eps = EpsilonTransducer.from_channel(GMtoEven())
    assert sum(eps.initial_distribution.values()) == pytest.approx(1.0)
    assert set(eps.initial_distribution) <= set(eps.states())


def test_non_unifilar_channel_rejected():
    tr = _non_unifilar_channel()
    with pytest.raises(StochasticValidationError):
        EpsilonTransducer.from_channel(tr)


def _non_unifilar_channel():
    from sofic.automata.transducers import MealyMachine

    tr = MealyMachine(
        input_alphabet=frozenset({"0"}),
        output_alphabet=frozenset({"0"}),
        initial_states=frozenset({"A"}),
    )
    tr.graph.add_state("A")
    tr.graph.add_state("B")
    tr.graph.add_state("C")
    # Two successors for the same (state, input, output): not joint-unifilar.
    tr.add_transition("A", "B", "0", "0", prob=0.5)
    tr.add_transition("A", "C", "0", "0", prob=0.5)
    tr.add_transition("B", "A", "0", "0", prob=1.0)
    tr.add_transition("C", "A", "0", "0", prob=1.0)
    return tr


def test_validate_rejects_bad_initial_distribution():
    eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))
    eps.initial_distribution = {next(iter(eps.states())): 0.5}
    with pytest.raises(StochasticValidationError):
        eps.validate()


def test_validate_rejects_non_unifilar_direct():
    # Stochastic rows sum to 1, but (A, '0', '0') has two successors: not unifilar.
    eps = EpsilonTransducer(
        input_alphabet=frozenset({"0"}),
        output_alphabet=frozenset({"0"}),
        initial_states=frozenset({"A"}),
        initial_distribution={"A": 1.0},
    )
    eps.graph.add_state("A")
    eps.graph.add_state("B")
    eps.graph.add_transition("A", "A", **{ATTR_SYMBOL: "0", ATTR_OUTPUT: "0", ATTR_PROB: 0.5})
    eps.graph.add_transition("A", "B", **{ATTR_SYMBOL: "0", ATTR_OUTPUT: "0", ATTR_PROB: 0.5})
    eps.graph.add_transition("B", "A", **{ATTR_SYMBOL: "0", ATTR_OUTPUT: "0", ATTR_PROB: 1.0})
    with pytest.raises(UnifilarityError):
        eps.validate()


def test_from_joint_generator_recovers_memoryless():
    joint = compose_tg(BinaryChannel(0.1, 0.2), _iid_input(), joint=True)
    eps = EpsilonTransducer.from_joint_generator(joint)
    eps.validate()
    assert len(list(eps.states())) == 1
    assert eps.output_alphabet == frozenset({"0", "1"})


def test_from_iohmm_alias():
    eps = EpsilonTransducer.from_iohmm(GMtoEven())
    assert isinstance(eps, EpsilonTransducer)
    assert len(list(eps.states())) == 2


def test_yaml_round_trip():
    eps = EpsilonTransducer.from_channel(RCT(0.5))
    restored = EpsilonTransducer.from_yaml(eps.to_yaml())
    assert isinstance(restored, EpsilonTransducer)
    assert len(list(restored.states())) == len(list(eps.states()))
    assert restored.initial_distribution == pytest.approx(eps.initial_distribution)


def test_wfst_round_trip_preserves_structure():
    eps = EpsilonTransducer.from_channel(GMtoEven())
    wfst = eps.to_wfst()
    recovered = EpsilonTransducer.from_wfst(wfst)
    assert len(list(recovered.states())) == len(list(eps.states()))
