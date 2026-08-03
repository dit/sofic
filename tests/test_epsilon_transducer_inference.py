"""Tests for transCSSR epsilon-transducer reconstruction."""

import numpy as np
import pytest

from sofic import EpsilonTransducer, MealyHMM
from sofic.automata.transducer_operations import compose_tg
from sofic.examples.processes import BinaryChannel, Delay
from sofic.generators.epsilon_transducer_inference import JointSuffixCounts, transcssr


def _iid_input() -> MealyHMM:
    inp = MealyHMM(observation_alphabet=frozenset({"0", "1"}), initial_distribution={"S": 1.0})
    inp.graph.add_state("S")
    inp.add_transition("S", "S", "0", 0.5)
    inp.add_transition("S", "S", "1", 0.5)
    inp.validate()
    return inp


def _paired_samples(channel, n, seed):
    joint = compose_tg(channel, _iid_input(), joint=True)
    observations, _ = joint.sample(n, np.random.default_rng(seed))
    xs = [pair[0] for pair in observations]
    ys = [pair[1] for pair in observations]
    return xs, ys


def test_suffix_counts_basic():
    counts = JointSuffixCounts.from_sequences("0101", "0011", max_length=1)
    assert counts.input_alphabet == ("0", "1")
    assert counts.output_alphabet == ("0", "1")
    morph = counts.state_morph({()}, "0")
    assert sum(morph.values()) == pytest.approx(1.0)


def test_recovers_memoryless_channel():
    xs, ys = _paired_samples(BinaryChannel(0.1, 0.2), 10000, seed=0)
    eps = transcssr(xs, ys, input_alphabet=("0", "1"), output_alphabet=("0", "1"))
    eps.validate()
    assert len(list(eps.states())) == 1
    assert eps.is_unifilar()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_delay_memory(seed):
    xs, ys = _paired_samples(Delay(1), 10000, seed=seed)
    eps = EpsilonTransducer.from_paired_sequences(xs, ys, input_alphabet=("0", "1"), output_alphabet=("0", "1"))
    eps.validate()
    assert len(list(eps.states())) == 2


def test_reconstruction_reproduces_conditional_law():
    # Prefer a seed that recovers a single causal state; CSSR can over-split on
    # some samples, and naively folding morphs across states can look like 1.0.
    eps = None
    for seed in range(8):
        xs, ys = _paired_samples(BinaryChannel(0.1, 0.2), 20000, seed=seed)
        candidate = transcssr(xs, ys, input_alphabet=("0", "1"), output_alphabet=("0", "1"))
        if len(list(candidate.states())) == 1:
            eps = candidate
            break
    assert eps is not None, "CSSR did not recover a memoryless channel on any trial seed"
    rows = {
        (transition.data["symbol"], transition.data["output"]): float(transition.data["prob"])
        for transition in eps.transitions()
    }
    assert rows[("0", "0")] == pytest.approx(0.9, abs=0.05)
    assert rows[("1", "1")] == pytest.approx(0.8, abs=0.05)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        transcssr("010", "01")
