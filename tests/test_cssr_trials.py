"""CSSR and transCSSR on independent trials: pooled counts, no cross-boundary histories."""

from collections import Counter

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sofic.automata.transducer_operations import compose_transducer_generator
from sofic.examples.epsilon_machines import even_process, golden_mean
from sofic.examples.processes import binary_channel
from sofic.generators.sampling import sample
from sofic.inference import Trials, topology_key
from sofic.inference.cssr import (
    JointSuffixCounts,
    SuffixCounts,
    learn_epsilon_machine_cssr,
    learn_epsilon_transducer_cssr,
)

from .test_epsilon_transducer_inference import _iid_input

trials_strategy = st.lists(st.lists(st.sampled_from("01"), min_size=1, max_size=12), min_size=1, max_size=5)


def _split(seq, k):
    bounds = np.linspace(0, len(seq), k + 1).astype(int)
    return Trials([seq[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)])


@given(trials_strategy)
def test_pooled_counts_are_the_sum_of_per_trial_counts(trials):
    """No history spans two trials: pooling equals summing each trial's own counts."""
    pooled = SuffixCounts.from_sequences(trials, alphabet="01", max_length=3)
    history, following = Counter(), Counter()
    for seq in trials:
        single = SuffixCounts.from_sequence(seq, alphabet="01", max_length=3)
        history.update(single.history_counts)
        following.update({(h, s): c for h, counter in single.next_counts.items() for s, c in counter.items()})
    assert pooled.history_counts == history
    assert Counter({(h, s): c for h, counter in pooled.next_counts.items() for s, c in counter.items()}) == following


@given(trials_strategy)
def test_joint_counts_never_span_trials(trials):
    outputs = [seq[::-1] for seq in trials]
    counts = JointSuffixCounts.from_sequences(Trials(trials), Trials(outputs), max_length=3)
    pairs = [tuple(zip(xs, ys, strict=True)) for xs, ys in zip(trials, outputs, strict=True)]
    for history in counts.history_counts:
        assert any(pair[i : i + len(history)] == history for pair in pairs for i in range(len(pair) - len(history) + 1))
    assert counts.history_counts[()] == sum(len(p) for p in pairs)


def test_single_sequence_is_one_trial():
    seq = "0110100111"
    assert SuffixCounts.from_sequence(seq, max_length=2) == SuffixCounts.from_sequences([seq], max_length=2)
    assert SuffixCounts.from_sequence(Trials([seq]), max_length=2) == SuffixCounts.from_sequence(seq, max_length=2)


def test_mismatched_trials_raise():
    with pytest.raises(ValueError, match="both be Trials"):
        JointSuffixCounts.from_sequences(Trials(["01"]), "01", max_length=1)
    with pytest.raises(ValueError, match="equal length"):
        JointSuffixCounts.from_sequences(Trials(["01", "1"]), Trials(["01", "10"]), max_length=1)


@pytest.mark.parametrize("factory", [golden_mean, even_process])
def test_cssr_on_trials_matches_the_unsplit_sample(factory):
    seq, _ = sample(factory(), 20000, np.random.default_rng(0))
    whole = learn_epsilon_machine_cssr(seq, max_history=3)
    split = learn_epsilon_machine_cssr(_split(list(seq), 4), max_history=3)
    assert topology_key(split) == topology_key(whole)
    assert split.entropy_rate() == pytest.approx(whole.entropy_rate(), abs=0.01)


def test_transcssr_on_trials_matches_the_unsplit_sample():
    joint = compose_transducer_generator(binary_channel(0.1, 0.2), _iid_input(), joint=True)
    observations, _ = joint.sample(20000, np.random.default_rng(1))
    xs = [pair[0] for pair in observations]
    ys = [pair[1] for pair in observations]
    whole = learn_epsilon_transducer_cssr(xs, ys, max_history=1)
    split = learn_epsilon_transducer_cssr(_split(xs, 3), _split(ys, 3), max_history=1)
    assert len(list(split.states())) == len(list(whole.states())) == 1
