"""Finite-state channels attached to dit channel codes."""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples.processes import binary_channel, ising_channel, trapdoor
from sofic.generators import FiniteStateChannel

dit_coding = pytest.importorskip("dit.coding")
if not hasattr(dit_coding, "as_channel"):
    pytest.skip("dit.coding has no channel protocol", allow_module_level=True)

from dit.coding import as_channel, binary_symmetric_channel, hamming, repetition  # noqa: E402
from dit.coding._channel import symbol_llrs  # noqa: E402

CHANNELS = [trapdoor, ising_channel]


def path_sum(machine, codeword, received):
    """``p(received | codeword)`` summed over state paths, from the raw machine."""
    matrices, states = machine.labeled_transition_matrices()
    from sofic.generators.capacity._channel import initial_law

    pi = initial_law(machine, states)
    total = 0.0
    for path in itertools.product(range(len(states)), repeat=len(codeword) + 1):
        prob = pi[path[0]]
        for t, (x, y) in enumerate(zip(codeword, received, strict=True)):
            matrix = matrices.get((str(x), str(y)))
            prob *= 0.0 if matrix is None else matrix[path[t], path[t + 1]]
        total += prob
    return total


words = st.integers(1, 5).flatmap(
    lambda n: st.tuples(
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
    )
)


def test_mealy_machine_is_a_dit_channel():
    channel = as_channel(trapdoor())
    assert isinstance(channel, FiniteStateChannel)
    assert channel.inputs == (0, 1)
    assert channel.outputs == (0, 1)


@pytest.mark.parametrize("factory", CHANNELS)
@settings(max_examples=40, deadline=None)
@given(pair=words)
def test_likelihood_matches_path_sum(factory, pair):
    machine = factory()
    codeword, received = pair
    expected = path_sum(machine, codeword, received)
    value = FiniteStateChannel(machine).log_likelihood(codeword, received)
    if expected == 0.0:
        assert value == -math.inf
    else:
        assert math.exp(value) == pytest.approx(expected)


@pytest.mark.parametrize("factory", CHANNELS)
@settings(max_examples=20, deadline=None)
@given(received=st.lists(st.sampled_from((0, 1)), min_size=1, max_size=5))
def test_symbol_likelihoods_match_marginal(factory, received):
    """BCJR symbol likelihoods equal ``p(y^n | x_i)`` under uniform inputs, up to row constants."""
    machine = factory()
    n = len(received)
    S = FiniteStateChannel(machine).symbol_log_likelihoods(received)
    for i in range(n):
        marginal = [0.0, 0.0]
        for codeword in itertools.product((0, 1), repeat=n):
            marginal[codeword[i]] += path_sum(machine, codeword, received)
        with np.errstate(divide="ignore"):
            expected = np.log(marginal)
        finite = np.isfinite(expected)
        assert np.array_equal(np.isfinite(S[i]), finite)
        if finite.all():
            assert S[i, 0] - S[i, 1] == pytest.approx(expected[0] - expected[1])


@settings(max_examples=25, deadline=None)
@given(
    p=st.floats(0.01, 0.45),
    received=st.lists(st.sampled_from((0, 1)), min_size=7, max_size=7),
    codeword=st.lists(st.sampled_from((0, 1)), min_size=7, max_size=7),
)
def test_one_state_channel_matches_bsc(p, received, codeword):
    """A one-state binary channel is the binary symmetric channel."""
    stateful = as_channel(binary_channel(p, p))
    bsc = as_channel(binary_symmetric_channel(p))
    assert stateful.log_likelihood(codeword, received) == pytest.approx(bsc.log_likelihood(codeword, received))
    assert symbol_llrs(stateful, received) == pytest.approx(symbol_llrs(bsc, received))


@pytest.mark.parametrize("p", [0.05, 0.2])
def test_one_state_error_probability_matches_bsc(p):
    code = hamming(3)
    expected = code.probability_of_error(binary_symmetric_channel(p), method="exact")
    assert code.probability_of_error(binary_channel(p, p), method="exact") == pytest.approx(expected)


@pytest.mark.parametrize("factory", CHANNELS)
def test_sample_frequencies_match_likelihood(factory):
    channel = FiniteStateChannel(factory())
    codeword = (0, 1, 1)
    prng = np.random.default_rng(0)
    draws = [channel.sample(codeword, prng) for _ in range(4000)]
    for received in itertools.product((0, 1), repeat=3):
        frequency = sum(d == received for d in draws) / len(draws)
        expected = math.exp(channel.log_likelihood(codeword, received))
        assert frequency == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize("factory", CHANNELS)
def test_exact_error_probability_over_memory(factory):
    """The exact block-error probability is the direct sum over messages and received words."""
    machine = factory()
    code = repetition(3)
    expected = 0.0
    for message in [(0,), (1,)]:
        for received in itertools.product((0, 1), repeat=3):
            if tuple(code.decode(received, channel=machine)) != message:
                expected += path_sum(machine, code.encode(message), received) / 2
    assert code.probability_of_error(machine, method="exact") == pytest.approx(expected)


def test_capacity_gap_is_a_bracket():
    p = 0.1
    code = hamming(3)
    lower, upper = code.capacity_gap(binary_channel(p, p))
    bsc_gap = code.capacity_gap(binary_symmetric_channel(p))
    assert lower <= bsc_gap + 1e-6
    assert bsc_gap <= upper + 1e-6


def test_symbol_maps():
    channel = FiniteStateChannel(trapdoor(), input_map={"0": "a", "1": "b"})
    assert channel.inputs == ("a", "b")
    assert channel.outputs == (0, 1)
    # Ball 0 left (prob 1/2): emit 0 surely, then emit the old 0 with prob 1/2.
    # Ball 1 left: after inserting 0 and emitting 0, two 1s remain, so 0 is impossible.
    assert channel.log_likelihood(("a", "b"), (0, 0)) == pytest.approx(math.log(0.25))
