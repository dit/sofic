"""Invariants of the generator search on random ε-machines, and known values of C_g."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.generative_search import generative_model, generative_model_from_bidirectional
from sofic.generators.hmm_search import hmm_generator_search

_TOL = 1e-7


def _topologies(n: int) -> list[tuple[tuple[int | None, ...], ...]]:
    """Binary topologies on ``n`` states: a target or ``None`` per (state, symbol), strongly connected."""
    out = []
    for targets in itertools.product([None, *range(n)], repeat=2 * n):
        rows = tuple(tuple(targets[2 * s + x] for x in range(2)) for s in range(n))
        if any(all(t is None for t in row) for row in rows):
            continue
        reach = [{s} for s in range(n)]
        for _ in range(n):
            reach = [r | {t for u in r for t in rows[u] if t is not None} for r in reach]
        if all(r == set(range(n)) for r in reach):
            out.append(rows)
    return out


@st.composite
def _epsilon_machines(draw):
    rows = draw(st.sampled_from(_topologies(2) + _topologies(3)))
    n = len(rows)
    mats = {x: np.zeros((n, n)) for x in (0, 1)}
    for s, row in enumerate(rows):
        live = [x for x in (0, 1) if row[x] is not None]
        p = draw(st.floats(min_value=0.1, max_value=0.9)) if len(live) == 2 else 1.0
        for x, prob in zip(live, (p, 1 - p) if len(live) == 2 else (1.0,), strict=True):
            mats[x][s, row[x]] += prob
    machine = from_symbol_matrices(tuple(range(n)), (0, 1), mats)
    return machine if isinstance(machine, EpsilonMachine) else EpsilonMachine.from_hmm(machine)


@pytest.mark.slow
@given(forward=_epsilon_machines())
@settings(max_examples=12, deadline=None, derandomize=True)
def test_generator_search_bracket_is_consistent(forward):
    try:
        bidir = BidirectionalEpsilonMachine.from_forward(forward)
    except Exception:  # noqa: BLE001
        assume(False)
    result = generative_model_from_bidirectional(bidir, time_budget=20, rng=np.random.default_rng(0))
    excess = float(bidir.excess_entropy())
    c_mu = min(
        float(bidir.forward_epsilon_machine().statistical_complexity()),
        float(bidir.reverse_epsilon_machine().statistical_complexity()),
    )
    assert excess - _TOL <= result.lower <= result.upper + _TOL
    assert result.upper <= c_mu + _TOL
    assert result.reproduction_error is not None and result.reproduction_error <= 1e-5
    assert float(result.model.state_entropy()) == pytest.approx(result.upper, abs=_TOL)
    assert result.structure["n_states"] == len(list(result.model.states()))
    assert sum(result.structure["weights"]) == pytest.approx(1.0)


def _ternary_golden_mean():
    """Golden mean at 1/2 with its 1 split into two symbols: A -0-> A (1/2), A -1-> B, A -2-> B (1/4 each), B -0-> A.

    The joint of S+ (last symbol 0 or not) and S- (next symbol 0 or not) is the binary golden mean's, and the
    persistent crypticities vanish, so C_g = G[S+ : S-] = H(1/3, 2/3)."""
    return from_symbol_matrices(
        ("A", "B"),
        (0, 1, 2),
        {
            0: np.array([[0.5, 0.0], [1.0, 0.0]]),
            1: np.array([[0.0, 0.25], [0.0, 0.0]]),
            2: np.array([[0.0, 0.25], [0.0, 0.0]]),
        },
    )


@pytest.mark.slow
def test_ternary_golden_mean_generative_complexity():
    target = float(-(1 / 3) * np.log2(1 / 3) - (2 / 3) * np.log2(2 / 3))
    machine = _ternary_golden_mean()
    result = generative_model(machine, time_budget=60, rng=np.random.default_rng(0))
    assert result.certified
    assert result.lower == pytest.approx(target, abs=1e-6)
    assert result.upper == pytest.approx(target, abs=1e-6)
    found = hmm_generator_search(machine, 2, starts=4, rng=np.random.default_rng(0))
    assert found and len(found[0].matrices) == 3
    assert found[0].entropy == pytest.approx(target, abs=1e-6)


@pytest.mark.slow
def test_run_length_limited_generative_complexity_is_three_halves():
    """(2, inf) RLL: C_g = 3/2 = C_mu > G (computer-assisted proof in the cpfci manuscript)."""
    machine = from_symbol_matrices(
        ("A", "B", "C"),
        (0, 1),
        {0: np.array([[0.5, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]), 1: np.array([[0, 0.5, 0], [0, 0, 0], [0, 0, 0]])},
    )
    bidir = BidirectionalEpsilonMachine.from_forward(machine)
    result = generative_model_from_bidirectional(bidir, rng=np.random.default_rng(0))
    assert result.upper == pytest.approx(1.5, abs=1e-6)
    assert result.lower == pytest.approx(1.5, abs=1e-6)
    assert result.certified
