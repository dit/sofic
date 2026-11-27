"""Tests for the simplex-chain search for exact one-sided generators."""

from __future__ import annotations

import functools
import itertools
import math
import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators import simplex_chains as sc
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import _GeneratorProblem
from sofic.generators.generative_search import generative_model_from_bidirectional
from sofic.generators.hmm_search import coupled_seed_from_hmm

# Binary, cryptic order 2, three forward and three reverse causal states; bracket [0.4638, 0.4777301239] before the
# simplex-chain search, which finds 0.4777300557 on both sides.
_BEST_KNOWN = 0.4777301239109031


@functools.cache
def _problem() -> _GeneratorProblem:
    a, b = 0.9253177336700834, 0.9575640890108456
    mats = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, a], [0.0, 0.0, b]]),
        1: np.array([[0.0, 1.0, 0.0], [0.0, 1 - a, 0.0], [1 - b, 0.0, 0.0]]),
    }
    return _GeneratorProblem(BidirectionalEpsilonMachine.from_forward(from_symbol_matrices((0, 1, 2), (0, 1), mats)))


@functools.cache
def _sides() -> dict[str, sc._Side]:
    return sc._sides(_problem(), 2)


def _own_law(problem: _GeneratorProblem, name: str) -> np.ndarray:
    lifted = problem.lifted
    return (lifted.minus if name == "reverse" else lifted.plus).T @ lifted.p


def _full_blocks(side: sc._Side) -> dict[tuple[int, int], tuple[int, ...]]:
    return {(s, x): tuple(range(side.n)) for s in range(side.n) for x in range(side.n_sym) if side.P[x, s].sum() > 0}


def test_the_vertex_chain_is_the_epsilon_machine():
    problem = _problem()
    for name, side in _sides().items():
        atoms, assign = np.eye(side.n), {}
        for s, x in itertools.product(range(side.n), range(side.n_sym)):
            y = atoms[s] @ side.P[x]
            if y.sum() > 1e-14:
                assign[(s, x)] = tuple(int(c) for c in np.flatnonzero(y > 0))
        ex = sc._exact_generator(side, problem, atoms, assign)
        assert ex is not None
        assert ex["word_error"] < 1e-12
        assert ex["entropy"] == pytest.approx(sc._entropy(_own_law(problem, name)), abs=1e-12)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), extra=st.integers(1, 5))
def test_chain_lemma_on_random_pools(seed, extra):
    """Any vertex of the lifted polytope of an invariant pool is an exact one-sided generator."""
    problem, rng = _problem(), np.random.default_rng(seed)
    for side in _sides().values():
        pool = np.vstack([np.eye(side.n), rng.dirichlet(np.ones(side.n), extra)])
        out = sc._lifted_lp(side, pool, rng.normal(size=len(pool)))
        assert out is not None  # the vertices make every pool invariant
        pi, _T, assign = out
        supp = np.flatnonzero(pi > 1e-12)
        index = {int(s): i for i, s in enumerate(supp)}
        sub = {
            (index[s], x): tuple(index[v] for v in idx if v in index) for (s, x), idx in assign.items() if s in index
        }
        sub = {b: i for b, i in sub.items() if i}
        snapped, res = sc._snap(side, pool[supp], sub)
        assert res < 1e-10
        full, outside = sc._complete_assign(side, snapped, sub)
        assert outside == 0
        ex = sc._exact_generator(side, problem, snapped, full)
        assert ex is not None
        assert ex["negative"] >= -1e-9
        assert ex["word_error"] < 1e-8


def test_a_failed_least_squares_step_is_a_failed_snap(monkeypatch):
    """LAPACK's SVD can fail to converge; the snap then reports an infinite residual instead of raising."""
    side = _sides()["forward"]
    atoms = np.eye(side.n)
    # one successor per block, so the wall residuals need no least squares and only the Gauss-Newton step does
    assign = {
        (s, x): (int(np.argmax(atoms[s] @ side.P[x])),)
        for s, x in itertools.product(range(side.n), range(side.n_sym))
        if (atoms[s] @ side.P[x]).sum() > 1e-14
    }

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")

    monkeypatch.setattr(np.linalg, "lstsq", fail)
    _, res = sc._snap(side, 0.9 * atoms + 0.1 / side.n, assign)
    assert res == math.inf


def test_every_closed_class_of_the_full_vertex_chain_generates():
    """The chain lemma with every block on all the vertices: ``T^x[s, .] = m_x(s) f_x(s)``."""
    problem = _problem()
    for side in _sides().values():
        atoms = np.eye(side.n)
        T, res, neg = sc._transport(side, atoms, _full_blocks(side))
        assert res < 1e-14
        assert neg >= -1e-14
        for pi in sc._stationaries(T):
            g_pi, K, _ = sc._to_generator(side, T, pi)
            assert sc._word_error(problem, g_pi, K) < 1e-12


def test_the_chamber_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    for side in _sides().values():
        atoms = np.eye(side.n)
        assign = {(s, x): tuple(range(side.n)) for s in range(side.n) for x in range(side.n_sym)}

        def H(A, side=side, assign=assign):  # plain linear solves: the chamber's smooth branch, signs unchecked
            T = np.zeros((side.n_sym, len(A), len(A)))
            for (s, x), idx in assign.items():
                T[x, s, list(idx)] = np.linalg.solve(A[list(idx)].T, A[s] @ side.P[x])
            k = len(A)
            a = np.vstack([T.sum(axis=0).T - np.eye(k), np.ones(k)])
            b = np.zeros(k + 1)
            b[-1] = 1.0
            return sc._entropy(np.linalg.lstsq(a, b, rcond=None)[0])

        T, _, _ = sc._transport(side, atoms, assign)
        grad = sc._chamber_gradient(side, atoms, assign, sc._stationaries(T)[0])
        for _ in range(5):
            d = rng.normal(size=atoms.shape)
            d -= d.mean(axis=1, keepdims=True)
            h = 1e-6
            fd = (H(atoms + h * d) - H(atoms - h * d)) / (2 * h)
            assert fd == pytest.approx(float((grad * d).sum()), rel=1e-4, abs=1e-8)


def test_the_synchronizing_complex_holds_the_companion_beliefs_and_is_invariant():
    for side in _sides().values():
        faces = sc._sync_complex(side, 2)
        for belief in side.rho:  # the companion ε-machine is a one-sided generator with these atoms
            assert sc._in_complex(np.flatnonzero(belief > 1e-12), faces)
        for f in faces:
            r = np.zeros(side.n)
            r[list(f)] = 1.0 / len(f)
            for x in range(side.n_sym):
                y = r @ side.P[x]
                if y.sum() > 1e-14:
                    assert sc._in_complex(np.flatnonzero(y > 1e-14), faces)


def test_the_search_finds_the_best_known_generators():
    problem = _problem()
    results = sc.simplex_chain_search(problem, seconds=60.0, restarts=2, rng=np.random.default_rng(0))
    assert {r.side for r in results} == {"reverse", "forward"}
    assert results[0].entropy <= _BEST_KNOWN
    for r in results:
        assert r.word_error < 1e-10
        assert r.entropy == pytest.approx(sc._entropy(r.pi), abs=1e-12)
        assert problem.residual(*coupled_seed_from_hmm(problem, r.matrices)) < 1e-9


def test_the_search_stops_at_the_target():
    results = sc.simplex_chain_search(_problem(), seconds=60.0, target=math.inf, rng=np.random.default_rng(0))
    assert [r.side for r in results] == ["reverse"]
    assert results[0].log[-1][0] == "restart vertices"  # no restarts after the ε-machine's descent


@functools.cache
def _infinite_reverse_problem() -> _GeneratorProblem:
    """Two forward causal states and an infinite reverse ε-machine (53 states at the merge tolerance)."""
    mats = {
        0: np.array([[0.1317773554901893, 0.0], [0.3703792631454206, 0.0]]),
        1: np.array([[0.1449685489375078, 0.0], [0.0, 0.39404348038124015]]),
        2: np.array([[0.0, 0.7232540955723029], [0.0, 0.23557725647333932]]),
    }
    return _GeneratorProblem(BidirectionalEpsilonMachine.from_forward(from_symbol_matrices((0, 1), (0, 1, 2), mats)))


@pytest.mark.filterwarnings("ignore:the reverse ε-machine of this process is infinite:RuntimeWarning")
def test_the_search_starts_from_the_epsilon_machine():
    """The generative complexity of this process is the forward ε-machine's entropy."""
    problem = _infinite_reverse_problem()
    (result,) = sc.simplex_chain_search(problem, sides=("forward",), seconds=0.0, restarts=0)
    assert result.entropy == pytest.approx(sc._entropy(_own_law(problem, "forward")), abs=1e-9)
    assert result.word_error < 1e-10


def test_the_wall_tangent_stops_at_its_deadline():
    side = _sides()["forward"]
    atoms, assign = np.eye(side.n), _full_blocks(side)
    assert sc._wall_tangent(side, atoms, assign, deadline=time.monotonic() - 1.0) is None
    assert sc._wall_tangent(side, atoms, assign) is not None


@pytest.mark.filterwarnings("ignore:the reverse ε-machine of this process is infinite:RuntimeWarning")
def test_the_search_keeps_its_time_on_a_large_side():
    """On the 53 reverse states one wall tangent took about a minute, past a 2 s budget (95 s in all)."""
    problem = _infinite_reverse_problem()
    start = time.monotonic()
    sc.simplex_chain_search(problem, sides=("reverse",), seconds=2.0, rng=np.random.default_rng(0))
    assert time.monotonic() - start < 20.0


def test_the_pipeline_offers_the_simplex_chain_generators():
    bidir = _problem().bidir
    kwargs = {"stages": ("machines", "chains"), "coverage": "off", "generator_heaviest_term": False}
    result = generative_model_from_bidirectional(bidir, rng=np.random.default_rng(0), **kwargs)
    assert result.upper <= _BEST_KNOWN
    assert result.upper_source.startswith("simplex chain")
    assert any(s.get("stage") == "simplex-chain search" for s in result.stages)
    off = generative_model_from_bidirectional(bidir, simplex_chains=False, rng=np.random.default_rng(0), **kwargs)
    assert not any(s.get("stage") == "simplex-chain search" for s in off.stages)
    assert off.upper > _BEST_KNOWN
