"""Symbolic HMM probabilities and Chaos Forgets appendix reproduction."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("sympy")
pytest.importorskip("dit")

import sympy as sp

from sofic.examples.epsilon_machines import (
    tent_map_misiurewicz_a,
    tent_map_misiurewicz_bidirectional,
    tent_map_misiurewicz_forward,
    tent_map_misiurewicz_hmm,
    tent_map_misiurewicz_information_expected,
)
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.prob import (
    SymbolConstraints,
    canonical_prob_key,
    is_symbolic,
    probs_equal,
)
from sofic.generators.words import hmm_word_probability


def test_symbolic_edge_probabilities_preserved():
    a = sp.symbols("a", positive=True)
    hmm = tent_map_misiurewicz_forward(a)
    probs = [t.data["prob"] for t in hmm.transitions()]
    assert any(is_symbolic(p) for p in probs)
    assert any(p.has(a) for p in probs if is_symbolic(p))


def test_symbolic_stationary_sums_to_one():
    a = sp.symbols("a", positive=True)
    fwd = tent_map_misiurewicz_forward(a)
    pi = fwd.stationary_distribution()
    assert sp.simplify(sum(pi) - 1) == 0


def test_fig6_hmm_from_hmm_recovers_fig7_entropy():
    hmm = tent_map_misiurewicz_hmm()
    assert set(hmm.states()) == {"A", "B", "C", "D"}
    assert not hmm.is_unifilar()
    assert "Bprime" not in hmm.states()
    eps = EpsilonMachine.from_hmm(hmm)
    fwd = tent_map_misiurewicz_forward()
    assert eps.entropy_rate() == pytest.approx(fwd.entropy_rate(), abs=1e-9)
    assert len(list(eps.states())) == len(list(fwd.states()))


def test_fig6_hmm_matches_fig7_word_probabilities():
    hmm = tent_map_misiurewicz_hmm()
    fwd = tent_map_misiurewicz_forward()
    for word in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 1, 0), (1, 1, 0)]:
        assert hmm_word_probability(hmm, word) == pytest.approx(
            hmm_word_probability(fwd, word),
            abs=1e-10,
        )


def test_symbol_constraints_residue_field_equality():
    """Probabilities equal only modulo the Misiurewicz minimal polynomial merge."""
    a = sp.symbols("a", positive=True)
    constraints = SymbolConstraints([a**3 - 2 * a - 2])
    left = a / (a + 1) ** 2
    right = 2 / (a**2 + 2 * a + 2)
    # Equal in Q[a]/(a**3 - 2a - 2), distinct as free-symbol rationals.
    assert probs_equal(left, right, constraints=constraints)
    assert not probs_equal(left, right)
    assert canonical_prob_key(left, constraints) == canonical_prob_key(right, constraints)
    assert canonical_prob_key(left) != canonical_prob_key(right)
    # Genuinely different values stay distinct under the constraint.
    assert not probs_equal(a / (a + 1), sp.Rational(1, 2), constraints=constraints)


def test_symbolic_fig6_from_hmm_merges_to_four_states():
    """Constrained symbolic Fig.~6 recovers the 4-state Fig.~7 machine."""
    a = sp.symbols("a", positive=True)
    hmm = tent_map_misiurewicz_hmm(a)
    assert hmm.symbol_constraints is not None
    eps = EpsilonMachine.from_hmm(hmm)
    assert len(list(eps.states())) == 4

    a_num = tent_map_misiurewicz_a()
    rate = eps.entropy_rate()
    rate_num = float(rate.subs(a, a_num)) if is_symbolic(rate) else float(rate)
    assert rate_num == pytest.approx(tent_map_misiurewicz_forward().entropy_rate(), abs=1e-9)


def test_symbolic_fig6_free_symbol_stays_five_states():
    """Without the algebraic constraint, a free ``a`` yields the distinct 5-state MSP."""
    a = sp.symbols("a", positive=True)
    hmm = tent_map_misiurewicz_hmm(a)
    hmm.symbol_constraints = None  # treat a as an unconstrained free parameter
    eps = EpsilonMachine.from_hmm(hmm)
    assert len(list(eps.states())) == 5


def test_symbolic_fig6_graphviz_labels():
    """Symbolic Fig.~6 edge probs render without float() coercion."""
    pytest.importorskip("graphviz")
    a = sp.symbols("a", positive=True)
    hmm = tent_map_misiurewicz_hmm(a)
    source = hmm.to_graphviz().source
    assert "1/(a + 1)" in source or "1/(a+1)" in source
    assert "a/(2*(a + 1))" in source or "a/(2*(a+1))" in source
    bundle = hmm._repr_mimebundle_()
    assert bundle is not None
    assert "image/svg+xml" in bundle


def test_symbolic_fig8_ephemeral_matches_closed_form_at_misiurewicz():
    """Supplement closed form holds at the Misiurewicz root of ``a``."""
    a = sp.symbols("a", positive=True)
    bidir = tent_map_misiurewicz_bidirectional(a)
    r = bidir.ephemeral_information()
    b = bidir.bound_information()
    h = bidir.entropy_rate()
    assert is_symbolic(r)

    a_num = tent_map_misiurewicz_a()
    expected = tent_map_misiurewicz_information_expected(a_num)["ephemeral_mu"]
    assert float(r.subs(a, a_num)) == pytest.approx(expected, abs=1e-12)

    # Anatomy identity h = b + r (numerically at the Misiurewicz root; sympy may
    # not cancel the unsimplified Shannon expressions identically).
    assert float((b + r - h).subs(a, a_num)) == pytest.approx(0.0, abs=1e-12)


def test_symbolic_fig8_matches_numeric_anatomy():
    a = sp.symbols("a", positive=True)
    a_num = tent_map_misiurewicz_a()
    sym = tent_map_misiurewicz_bidirectional(a)
    num = tent_map_misiurewicz_bidirectional(a_num)
    expected = tent_map_misiurewicz_information_expected(a_num)

    assert float(sym.ephemeral_information().subs(a, a_num)) == pytest.approx(
        num.ephemeral_information(), abs=1e-10
    )
    assert float(sym.bound_information().subs(a, a_num)) == pytest.approx(
        num.bound_information(), abs=1e-10
    )
    assert float(sym.entropy_rate().subs(a, a_num)) == pytest.approx(
        expected["entropy_rate"], abs=1e-10
    )
    assert expected["entropy_rate"] == pytest.approx(math.log2(a_num), abs=1e-12)


def test_closed_form_rational_expression():
    """The supplement's rational-in-``a`` ephemeral formula."""
    a = sp.symbols("a", positive=True)
    expected = tent_map_misiurewicz_information_expected(a)["ephemeral_mu"]
    rational = sp.Rational(1, 4) * (3 - 2 / (a + 1) - 4 / (a + 2) + 9 / (2 * a + 3))
    assert sp.simplify(expected - rational) == 0
