"""Tests for Büchi automata."""

import pytest
from hypothesis import given

from sofic.automata.buchi import BuchiAutomaton
from sofic.graph import ATTR_SYMBOL
from sofic.testing.strategies import buchi_automata


def _accepting_loop_ba() -> BuchiAutomaton:
    ba = BuchiAutomaton(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    ba.graph.add_state("q0")
    ba.graph.add_state("q1")
    ba.graph.add_transition("q0", "q1", **{ATTR_SYMBOL: "a"})
    ba.graph.add_transition("q1", "q1", **{ATTR_SYMBOL: "a"})
    ba.graph.add_transition("q1", "q0", **{ATTR_SYMBOL: "b"})
    return ba


def _rejecting_loop_ba() -> BuchiAutomaton:
    ba = BuchiAutomaton(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    ba.graph.add_state("q0")
    ba.graph.add_state("q1")
    ba.graph.add_transition("q0", "q1", **{ATTR_SYMBOL: "a"})
    return ba


def test_accepts_lasso_visits_accepting_infinitely():
    ba = _accepting_loop_ba()
    assert ba.accepts_lasso((), ("a",))
    assert ba.accepts_lasso(("a",), ("a",))


def test_rejects_lasso_finite_accept_visits():
    ba = _rejecting_loop_ba()
    assert not ba.accepts_lasso((), ("a",))


def test_accepts_omega_periodic_pair():
    ba = _accepting_loop_ba()
    assert ba.accepts_omega(((), ("a",)))
    assert ba.accepts_omega((("a",), ("a",)))


def test_accepts_omega_non_periodic_raises():
    ba = _accepting_loop_ba()
    with pytest.raises(NotImplementedError):
        ba.accepts_omega(("a", "b", "a"))


def test_empty_loop_is_not_an_omega_word():
    with pytest.raises(ValueError, match="non-empty loop"):
        _accepting_loop_ba().accepts_lasso(("a",), ())


def _mid_loop_ba() -> BuchiAutomaton:
    ba = BuchiAutomaton(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({0}),
        accepting_states=frozenset({1}),
    )
    ba.graph.add_state(0)
    ba.graph.add_state(1)
    ba.add_transition(0, 1, "a")
    ba.add_transition(1, 0, "b")
    return ba


def test_accepting_state_visited_mid_loop():
    ba = _mid_loop_ba()
    assert ba.accepts_lasso((), ("a", "b"))
    assert ba.accepts_lasso(("a",), ("b", "a"))
    assert not ba.accepts_lasso((), ("a",))


def test_pure_epsilon_cycle_through_accepting_state_rejects():
    from sofic.graph import EPSILON

    ba = BuchiAutomaton(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({0}),
        accepting_states=frozenset({1}),
    )
    ba.graph.add_state(0)
    ba.graph.add_state(1)
    ba.graph.add_state(2)
    ba.add_transition(0, 1, EPSILON)
    ba.add_transition(1, 0, EPSILON)
    ba.add_transition(0, 2, "a")
    ba.add_transition(2, 2, "a")
    assert not ba.accepts_lasso((), ("a",))
    ba.add_transition(2, 0, "a")
    assert ba.accepts_lasso((), ("a",))


def _brute_lasso(ba: BuchiAutomaton, prefix, loop) -> bool:
    """Accept iff some infinite path on ``prefix loop^omega`` revisits an accepting (state, phase)."""
    start = {(q, 0) for q in ba._run_nfa(prefix)}

    def succ(node):
        q, i = node
        return {(t, (i + 1) % len(loop)) for t in ba.delta(q, loop[i])}

    seen, stack = set(start), list(start)
    while stack:
        for y in succ(stack.pop()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    for x in seen:
        if x[0] not in ba.accepting_states:
            continue
        visited, stack = set(), list(succ(x))
        while stack:
            y = stack.pop()
            if y == x:
                return True
            if y not in visited:
                visited.add(y)
                stack.extend(succ(y))
    return False


def test_lasso_acceptance_matches_brute_force():
    import itertools
    import random

    rng = random.Random(0)
    for _ in range(150):
        n = rng.randint(1, 3)
        ba = BuchiAutomaton(
            input_alphabet=frozenset("ab"),
            initial_states=frozenset({0}),
            accepting_states=frozenset(s for s in range(n) if rng.random() < 0.4),
        )
        for s in range(n):
            ba.graph.add_state(s)
        for s, t, c in itertools.product(range(n), range(n), "ab"):
            if rng.random() < 0.35:
                ba.add_transition(s, t, c)
        for pl, ll in itertools.product(range(3), range(1, 4)):
            for p in itertools.product("ab", repeat=pl):
                for loop in itertools.product("ab", repeat=ll):
                    assert ba.accepts_lasso(p, loop) == _brute_lasso(ba, p, loop)


def test_buchi_emptiness_uses_omega_semantics():
    accepting = _accepting_loop_ba()
    assert not accepting.is_empty()
    prefix, loop = accepting.accepted_lasso()
    assert accepting.accepts_lasso(prefix, loop)

    rejecting = _rejecting_loop_ba()
    assert rejecting.accepted_lasso() is None
    assert rejecting.is_empty()


def test_buchi_finite_word_decisions_are_refused():
    ba = _accepting_loop_ba()
    for call in (ba.accepted_word, ba.is_universal, lambda: ba.includes(ba)):
        with pytest.raises(NotImplementedError):
            call()


def _short_lassos(alphabet, max_prefix=2, max_loop=3):
    from itertools import product

    for p in range(max_prefix + 1):
        for prefix in product(sorted(alphabet), repeat=p):
            for q in range(1, max_loop + 1):
                for loop in product(sorted(alphabet), repeat=q):
                    yield prefix, loop


@given(buchi_automata(allow_epsilon=True))
def test_accepted_lasso_matches_lasso_oracle(ba):
    witness = ba.accepted_lasso()
    if witness is not None:
        assert ba.accepts_lasso(*witness)
    else:
        assert not any(ba.accepts_lasso(p, l) for p, l in _short_lassos(ba.input_alphabet))
