"""Structural predicates in :mod:`sofic.properties`."""

from __future__ import annotations

from sofic import properties as P
from sofic.automata.nfa import NFA
from sofic.automata.transducers import MealyMachine
from sofic.examples import even_process, golden_mean
from sofic.examples.processes import sns
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, EPSILON


def _chain(edges, initial=None) -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset("ab"), initial_distribution=initial or {})
    for source, target, symbol, prob in edges:
        hmm.graph.add_state(source)
        hmm.graph.add_state(target)
        hmm.add_transition(source, target, symbol, prob)
    return hmm


def _cycle() -> MealyHMM:
    return _chain([(0, 1, "a", 1.0), (1, 2, "a", 1.0), (2, 0, "a", 1.0)], {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})


def _transducer(edges, initial=(0,)) -> MealyMachine:
    machine = MealyMachine(initial_states=frozenset(initial))
    for source, target, symbol, output in edges:
        machine.graph.add_state(source)
        machine.graph.add_state(target)
        machine.add_transition(source, target, symbol, output)
    return machine


def test_unifilarity_ignores_unlabeled_and_excluded_edges():
    hmm = _chain([(0, 0, "a", 0.5), (0, 1, "b", 0.5), (1, 0, "a", 1.0)])
    hmm.graph.add_transition(0, 1, prob=0.0)
    assert P.is_unifilar_emissions(hmm)
    hmm.add_transition(0, 1, "a", 0.0)
    assert not P.is_unifilar_emissions(hmm)
    assert P.is_unifilar_labeled(hmm, label_attr=ATTR_EMISSION, exclude_labels=("a",))


def test_counifilar_emissions():
    hmm = _chain([(0, 0, "a", 0.5), (0, 1, "b", 0.5), (1, 0, "b", 1.0)])
    hmm.graph.add_transition(1, 1, prob=0.0)
    assert P.is_counifilar_emissions(hmm)
    hmm.add_transition(1, 0, "a", 0.0)
    assert not P.is_counifilar_emissions(hmm)


def test_deterministic_automata_need_one_initial_state_and_no_epsilon():
    nfa = NFA(input_alphabet=frozenset("a"), initial_states=frozenset({0}), accepting_states=frozenset({0}))
    nfa.graph.add_state(0)
    nfa.add_transition(0, 0, "a")
    assert P.is_deterministic_automaton(nfa)
    nfa.graph.add_state(1)
    nfa.add_transition(0, 1, EPSILON)
    assert not P.is_deterministic_automaton(nfa)
    nfa.initial_states = frozenset({0, 1})
    assert not P.is_deterministic_automaton(nfa)


def test_deterministic_markov_chains():
    assert P.is_deterministic_markov(_cycle())
    assert not P.is_deterministic_markov(_chain([(0, 0, "a", 0.5), (0, 0, "b", 0.5)]))
    assert not P.is_deterministic_markov(_chain([(0, 0, "a", 0.5)]))


def test_transducer_unifilarity_and_counifilarity_skip_epsilon():
    machine = _transducer([(0, 1, "a", "x"), (0, 0, "a", "y"), (1, 0, EPSILON, "x"), (1, 1, EPSILON, "x")])
    assert P.is_unifilar_transducer(machine)
    assert P.is_counifilar_transducer(machine)
    assert not P.is_deterministic_transducer(machine)
    machine.add_transition(0, 0, "a", "x")
    assert not P.is_unifilar_transducer(machine)
    other = _transducer([(0, 1, "a", "x"), (1, 1, "a", "x")])
    assert not P.is_counifilar_transducer(other)


def test_sequential_and_subsequential_transducers():
    machine = _transducer([(0, 1, "a", "x"), (1, 0, "b", "y")])
    assert P.is_sequential_transducer(machine)
    assert not P.is_subsequential_transducer(machine)
    machine.final_output = {0: (), 1: ("z",)}
    assert P.is_subsequential_transducer(machine)
    assert not P.is_sequential_transducer(_transducer([(0, 1, "a", "x"), (0, 0, "a", "y")]))
    assert not P.is_sequential_transducer(_transducer([(0, 1, EPSILON, "x")]))
    assert not P.is_sequential_transducer(_transducer([(0, 1, "a", "x")], initial=(0, 1)))


def test_irreducibility_and_recurrent_states():
    assert P.is_irreducible(_chain([(0, 0, "a", 1.0)]))
    assert P.is_irreducible(_cycle())
    transient = _chain([(0, 1, "a", 1.0), (1, 1, "a", 1.0)])
    assert not P.is_irreducible(transient)
    assert P.recurrent_states(transient) == frozenset({1})


def test_ergodicity():
    assert not P.is_ergodic(MealyHMM(observation_alphabet=frozenset("a")))
    assert P.is_ergodic(_cycle())
    assert not P.is_ergodic(_cycle(), weak=False)
    assert P.is_ergodic(golden_mean(), weak=False)
    split = _chain([(0, 1, "a", 0.5), (0, 2, "b", 0.5), (1, 1, "a", 1.0), (2, 2, "b", 1.0)], {0: 1.0})
    assert not P.is_ergodic(split)
    assert P.is_ergodic(_chain([(0, 1, "a", 0.5), (0, 2, "b", 0.5), (1, 1, "a", 1.0), (2, 2, "b", 1.0)], {1: 1.0}))


def test_stationarity():
    assert P.is_stationary(golden_mean())
    assert not P.is_stationary(_chain([(0, 1, "a", 1.0), (1, 0, "a", 1.0)], {0: 1.0}))
    assert not P.is_stationary(_transducer([(0, 0, "a", "x")]))


def test_detailed_balance():
    reversible = _chain([(0, 0, "a", 0.5), (0, 1, "a", 0.5), (1, 0, "a", 0.5), (1, 1, "a", 0.5)], {0: 0.5, 1: 0.5})
    assert P.is_detailed_balance(reversible)
    assert not P.is_detailed_balance(_cycle())
    # Labeled flows: the golden mean's 1-edge A -> B has no reverse 1-edge.
    assert not P.is_detailed_balance(golden_mean())


def test_periodicity():
    assert not P.is_periodic(MealyHMM(observation_alphabet=frozenset("a")))
    assert P.is_periodic(_cycle())
    assert not P.is_periodic(golden_mean())


def test_strictly_sofic_supports():
    assert P.is_strictly_sofic(even_process())
    assert not P.is_strictly_sofic(golden_mean())
    # The simple nonunifilar source has a non-unifilar support presentation of finite type.
    assert not P.is_strictly_sofic(sns())
