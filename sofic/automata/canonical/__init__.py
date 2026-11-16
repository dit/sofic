"""Canonical residual automata, atomata, their duality, and the other canonical NFAs of closures of residuals."""

from sofic.automata.canonical.atomaton import (
    Atomaton,
    AtomicAutomaton,
    MaximizedPrimeAtomaton,
    atomic_states,
    is_atomic,
)
from sofic.automata.canonical.closures import (
    Distromaton,
    UniversalAutomaton,
    distromaton_from_language,
    is_intersection_closed,
    minimal_structured_dfa,
    minimal_xor_automaton,
    universal_automaton_from_language,
    xor_caba_automaton,
)
from sofic.automata.canonical.dual import dual_atomaton_from_rfsa, dual_rfsa_from_atomaton
from sofic.automata.canonical.residual import (
    ResidualTable,
    atomaton_from_language,
    canonical_rfsa_from_language,
    maximized_prime_atomaton_from_language,
    observation_to_atomaton,
    observation_to_canonical_rfsa,
    observation_to_maximized_prime_atomaton,
    observation_to_minimal_dfa,
)
from sofic.automata.canonical.rfsa import CanonicalRFSA, ResidualFiniteStateAutomaton

__all__ = [
    "Atomaton",
    "AtomicAutomaton",
    "CanonicalRFSA",
    "Distromaton",
    "MaximizedPrimeAtomaton",
    "ResidualFiniteStateAutomaton",
    "ResidualTable",
    "UniversalAutomaton",
    "atomaton_from_language",
    "atomic_states",
    "canonical_rfsa_from_language",
    "distromaton_from_language",
    "dual_atomaton_from_rfsa",
    "dual_rfsa_from_atomaton",
    "is_atomic",
    "is_intersection_closed",
    "maximized_prime_atomaton_from_language",
    "minimal_structured_dfa",
    "minimal_xor_automaton",
    "observation_to_atomaton",
    "observation_to_canonical_rfsa",
    "observation_to_maximized_prime_atomaton",
    "observation_to_minimal_dfa",
    "universal_automaton_from_language",
    "xor_caba_automaton",
]
