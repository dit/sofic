"""Canonical residual automata, atomata, and their duality."""

from sofic.automata.canonical.atomaton import (
    Atomaton,
    AtomicAutomaton,
    MaximizedPrimeAtomaton,
    atomic_states,
    is_atomic,
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
    "MaximizedPrimeAtomaton",
    "ResidualFiniteStateAutomaton",
    "ResidualTable",
    "atomaton_from_language",
    "atomic_states",
    "canonical_rfsa_from_language",
    "dual_atomaton_from_rfsa",
    "dual_rfsa_from_atomaton",
    "is_atomic",
    "maximized_prime_atomaton_from_language",
    "observation_to_atomaton",
    "observation_to_canonical_rfsa",
    "observation_to_maximized_prime_atomaton",
    "observation_to_minimal_dfa",
]
