"""Finite automata and transducers."""

# ``atomaton`` is an intentional pun on atomic automaton.
from pensive.automata.algorithms import (
    MinimizationAlgorithm,
    complete,
    determinize,
    equivalent,
    minimize,
    trim,
)
from pensive.automata.atomaton import Atomaton, AtomicAutomaton, MaximizedPrimeAtomaton
from pensive.automata.base import LabeledAutomaton
from pensive.automata.buchi import BuchiAutomaton
from pensive.automata.dfa import DFA
from pensive.automata.icdfa import (
    ICDFAString,
    count_flag_sequences,
    count_icdfa,
    count_icdfa_empty,
    dfa_to_icdfa_string,
    first_icdfa_empty_string,
    flags_from_string,
    icdfa_string_to_dfa,
    iter_icdfa,
    iter_icdfa_empty_strings,
    last_icdfa_empty_string,
    next_flags,
    next_icdfa_empty_string,
    string_from_flags,
    validate_icdfa_empty_string,
)
from pensive.automata.idfa import (
    MISSING_TRANSITION,
    count_accessible_idfa,
    first_idfa_string,
    iter_idfa_strings,
    rank_idfa_string,
    reroot_idfa_string,
    unrank_idfa_string,
    validate_idfa_string,
)
from pensive.automata.languages import AutomatonLanguage, RegularLanguage
from pensive.automata.nfa import NFA
from pensive.automata.observation import ObservationTable
from pensive.automata.rfsa import CanonicalRFSA, ResidualFiniteStateAutomaton
from pensive.automata.transducers import MealyMachine, MooreMachine, Transducer
from pensive.automata.unifilar import UnifilarAutomaton
from pensive.automata.vpa import VisiblyPushdownAutomaton

__all__ = [
    "Atomaton",
    "AtomicAutomaton",
    "AutomatonLanguage",
    "BuchiAutomaton",
    "CanonicalRFSA",
    "DFA",
    "ICDFAString",
    "LabeledAutomaton",
    "MaximizedPrimeAtomaton",
    "MealyMachine",
    "MinimizationAlgorithm",
    "MooreMachine",
    "NFA",
    "ObservationTable",
    "RegularLanguage",
    "ResidualFiniteStateAutomaton",
    "Transducer",
    "UnifilarAutomaton",
    "VisiblyPushdownAutomaton",
    "complete",
    "count_accessible_idfa",
    "count_flag_sequences",
    "count_icdfa",
    "count_icdfa_empty",
    "determinize",
    "dfa_to_icdfa_string",
    "equivalent",
    "first_icdfa_empty_string",
    "first_idfa_string",
    "flags_from_string",
    "icdfa_string_to_dfa",
    "iter_icdfa",
    "iter_icdfa_empty_strings",
    "iter_idfa_strings",
    "last_icdfa_empty_string",
    "MISSING_TRANSITION",
    "minimize",
    "next_flags",
    "next_icdfa_empty_string",
    "rank_idfa_string",
    "reroot_idfa_string",
    "string_from_flags",
    "trim",
    "unrank_idfa_string",
    "validate_idfa_string",
    "validate_icdfa_empty_string",
]
