"""
pensive is a Python package for hidden Markov models, symbolic dynamics,
finite state machines, and other stochastic symbol generators.
"""

try:
    from importlib.metadata import version

    __version__ = version("pensive")
except Exception:
    __version__ = "0.0.0"

from pensive.automata import (
    DFA,
    NFA,
    Atomaton,
    AtomicAutomaton,
    AutomatonLanguage,
    BuchiAutomaton,
    CanonicalRFSA,
    LabeledAutomaton,
    MaximizedPrimeAtomaton,
    MealyMachine,
    MinimizationAlgorithm,
    MooreMachine,
    ObservationTable,
    RegularLanguage,
    ResidualFiniteStateAutomaton,
    Transducer,
    UnifilarAutomaton,
    VisiblyPushdownAutomaton,
    complete,
    determinize,
    equivalent,
    minimize,
    trim,
)
from pensive.base import StateMachine
from pensive.generators import (
    BidirectionalEpsilonMachine,
    EpsilonMachine,
    HiddenMarkovModel,
    MarkovChain,
    MealyHMM,
    MixedState,
    MixedStatePresentation,
    MooreHMM,
    NMachine,
    ProbabilisticFiniteAutomaton,
    QuasiRealization,
    QuasiStochasticModel,
    StochasticModel,
)
from pensive.graph import EPSILON, TransitionGraph
from pensive.indexing import StateIndex
from pensive.operations import reverse
from pensive.shifts import (
    LeftFischerCover,
    LeftKriegerCover,
    RightFischerCover,
    RightKriegerCover,
    ShiftOfFiniteType,
    SoficShift,
    SymbolicModel,
    TopologicalMarkovChain,
)

__all__ = [
    "Atomaton",
    "AtomicAutomaton",
    "AutomatonLanguage",
    "BuchiAutomaton",
    "BidirectionalEpsilonMachine",
    "CanonicalRFSA",
    "DFA",
    "EPSILON",
    "EpsilonMachine",
    "HiddenMarkovModel",
    "LabeledAutomaton",
    "LeftFischerCover",
    "LeftKriegerCover",
    "MarkovChain",
    "MaximizedPrimeAtomaton",
    "MealyHMM",
    "MealyMachine",
    "MinimizationAlgorithm",
    "MixedState",
    "MixedStatePresentation",
    "MooreHMM",
    "MooreMachine",
    "NFA",
    "NMachine",
    "ObservationTable",
    "ProbabilisticFiniteAutomaton",
    "QuasiRealization",
    "QuasiStochasticModel",
    "RegularLanguage",
    "ResidualFiniteStateAutomaton",
    "RightFischerCover",
    "RightKriegerCover",
    "ShiftOfFiniteType",
    "SoficShift",
    "StateIndex",
    "StateMachine",
    "StochasticModel",
    "SymbolicModel",
    "TopologicalMarkovChain",
    "Transducer",
    "TransitionGraph",
    "UnifilarAutomaton",
    "VisiblyPushdownAutomaton",
    "complete",
    "determinize",
    "equivalent",
    "minimize",
    "reverse",
    "trim",
    "__version__",
]
