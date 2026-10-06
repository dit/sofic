"""Visibly pushdown automata, their constructions, and simulation."""

from sofic.automata.vpa.base import VisiblyPushdownAutomaton
from sofic.automata.vpa.canonical import CanonicalVisiblyPushdownAutomaton
from sofic.automata.vpa.deterministic import DeterministicVisiblyPushdownAutomaton
from sofic.automata.vpa.modular import (
    CallDrivenAutomaton,
    MultipleEntryVisiblyPushdownAutomaton,
    SingleEntryVisiblyPushdownAutomaton,
)
from sofic.automata.vpa.operations import BOTTOM, NormalVPA, to_multiple_entry, to_single_entry
from sofic.automata.vpa.simulation import recognizes_vpa

__all__ = [
    "BOTTOM",
    "CallDrivenAutomaton",
    "CanonicalVisiblyPushdownAutomaton",
    "DeterministicVisiblyPushdownAutomaton",
    "MultipleEntryVisiblyPushdownAutomaton",
    "NormalVPA",
    "SingleEntryVisiblyPushdownAutomaton",
    "VisiblyPushdownAutomaton",
    "recognizes_vpa",
    "to_multiple_entry",
    "to_single_entry",
]
