"""Büchi automata for infinite-word recognition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pensive.automata.nfa import NFA


class BuchiAutomaton(NFA):
    """ω-automaton with Büchi acceptance on ultimately periodic inputs."""

    def accepts_lasso(self, prefix: Sequence[Any], loop: Sequence[Any]) -> bool:
        """Accept if infinitely repeating ``loop`` after ``prefix`` visits accepting states infinitely often."""
        from pensive.automata.buchi_simulation import accepts_lasso_buchi

        return accepts_lasso_buchi(self, prefix, loop)

    def accepts_omega(self, word: Sequence[Any]) -> bool:
        from pensive.automata.buchi_simulation import accepts_omega_buchi

        return accepts_omega_buchi(self, word)
