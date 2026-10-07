"""Büchi automata for infinite-word recognition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sofic.automata.nfa import NFA


class BuchiAutomaton(NFA):
    """ω-automaton with Büchi acceptance on ultimately periodic inputs."""

    def accepts_lasso(self, prefix: Sequence[Any], loop: Sequence[Any]) -> bool:
        """Accept if infinitely repeating ``loop`` after ``prefix`` visits accepting states infinitely often."""
        from sofic.automata.buchi_simulation import accepts_lasso_buchi

        return accepts_lasso_buchi(self, prefix, loop)

    def accepts_omega(self, word: Sequence[Any]) -> bool:
        from sofic.automata.buchi_simulation import accepts_omega_buchi

        return accepts_omega_buchi(self, word)

    def accepted_lasso(self) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
        """Return ``(prefix, loop)`` with ``prefix loop^omega`` accepted, or ``None`` when the ω-language is empty."""
        from sofic.automata.buchi_simulation import accepted_lasso_buchi

        return accepted_lasso_buchi(self)

    def is_empty(self) -> bool:
        """Return whether no infinite word is accepted (Büchi emptiness :cite:`BaierKatoen2008`)."""
        return self.accepted_lasso() is None

    def accepted_word(self) -> tuple[Any, ...] | None:
        raise NotImplementedError("a Büchi automaton accepts infinite words; use accepted_lasso()")

    def is_universal(self, alphabet: frozenset[Any] | None = None) -> bool:
        raise NotImplementedError("Büchi universality needs Büchi complementation, which sofic does not implement")

    def includes(self, other: Any) -> bool:
        raise NotImplementedError("Büchi inclusion needs Büchi complementation, which sofic does not implement")
