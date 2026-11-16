r"""Xor automata: NFAs read with GF(2)-weighted acceptance.

A xor automaton has the shape of an NFA without ε-moves, but a word is accepted
when it labels an *odd* number of accepting paths :cite:`VuilleminGama2010`. It
is a weighted automaton over the two-element field: the reachable "state set"
after a word is a vector over GF(2), and transitions act on it linearly, by
symmetric difference of successor sets. Parallel duplicate transitions count
once, since the transitions form a relation.

Because the acceptance condition differs from an NFA's, :class:`XorAutomaton`
is deliberately not a :class:`~sofic.automata.base.LabeledAutomaton`: language
operations go through :meth:`XorAutomaton.determinize`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

from sofic.base import StateMachine
from sofic.graph import ATTR_SYMBOL, EPSILON

if TYPE_CHECKING:
    from sofic.automata.dfa import DFA


class XorAutomaton(StateMachine):
    """Finite automaton accepting the words with an odd number of accepting paths :cite:`VuilleminGama2010`."""

    input_alphabet: frozenset[Any]
    initial_states: frozenset[Hashable]
    accepting_states: frozenset[Hashable]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        accepting_states: frozenset[Hashable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_alphabet = input_alphabet if input_alphabet is not None else frozenset()
        self.initial_states = initial_states if initial_states is not None else frozenset()
        self.accepting_states = accepting_states if accepting_states is not None else frozenset()

    def validate(self) -> None:
        for state in self.initial_states | self.accepting_states:
            self._require(self.graph.has_state(state), f"missing state {state!r}")
        for transition in self.transitions():
            symbol = transition.data.get(ATTR_SYMBOL)
            self._require(symbol is not None and symbol is not EPSILON, "xor automata have no ε-transitions")
            self._require(symbol in self.input_alphabet, f"symbol {symbol!r} not in input alphabet")

    @classmethod
    def from_language(cls, language: Any, **kwargs: Any) -> XorAutomaton:
        """Return the minimal xor automaton; see :func:`~sofic.automata.canonical.closures.minimal_xor_automaton`."""
        from sofic.automata.canonical.closures import minimal_xor_automaton

        return minimal_xor_automaton(language)

    def add_transition(self, source: Hashable, target: Hashable, symbol: Any, **attrs: Any) -> int:
        return self.graph.add_transition(source, target, **{ATTR_SYMBOL: symbol, **attrs})

    def delta(self, state: Hashable, symbol: Any) -> set[Hashable]:
        return {t.target for t in self.graph.out_transitions(state) if t.data.get(ATTR_SYMBOL) == symbol}

    def step(self, states: frozenset[Hashable], symbol: Any) -> frozenset[Hashable]:
        """Return the GF(2) state vector after reading ``symbol`` from the vector ``states``."""
        result: set[Hashable] = set()
        for state in states:
            result ^= self.delta(state, symbol)
        return frozenset(result)

    def run(self, word: Sequence[Any]) -> frozenset[Hashable]:
        """Return the states reached from the initial states by an odd number of paths labeled ``word``."""
        current = frozenset(self.initial_states)
        for symbol in word:
            current = self.step(current, symbol)
        return current

    def recognizes(self, word: Sequence[Any]) -> bool:
        """Return whether ``word`` labels an odd number of accepting paths."""
        return len(self.run(word) & self.accepting_states) % 2 == 1

    def determinize(self, *, alphabet: frozenset[Any] | None = None) -> DFA:
        """Return the complete DFA on the reachable GF(2) state vectors (the xor subset construction)."""
        from sofic.automata.dfa import DFA

        symbols = sorted(self.input_alphabet if alphabet is None else alphabet, key=repr)
        start = frozenset(self.initial_states)
        dfa = DFA(input_alphabet=frozenset(symbols), initial_states=frozenset({start}))
        dfa.graph.add_state(start)
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for symbol in symbols:
                target = self.step(current, symbol)
                if target not in seen:
                    seen.add(target)
                    dfa.graph.add_state(target)
                    queue.append(target)
                dfa.add_transition(current, target, symbol)
        dfa.accepting_states = frozenset(s for s in seen if len(s & self.accepting_states) % 2 == 1)
        return dfa

    def minimize(self, *, alphabet: frozenset[Any] | None = None) -> DFA:
        """Return the minimal DFA of the accepted language."""
        return self.determinize(alphabet=alphabet).minimize()

    def reverse(self) -> XorAutomaton:
        """Return the xor automaton of the reversed language; reversal preserves the number of paths."""
        return XorAutomaton(
            input_alphabet=self.input_alphabet,
            initial_states=frozenset(self.accepting_states),
            accepting_states=frozenset(self.initial_states),
            graph=self.graph.reverse(),
        )
