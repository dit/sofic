"""Canonical automaton extraction from languages and observation tables.

Every construction works on the minimal complete DFA ``D`` of the language
``L``. Its states are the residuals (left quotients) ``u^{-1} L``, and inclusion
and union questions between residuals are decided exactly on ``D``:

* ``L_p <= L_q`` holds iff no pair reachable from ``(p, q)`` in ``D x D`` is
  accepting in the first component only;
* ``L_q <= union(L_p for p in S)`` holds iff no pair reachable from ``(q, S)`` in
  ``D x subsets(D)`` accepts on the left with no accepting state on the right.

A residual is *prime* when it is non-empty and not the union of the residuals
strictly contained in it :cite:`Denis2002`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from sofic.automata.base import LabeledAutomaton
from sofic.automata.canonical.atomaton import Atomaton, MaximizedPrimeAtomaton
from sofic.automata.canonical.rfsa import CanonicalRFSA
from sofic.automata.dfa import DFA
from sofic.automata.languages.base import AutomatonLanguage, RegularLanguage, as_language
from sofic.automata.learning.observation import ObservationTable
from sofic.automata.nfa import NFA
from sofic.graph import ATTR_SYMBOL
from sofic.states import sequential_labels


def _language_automaton(language: RegularLanguage | LabeledAutomaton) -> LabeledAutomaton:
    if isinstance(language, LabeledAutomaton):
        return language
    lang = as_language(language)  # type: ignore[arg-type]
    if isinstance(lang, AutomatonLanguage):
        return lang.automaton
    raise TypeError(f"cannot extract automaton from {type(language)!r}")


@dataclass(frozen=True)
class ResidualTable:
    """Minimal complete DFA of a language, viewed as its table of residuals."""

    dfa: DFA
    alphabet: tuple[Any, ...]
    start: Hashable
    accepting: frozenset[Hashable]
    delta: dict[tuple[Hashable, Any], Hashable]

    @classmethod
    def from_automaton(cls, aut: LabeledAutomaton) -> ResidualTable:
        from sofic.automata.algorithms import _transition_alphabet, complete, minimize

        symbols = _transition_alphabet(aut)
        source = aut if isinstance(aut, (NFA, DFA)) else NFA(**_automaton_kwargs(aut))
        dfa = complete(minimize(source, alphabet=symbols), symbols)
        if not dfa.initial_states:
            empty = DFA(input_alphabet=frozenset(symbols), initial_states=frozenset({"empty"}))
            empty.graph.add_state("empty")
            dfa = complete(empty, symbols)
        delta = {
            (transition.source, transition.data[ATTR_SYMBOL]): transition.target for transition in dfa.transitions()
        }
        return cls(
            dfa=dfa,
            alphabet=tuple(sorted(symbols, key=repr)),
            start=next(iter(dfa.initial_states)),
            accepting=frozenset(dfa.accepting_states),
            delta=delta,
        )

    @property
    def states(self) -> tuple[Hashable, ...]:
        return tuple(self.dfa.states())

    def is_empty(self, state: Hashable) -> bool:
        """Whether the residual ``L_state`` is empty."""
        seen = {state}
        queue = deque([state])
        while queue:
            current = queue.popleft()
            if current in self.accepting:
                return False
            for symbol in self.alphabet:
                target = self.delta[(current, symbol)]
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return True

    def includes(self, smaller: Hashable, larger: Hashable) -> bool:
        """Whether ``L_smaller <= L_larger``."""
        seen = {(smaller, larger)}
        queue = deque(seen)
        while queue:
            left, right = queue.popleft()
            if left in self.accepting and right not in self.accepting:
                return False
            for symbol in self.alphabet:
                pair = (self.delta[(left, symbol)], self.delta[(right, symbol)])
                if pair not in seen:
                    seen.add(pair)
                    queue.append(pair)
        return True

    def is_covered(self, state: Hashable, cover: Iterable[Hashable]) -> bool:
        """Whether ``L_state`` is contained in the union of ``L_p`` for ``p`` in ``cover``."""
        start = (state, frozenset(cover))
        seen = {start}
        queue = deque([start])
        while queue:
            left, right = queue.popleft()
            if left in self.accepting and not (right & self.accepting):
                return False
            for symbol in self.alphabet:
                pair = (self.delta[(left, symbol)], frozenset(self.delta[(q, symbol)] for q in right))
                if pair not in seen:
                    seen.add(pair)
                    queue.append(pair)
        return True

    def prime_states(self) -> tuple[Hashable, ...]:
        """States whose residual is prime, in breadth-first order from the start."""
        nonempty = [q for q in self._bfs_order() if not self.is_empty(q)]
        primes = []
        for q in nonempty:
            strictly_smaller = [p for p in nonempty if p != q and self.includes(p, q)]
            if not self.is_covered(q, strictly_smaller):
                primes.append(q)
        return tuple(primes)

    def _bfs_order(self) -> list[Hashable]:
        order = [self.start]
        seen = {self.start}
        for state in order:
            for symbol in self.alphabet:
                target = self.delta[(state, symbol)]
                if target not in seen:
                    seen.add(target)
                    order.append(target)
        return order

    def residual_automaton(self, state: Hashable) -> DFA:
        """DFA recognizing the residual ``L_state``."""
        result = self.dfa.copy()
        result.initial_states = frozenset({state})
        return result

    def left_language_automaton(self, state: Hashable) -> DFA:
        """DFA recognizing the words that lead from the start to ``state``."""
        result = self.dfa.copy()
        result.accepting_states = frozenset({state})
        return result


def _automaton_kwargs(aut: LabeledAutomaton) -> dict[str, Any]:
    return {
        "input_alphabet": aut.input_alphabet,
        "initial_states": aut.initial_states,
        "accepting_states": aut.accepting_states,
        "graph": aut.graph.copy(),
    }


def _labels(count: int) -> tuple[Hashable, ...]:
    return sequential_labels(count) if count <= 26 else tuple(range(count))


def canonical_rfsa_from_language(language: RegularLanguage | LabeledAutomaton) -> CanonicalRFSA:
    r"""Build the canonical residual finite-state automaton of a regular language.

    Following :cite:`Denis2002`, the states are the prime residuals of ``L``; the
    initial states are the primes contained in ``L``; the accepting states are
    the primes containing the empty word; and there is a transition
    :math:`p \xrightarrow{a} p'` exactly when :math:`L_{p'} \subseteq a^{-1} L_p`.
    The canonical RFSA is saturated (it has every such transition) and is never
    larger than the minimal DFA, often exponentially smaller.
    """
    table = ResidualTable.from_automaton(_language_automaton(language))
    primes = table.prime_states()
    name = dict(zip(primes, _labels(len(primes)), strict=True))
    rfsa = CanonicalRFSA(
        input_alphabet=frozenset(table.alphabet),
        initial_states=frozenset(name[p] for p in primes if table.includes(p, table.start)),
        accepting_states=frozenset(name[p] for p in primes if p in table.accepting),
    )
    for p in primes:
        rfsa.graph.add_state(name[p])
    for p in primes:
        for symbol in table.alphabet:
            successor = table.delta[(p, symbol)]
            for target in primes:
                if table.includes(target, successor):
                    rfsa.add_transition(name[p], name[target], symbol)
    return rfsa


def atomaton_from_language(language: RegularLanguage | LabeledAutomaton) -> Atomaton:
    """Build átomaton via double-reversal pipeline.

    The átomaton is the *reverse of the minimal DFA of the reverse language*
    (:cite:`BrzozowskiTamm2014`, Theorem 2), so the pipeline must stop at the
    reversal: determinizing once more would collapse it back to the minimal DFA
    of ``language``, which is Brzozowski's minimization rather than the
    átomaton.
    """
    from sofic.automata.languages.automaton_ops import minimal_dfa_from_language

    aut = _language_automaton(language)
    dfa = minimal_dfa_from_language(aut)
    rev = dfa.reverse().determinize().minimize()
    atom = rev.reverse()
    return Atomaton(
        input_alphabet=atom.input_alphabet,
        initial_states=atom.initial_states,
        accepting_states=atom.accepting_states,
        graph=atom.graph.copy(),
    )


def maximized_prime_atomaton_from_language(
    language: RegularLanguage | LabeledAutomaton,
) -> MaximizedPrimeAtomaton:
    r"""Build the maximized prime átomaton of a regular language :cite:`MaarandTamm2022`.

    It is the reverse of the canonical RFSA of :math:`L^R`. Equivalently, it is
    the subautomaton of the maximized átomaton of ``L`` -- the reverse of the
    saturated minimal DFA of :math:`L^R` -- on the maximized atoms whose
    quotients of :math:`L^R` are prime :cite:`Tamm2015`.
    """
    reversed_rfsa = canonical_rfsa_from_language(_language_automaton(language).reverse())
    return _reverse_into(MaximizedPrimeAtomaton, reversed_rfsa)


def _reverse_into(cls: type[NFA], aut: NFA) -> Any:
    reversed_aut = aut.reverse()
    return cls(
        input_alphabet=reversed_aut.input_alphabet,
        initial_states=reversed_aut.initial_states,
        accepting_states=reversed_aut.accepting_states,
        graph=reversed_aut.graph.copy(),
    )


def observation_to_minimal_dfa(table: ObservationTable) -> DFA:
    """Angluin-style DFA extraction from a closed, consistent table."""
    access = sorted(table.access_words, key=lambda w: (len(w), w))
    experiments = sorted(table.experiments, key=lambda w: (len(w), w))

    def signature(word: tuple[Any, ...]) -> tuple[bool, ...]:
        return tuple(table.membership.get(word + exp, False) for exp in experiments)

    classes: dict[tuple[bool, ...], tuple[Any, ...]] = {}
    for word in access:
        sig = signature(word)
        classes.setdefault(sig, word)

    state_for_word = {word: classes[signature(word)] for word in access}
    alphabet = _infer_alphabet(table)
    dfa = DFA(
        input_alphabet=alphabet,
        initial_states=frozenset({()}),
        accepting_states=frozenset(),
    )
    for word in access:
        dfa.graph.add_state(state_for_word[word])
    for word in access:
        for symbol in alphabet:
            target_word = word + (symbol,)
            if target_word in state_for_word:
                dfa.add_transition(state_for_word[word], state_for_word[target_word], symbol)
    accepting = {state_for_word[word] for word in access if table.membership.get(word, False)}
    dfa.accepting_states = frozenset(accepting)
    return dfa


def observation_to_canonical_rfsa(table: ObservationTable) -> CanonicalRFSA:
    r"""NL\*-style RFSA extraction from an RFSA-closed, RFSA-consistent table.

    States are the rows of the access words that are prime among all rows of
    the table, exactly as in the NL\* hypothesis :cite:`Bollig2009`.
    """
    from sofic.automata.learning.nlstar import _hypothesis, _primes

    experiments = sorted(table.experiments, key=lambda w: (len(w), w))
    if () in experiments:
        experiments.remove(())
    experiments.insert(0, ())
    symbols = tuple(sorted(_infer_alphabet(table), key=repr))
    access = sorted(table.access_words, key=lambda w: (len(w), w))

    def row(word: tuple[Any, ...]) -> tuple[bool, ...]:
        return tuple(table.membership.get(word + exp, False) for exp in experiments)

    upper = {u: row(u) for u in access}
    lower = {u + (a,): row(u + (a,)) for u in access for a in symbols}
    primes_upper = _primes(set(upper.values()) | set(lower.values()), len(experiments)) & set(upper.values())
    return _hypothesis(access, symbols, upper, primes_upper, row)


def observation_to_atomaton(table: ObservationTable) -> Atomaton:
    dfa = observation_to_minimal_dfa(table)
    return atomaton_from_language(dfa)


def observation_to_maximized_prime_atomaton(table: ObservationTable) -> MaximizedPrimeAtomaton:
    dfa = observation_to_minimal_dfa(table)
    return maximized_prime_atomaton_from_language(dfa)


def _infer_alphabet(table: ObservationTable) -> frozenset[Any]:
    symbols: set[Any] = set()
    for word in table.access_words | table.experiments:
        symbols.update(word)
    return frozenset(symbols)
