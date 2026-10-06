r"""NL\*: Angluin-style active learning of canonical RFSAs :cite:`Bollig2009`.

NL\* keeps an observation table with rows for the access words ``U`` and their
one-symbol extensions ``U Sigma``, and columns for a suffix-closed set ``V`` of
experiments. Rows are compared pointwise: ``r <= r'`` when ``r'`` accepts every
experiment ``r`` accepts, and the join of rows is their pointwise ``or``. A row
is *prime* when it is not the join of the rows strictly below it.

The table is

* **RFSA-closed** when every row of ``U Sigma`` is the join of the prime rows of
  ``U`` below it, and
* **RFSA-consistent** when ``row(u') <= row(u)`` implies
  ``row(u' a) <= row(u a)`` for all ``u, u'`` in ``U`` and symbols ``a``.

A closed, consistent table yields the hypothesis whose states are the prime
rows of ``U``. Counterexamples add all of their suffixes to ``V``. When the
equivalence oracle accepts, the hypothesis is the canonical RFSA of the target.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sofic.automata.active import (
    AutomatonEquivalenceOracle,
    EquivalenceOracle,
    ExhaustiveEquivalenceOracle,
    LanguageMembershipOracle,
    MembershipOracle,
    _MembershipCache,
)
from sofic.automata.atomaton import MaximizedPrimeAtomaton
from sofic.automata.base import LabeledAutomaton
from sofic.automata.rfsa import CanonicalRFSA

Word = tuple[Any, ...]
Row = tuple[bool, ...]


def _leq(left: Row, right: Row) -> bool:
    return all(not a or b for a, b in zip(left, right, strict=True))


def _join(rows: Iterable[Row], width: int) -> Row:
    result = [False] * width
    for row in rows:
        result = [a or b for a, b in zip(result, row, strict=True)]
    return tuple(result)


def _primes(rows: set[Row], width: int) -> set[Row]:
    return {row for row in rows if _join((r for r in rows if r != row and _leq(r, row)), width) != row}


def learn_rfsa_nlstar(
    alphabet: Iterable[Any],
    membership: MembershipOracle,
    equivalence: EquivalenceOracle,
    *,
    max_rounds: int = 100,
) -> CanonicalRFSA:
    r"""Learn the canonical RFSA of the target language with NL\* :cite:`Bollig2009`."""
    symbols = tuple(sorted(alphabet, key=repr))
    member = _MembershipCache(membership).member
    access: list[Word] = [()]
    experiments: list[Word] = [()]

    def row(word: Word) -> Row:
        return tuple(member(word + suffix) for suffix in experiments)

    for _ in range(max_rounds):
        while True:
            width = len(experiments)
            upper = {u: row(u) for u in access}
            lower = {u + (a,): row(u + (a,)) for u in access for a in symbols}
            all_primes = _primes(set(upper.values()) | set(lower.values()), width)
            primes_upper = all_primes & set(upper.values())

            unclosed = next(
                (
                    w
                    for w, r in sorted(lower.items(), key=lambda item: (len(item[0]), repr(item[0])))
                    if r in all_primes and r not in upper.values()
                ),
                None,
            )
            if unclosed is not None:
                access.append(unclosed)
                continue

            inconsistency = _find_inconsistency(access, symbols, upper, row, experiments)
            if inconsistency is not None:
                experiments.append(inconsistency)
                continue
            break

        hypothesis = _hypothesis(access, symbols, upper, primes_upper, row)
        counterexample = equivalence.find_counterexample(hypothesis)
        if counterexample is None:
            return hypothesis
        for start in range(len(counterexample) + 1):
            suffix = tuple(counterexample[start:])
            if suffix not in experiments:
                experiments.append(suffix)
    raise RuntimeError(f"NL* did not converge within {max_rounds} equivalence rounds")


def _find_inconsistency(access, symbols, upper, row, experiments) -> Word | None:
    for u in access:
        for other in access:
            if u == other or not _leq(upper[other], upper[u]):
                continue
            for symbol in symbols:
                below, above = row(other + (symbol,)), row(u + (symbol,))
                if not _leq(below, above):
                    index = next(i for i, (b, a) in enumerate(zip(below, above, strict=True)) if b and not a)
                    return (symbol, *experiments[index])
    return None


def _hypothesis(access, symbols, upper, primes_upper, row) -> CanonicalRFSA:
    representative: dict[Row, Word] = {}
    for u in sorted(access, key=lambda w: (len(w), repr(w))):
        if upper[u] in primes_upper:
            representative.setdefault(upper[u], u)
    states = sorted(representative, key=lambda r: (len(representative[r]), repr(representative[r])))
    name = {r: index for index, r in enumerate(states)}
    epsilon_row = upper[()]
    rfsa = CanonicalRFSA(
        input_alphabet=frozenset(symbols),
        initial_states=frozenset(name[r] for r in states if _leq(r, epsilon_row)),
        accepting_states=frozenset(name[r] for r in states if r[0]),
    )
    for r in states:
        rfsa.graph.add_state(name[r])
    for r in states:
        for symbol in symbols:
            successor = row(representative[r] + (symbol,))
            for target in states:
                if _leq(target, successor):
                    rfsa.add_transition(name[r], name[target], symbol)
    return rfsa


class _ReversedMembership:
    def __init__(self, membership: MembershipOracle) -> None:
        self._membership = membership

    def member(self, word: Sequence[Any]) -> bool:
        return bool(self._membership.member(tuple(reversed(tuple(word)))))


class _ReversedEquivalence:
    def __init__(self, equivalence: EquivalenceOracle) -> None:
        self._equivalence = equivalence

    def find_counterexample(self, hypothesis: Any) -> Word | None:
        counterexample = self._equivalence.find_counterexample(hypothesis.reverse())
        return None if counterexample is None else tuple(reversed(tuple(counterexample)))


def learn_prime_atomaton_nlstar(
    alphabet: Iterable[Any],
    membership: MembershipOracle,
    equivalence: EquivalenceOracle,
    *,
    max_rounds: int = 100,
) -> MaximizedPrimeAtomaton:
    r"""Learn the maximized prime átomaton by running NL\* on the reversed target.

    The maximized prime átomaton of ``L`` is the reverse of the canonical RFSA of
    :math:`L^R` :cite:`MaarandTamm2022`, so NL\* is run against reversed
    membership and equivalence oracles and its result reversed.
    """
    from sofic.automata.canonical_dual import dual_atomaton_from_rfsa

    reversed_rfsa = learn_rfsa_nlstar(
        alphabet, _ReversedMembership(membership), _ReversedEquivalence(equivalence), max_rounds=max_rounds
    )
    return dual_atomaton_from_rfsa(reversed_rfsa)


def learn_rfsa_from_language(
    target: Any,
    alphabet: Iterable[Any],
    *,
    max_length: int = 12,
    max_rounds: int = 100,
) -> CanonicalRFSA:
    r"""Learn the canonical RFSA of ``target`` with NL\*.

    Uses an exact equivalence oracle when ``target`` is a finite automaton and a
    bounded exhaustive one (words up to ``max_length``) otherwise.
    """
    membership = LanguageMembershipOracle(target)
    if isinstance(target, LabeledAutomaton):
        equivalence: EquivalenceOracle = AutomatonEquivalenceOracle(target, alphabet)
    else:
        equivalence = ExhaustiveEquivalenceOracle(membership, alphabet, max_length=max_length)
    return learn_rfsa_nlstar(alphabet, membership, equivalence, max_rounds=max_rounds)
