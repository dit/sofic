r"""Active automata learning: L* and a discrimination-tree (TTT-family) learner.

Active learning reconstructs an automaton from a *teacher* answering two kinds of
query: **membership** ("is this word in the language / what does the machine
output?") and **equivalence** ("is my hypothesis correct, and if not, give a
counterexample"). This module provides

* oracle protocols (:class:`MembershipOracle`, :class:`EquivalenceOracle` and
  their Mealy analogues) with adapters over sofic models,
* Angluin's **L\*** :cite:`Angluin1987` for :class:`~sofic.automata.dfa.DFA` and
  its Mealy variant :cite:`Shahbaz2009`, both using the Rivest-Schapire
  counterexample analysis :cite:`RivestSchapire1993`, and
* a redundancy-free **discrimination-tree** learner in the TTT family
  :cite:`KearnsVazirani1994,Isberner2014`.

These complement the existing NL\* átomaton learner
(:func:`sofic.automata.learning.learn_maximized_prime_atomaton`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sofic.automata.dfa import DFA
from sofic.automata.transducers import MealyMachine

__all__ = [
    "MembershipOracle",
    "EquivalenceOracle",
    "MealyMembershipOracle",
    "MealyEquivalenceOracle",
    "FunctionMembershipOracle",
    "LanguageMembershipOracle",
    "FunctionMealyOracle",
    "TransducerOutputOracle",
    "ExhaustiveEquivalenceOracle",
    "RandomWalkEquivalenceOracle",
    "MealyExhaustiveEquivalenceOracle",
    "learn_dfa_lstar",
    "learn_dfa_ttt",
    "learn_mealy_lstar",
    "learn_dfa_from_language",
    "learn_mealy_from_transducer",
]

Word = tuple[Any, ...]


# --------------------------------------------------------------------------- oracles


@runtime_checkable
class MembershipOracle(Protocol):
    """Answers whether a word belongs to the target language."""

    def member(self, word: Sequence[Any]) -> bool: ...


@runtime_checkable
class EquivalenceOracle(Protocol):
    """Returns a counterexample word where ``hypothesis`` disagrees, or ``None``."""

    def find_counterexample(self, hypothesis: DFA) -> Word | None: ...


@runtime_checkable
class MealyMembershipOracle(Protocol):
    """Returns the output word produced by the target for an input word."""

    def output(self, word: Sequence[Any]) -> Word: ...


@runtime_checkable
class MealyEquivalenceOracle(Protocol):
    """Returns an input word where the Mealy ``hypothesis`` disagrees, or ``None``."""

    def find_counterexample(self, hypothesis: MealyMachine) -> Word | None: ...


class FunctionMembershipOracle:
    """Wrap a boolean predicate as a :class:`MembershipOracle`."""

    def __init__(self, predicate: Callable[[Word], bool]) -> None:
        self._predicate = predicate

    def member(self, word: Sequence[Any]) -> bool:
        return bool(self._predicate(tuple(word)))


class LanguageMembershipOracle:
    """Membership over any sofic model exposing ``recognizes`` or ``__contains__``.

    Works with :class:`~sofic.automata.dfa.DFA`, :class:`~sofic.automata.nfa.NFA`,
    átomata, and any :class:`~sofic.automata.languages.base.RegularLanguage`. For
    a sofic shift or ε-machine, pass its support automaton
    (``model.to_support_dfa()``).
    """

    def __init__(self, model: Any) -> None:
        if hasattr(model, "recognizes"):
            self._member = model.recognizes
        elif hasattr(model, "__contains__"):
            self._member = model.__contains__
        else:
            raise TypeError(f"{type(model).__name__} exposes neither recognizes() nor __contains__()")

    def member(self, word: Sequence[Any]) -> bool:
        return bool(self._member(tuple(word)))


class FunctionMealyOracle:
    """Wrap an output function as a :class:`MealyMembershipOracle`."""

    def __init__(self, output: Callable[[Word], Sequence[Any]]) -> None:
        self._output = output

    def output(self, word: Sequence[Any]) -> Word:
        return tuple(self._output(tuple(word)))


class TransducerOutputOracle:
    """Output oracle backed by a deterministic, complete :class:`MealyMachine`."""

    def __init__(self, machine: MealyMachine) -> None:
        self._machine = machine

    def output(self, word: Sequence[Any]) -> Word:
        outputs = self._machine.transduce(tuple(word))
        if not outputs:
            raise ValueError(f"target produced no output for {tuple(word)!r}; is it complete?")
        return next(iter(outputs))


def _words_up_to(max_length: int, alphabet: Sequence[Any]) -> Iterator[Word]:
    frontier: list[Word] = [()]
    yield ()
    for _ in range(max_length):
        nxt: list[Word] = []
        for word in frontier:
            for symbol in alphabet:
                extended = (*word, symbol)
                yield extended
                nxt.append(extended)
        frontier = nxt


class ExhaustiveEquivalenceOracle:
    """Bounded exhaustive equivalence test for DFA hypotheses."""

    def __init__(self, membership: MembershipOracle, alphabet: Iterable[Any], *, max_length: int = 10) -> None:
        self._membership = membership
        self._alphabet = tuple(sorted(alphabet, key=repr))
        self._max_length = int(max_length)

    def find_counterexample(self, hypothesis: DFA) -> Word | None:
        for word in _words_up_to(self._max_length, self._alphabet):
            if self._membership.member(word) != hypothesis.recognizes(word):
                return word
        return None


class RandomWalkEquivalenceOracle:
    """Randomized equivalence test drawing random input words for DFA hypotheses."""

    def __init__(
        self,
        membership: MembershipOracle,
        alphabet: Iterable[Any],
        *,
        num_walks: int = 2000,
        max_steps: int = 30,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self._membership = membership
        self._alphabet = tuple(sorted(alphabet, key=repr))
        self._num_walks = int(num_walks)
        self._max_steps = int(max_steps)
        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def find_counterexample(self, hypothesis: DFA) -> Word | None:
        n_symbols = len(self._alphabet)
        for _ in range(self._num_walks):
            length = int(self._rng.integers(0, self._max_steps + 1))
            word = tuple(self._alphabet[int(self._rng.integers(0, n_symbols))] for _ in range(length))
            if self._membership.member(word) != hypothesis.recognizes(word):
                return word
        return None


class MealyExhaustiveEquivalenceOracle:
    """Bounded exhaustive equivalence test for Mealy hypotheses."""

    def __init__(self, oracle: MealyMembershipOracle, alphabet: Iterable[Any], *, max_length: int = 10) -> None:
        self._oracle = oracle
        self._alphabet = tuple(sorted(alphabet, key=repr))
        self._max_length = int(max_length)

    def find_counterexample(self, hypothesis: MealyMachine) -> Word | None:
        for word in _words_up_to(self._max_length, self._alphabet):
            if not word:
                continue
            produced = hypothesis.transduce(word)
            hyp_out = next(iter(produced)) if produced else None
            if self._oracle.output(word) != hyp_out:
                return word
        return None


# ------------------------------------------------------------------------------ L*


class _MembershipCache:
    def __init__(self, oracle: MembershipOracle) -> None:
        self._oracle = oracle
        self._cache: dict[Word, bool] = {}

    def member(self, word: Word) -> bool:
        value = self._cache.get(word, None)
        if value is None:
            value = bool(self._oracle.member(word))
            self._cache[word] = value
        return value


def _build_dfa_from_rows(
    access: Iterable[Word],
    experiments: Sequence[Word],
    member: Callable[[Word], bool],
    alphabet: Sequence[Any],
) -> DFA:
    def row(word: Word) -> tuple[bool, ...]:
        return tuple(member(word + suffix) for suffix in experiments)

    representatives: dict[tuple[bool, ...], Word] = {}
    for word in sorted(access, key=lambda w: (len(w), repr(w))):
        representatives.setdefault(row(word), word)

    dfa = DFA(input_alphabet=frozenset(alphabet))
    for state in representatives.values():
        dfa.graph.add_state(state)
    for state in representatives.values():
        for symbol in alphabet:
            target = representatives[row(state + (symbol,))]
            dfa.add_transition(state, target, symbol)
    dfa.initial_states = frozenset({representatives[row(())]})
    dfa.accepting_states = frozenset(state for signature, state in representatives.items() if signature[0])
    dfa.validate()
    return dfa


def learn_dfa_lstar(
    alphabet: Iterable[Any],
    membership: MembershipOracle,
    equivalence: EquivalenceOracle,
    *,
    max_rounds: int = 100,
) -> DFA:
    r"""Learn a minimal DFA with Angluin's L\* algorithm :cite:`Angluin1987`.

    Maintains a closed and consistent observation table over access prefixes and
    suffix experiments, building a hypothesis DFA and refining it from each
    counterexample until the equivalence oracle is satisfied.
    """
    alphabet = tuple(sorted(alphabet, key=repr))
    cache = _MembershipCache(membership)
    member = cache.member

    prefixes: set[Word] = {()}
    experiments: list[Word] = [()]

    def row(word: Word) -> tuple[bool, ...]:
        return tuple(member(word + suffix) for suffix in experiments)

    for _ in range(max_rounds):
        while True:
            prefix_rows = {row(p) for p in prefixes}
            unclosed = None
            for p in prefixes:
                for symbol in alphabet:
                    if row(p + (symbol,)) not in prefix_rows:
                        unclosed = p + (symbol,)
                        break
                if unclosed is not None:
                    break
            if unclosed is not None:
                prefixes.add(unclosed)
                continue

            inconsistency = _find_inconsistency(prefixes, experiments, alphabet, row, member)
            if inconsistency is not None:
                experiments.append(inconsistency)
                continue
            break

        closure = set(prefixes)
        for p in list(prefixes):
            for symbol in alphabet:
                closure.add(p + (symbol,))
        hypothesis = _build_dfa_from_rows(closure, experiments, member, alphabet)

        counterexample = equivalence.find_counterexample(hypothesis)
        if counterexample is None:
            return hypothesis
        for index in range(len(counterexample) + 1):
            prefixes.add(tuple(counterexample[:index]))

    raise RuntimeError("L* did not converge within max_rounds; check the equivalence oracle")


def _find_inconsistency(
    prefixes: Iterable[Word],
    experiments: Sequence[Word],
    alphabet: Sequence[Any],
    row: Callable[[Word], tuple[bool, ...]],
    member: Callable[[Word], bool],
) -> Word | None:
    prefixes = list(prefixes)
    for i, p in enumerate(prefixes):
        for q in prefixes[i + 1 :]:
            if row(p) != row(q):
                continue
            for symbol in alphabet:
                rp, rq = row(p + (symbol,)), row(q + (symbol,))
                if rp != rq:
                    for index, suffix in enumerate(experiments):
                        if rp[index] != rq[index]:
                            return (symbol, *suffix)
    return None


# ----------------------------------------------------------- discrimination-tree (TTT)


class _DTNode:
    __slots__ = ("discriminator", "zero", "one", "access")

    def __init__(self, *, access: Word | None = None, discriminator: Word | None = None) -> None:
        self.discriminator = discriminator
        self.access = access
        self.zero: _DTNode | None = None
        self.one: _DTNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.discriminator is None


def learn_dfa_ttt(
    alphabet: Iterable[Any],
    membership: MembershipOracle,
    equivalence: EquivalenceOracle,
    *,
    max_rounds: int = 100,
) -> DFA:
    """Learn a minimal DFA with a discrimination-tree active learner.

    Uses a binary **discrimination tree** of distinguishing suffixes -- the
    redundancy-free state representation of the TTT family
    :cite:`KearnsVazirani1994,Isberner2014` -- refined by Rivest-Schapire
    counterexample decomposition :cite:`RivestSchapire1993`. Each counterexample
    splits exactly one leaf, so the tree grows to the minimal number of states.
    (Discriminator finalization, TTT's further space optimization, is not
    performed; the learned DFA is identical.)
    """
    alphabet = tuple(sorted(alphabet, key=repr))
    cache = _MembershipCache(membership)
    member = cache.member

    root = _DTNode(access=())

    def sift(word: Word) -> _DTNode:
        node = root
        while not node.is_leaf:
            node = node.one if member(word + node.discriminator) else node.zero
        return node

    def build() -> DFA:
        leaves: list[_DTNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves.append(node)
            else:
                stack.extend((node.zero, node.one))
        dfa = DFA(input_alphabet=frozenset(alphabet))
        for leaf in leaves:
            dfa.graph.add_state(leaf.access)
        for leaf in leaves:
            for symbol in alphabet:
                target = sift(leaf.access + (symbol,))
                dfa.add_transition(leaf.access, target.access, symbol)
        dfa.initial_states = frozenset({sift(()).access})
        dfa.accepting_states = frozenset(leaf.access for leaf in leaves if member(leaf.access))
        dfa.validate()
        return dfa

    for _ in range(max_rounds):
        hypothesis = build()
        counterexample = equivalence.find_counterexample(hypothesis)
        if counterexample is None:
            return hypothesis
        _split_leaf(counterexample, sift, member, alphabet)

    raise RuntimeError("TTT did not converge within max_rounds; check the equivalence oracle")


def _hypothesis_access(word: Word, sift: Callable[[Word], _DTNode]) -> Word:
    return sift(word).access


def _split_leaf(
    counterexample: Word,
    sift: Callable[[Word], _DTNode],
    member: Callable[[Word], bool],
    alphabet: Sequence[Any],
) -> None:
    counterexample = tuple(counterexample)
    length = len(counterexample)

    def alpha(index: int) -> Word:
        return _hypothesis_access(counterexample[:index], sift) + counterexample[index:]

    base = member(alpha(0))
    breakpoint_index = None
    for index in range(length):
        if member(alpha(index + 1)) != base:
            breakpoint_index = index
            break
    if breakpoint_index is None:  # pragma: no cover - guaranteed by a valid counterexample
        raise RuntimeError("counterexample analysis found no breakpoint")

    state_access = _hypothesis_access(counterexample[:breakpoint_index], sift)
    symbol = counterexample[breakpoint_index]
    discriminator = counterexample[breakpoint_index + 1 :]
    new_access = state_access + (symbol,)

    leaf = sift(new_access)
    old_access = leaf.access

    old_leaf = _DTNode(access=old_access)
    new_leaf = _DTNode(access=new_access)
    leaf.discriminator = discriminator
    leaf.access = None
    if member(old_access + discriminator):
        leaf.one, leaf.zero = old_leaf, new_leaf
    else:
        leaf.one, leaf.zero = new_leaf, old_leaf


# ------------------------------------------------------------------------ Mealy L*


def learn_mealy_lstar(
    alphabet: Iterable[Any],
    oracle: MealyMembershipOracle,
    equivalence: MealyEquivalenceOracle,
    *,
    max_rounds: int = 100,
) -> MealyMachine:
    r"""Learn a minimal Mealy machine with the L\*-Mealy algorithm.

    The Mealy adaptation of L\* :cite:`Shahbaz2009`: table cells hold the last
    output symbol of an output query, suffix experiments are seeded with the
    single input symbols, and states are distinguished by their output rows.
    """
    alphabet = tuple(sorted(alphabet, key=repr))
    output_cache: dict[Word, Word] = {}

    def out(word: Word) -> Word:
        value = output_cache.get(word)
        if value is None:
            value = tuple(oracle.output(word))
            output_cache[word] = value
        return value

    def cell(prefix: Word, suffix: Word) -> Any:
        produced = out(prefix + suffix)
        return produced[-1] if produced else None

    prefixes: set[Word] = {()}
    experiments: list[Word] = [(symbol,) for symbol in alphabet]

    def row(word: Word) -> tuple[Any, ...]:
        return tuple(cell(word, suffix) for suffix in experiments)

    for _ in range(max_rounds):
        while True:
            prefix_rows = {row(p) for p in prefixes}
            unclosed = None
            for p in prefixes:
                for symbol in alphabet:
                    if row(p + (symbol,)) not in prefix_rows:
                        unclosed = p + (symbol,)
                        break
                if unclosed is not None:
                    break
            if unclosed is not None:
                prefixes.add(unclosed)
                continue

            inconsistency = _find_mealy_inconsistency(prefixes, experiments, alphabet, row)
            if inconsistency is not None:
                experiments.append(inconsistency)
                continue
            break

        hypothesis = _build_mealy(prefixes, alphabet, row, cell)
        counterexample = equivalence.find_counterexample(hypothesis)
        if counterexample is None:
            return hypothesis
        for index in range(1, len(counterexample) + 1):
            prefixes.add(tuple(counterexample[:index]))

    raise RuntimeError("L*-Mealy did not converge within max_rounds; check the equivalence oracle")


def _find_mealy_inconsistency(
    prefixes: Iterable[Word],
    experiments: Sequence[Word],
    alphabet: Sequence[Any],
    row: Callable[[Word], tuple[Any, ...]],
) -> Word | None:
    prefixes = list(prefixes)
    for i, p in enumerate(prefixes):
        for q in prefixes[i + 1 :]:
            if row(p) != row(q):
                continue
            for symbol in alphabet:
                rp, rq = row(p + (symbol,)), row(q + (symbol,))
                if rp != rq:
                    for index, suffix in enumerate(experiments):
                        if rp[index] != rq[index]:
                            return (symbol, *suffix)
    return None


def _build_mealy(
    prefixes: Iterable[Word],
    alphabet: Sequence[Any],
    row: Callable[[Word], tuple[Any, ...]],
    cell: Callable[[Word, Word], Any],
) -> MealyMachine:
    representatives: dict[tuple[Any, ...], Word] = {}
    for word in sorted(prefixes, key=lambda w: (len(w), repr(w))):
        representatives.setdefault(row(word), word)

    outputs: set[Any] = set()
    transitions: list[tuple[Word, Word, Any, Any]] = []
    for state in representatives.values():
        for symbol in alphabet:
            target = representatives[row(state + (symbol,))]
            output = cell(state, (symbol,))
            outputs.add(output)
            transitions.append((state, target, symbol, output))

    machine = MealyMachine(
        input_alphabet=frozenset(alphabet),
        output_alphabet=frozenset(outputs),
        initial_states=frozenset({representatives[row(())]}),
    )
    for state in representatives.values():
        machine.graph.add_state(state)
    for source, target, symbol, output in transitions:
        machine.add_transition(source, target, symbol, output=output)
    machine.validate()
    return machine


# ------------------------------------------------------------------- convenience


def learn_dfa_from_language(
    target: Any,
    alphabet: Iterable[Any],
    *,
    algorithm: str = "lstar",
    max_length: int = 12,
    max_rounds: int = 100,
) -> DFA:
    """Learn a DFA for a sofic language model using a bounded exhaustive teacher.

    ``target`` is any model accepted by :class:`LanguageMembershipOracle` (a DFA,
    NFA, átomaton, or :class:`~sofic.automata.languages.base.RegularLanguage`);
    for a sofic shift or ε-machine pass ``model.to_support_dfa()``. ``algorithm``
    selects ``"lstar"`` or ``"ttt"``.
    """
    membership = LanguageMembershipOracle(target)
    equivalence = ExhaustiveEquivalenceOracle(membership, alphabet, max_length=max_length)
    if algorithm == "lstar":
        return learn_dfa_lstar(alphabet, membership, equivalence, max_rounds=max_rounds)
    if algorithm == "ttt":
        return learn_dfa_ttt(alphabet, membership, equivalence, max_rounds=max_rounds)
    raise ValueError(f"unknown algorithm {algorithm!r}; use 'lstar' or 'ttt'")


def learn_mealy_from_transducer(
    target: MealyMachine,
    alphabet: Iterable[Any] | None = None,
    *,
    max_length: int = 12,
    max_rounds: int = 100,
) -> MealyMachine:
    """Learn a Mealy machine equivalent to ``target`` with a bounded exhaustive teacher."""
    inputs = alphabet if alphabet is not None else target.alphabets()[0]
    oracle = TransducerOutputOracle(target)
    equivalence = MealyExhaustiveEquivalenceOracle(oracle, inputs, max_length=max_length)
    return learn_mealy_lstar(inputs, oracle, equivalence, max_rounds=max_rounds)
