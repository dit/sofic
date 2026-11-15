r"""Active (query) learning of ε-machines: an L\*-style learner over predictive rows.

Angluin's L\* :cite:`Angluin1987` learns a minimal DFA from membership and
equivalence queries. Its stochastic analogue here learns an ε-machine
:cite:`Shalizi2001` from a teacher answering

* **probability queries** ``oracle.word_probability(w)``: the stationary
  probability :math:`P(w)` of a finite word, and
* optionally **equivalence queries** ``oracle.equivalent(machine)``: ``None`` when
  ``machine`` generates the target process, else a counterexample word whose
  probability differs.

Rows of the observation table are histories :math:`h` (prefixes) and a row's
content is the vector of conditional future probabilities :math:`P(s \mid h)` over
the test suffixes :math:`s`. A row is a finite-history *mixed state*
:cite:`Ellison2009`; distinct rows are distinct predictive distributions, so the
hypothesis is unifilar and minimal by construction. :class:`ProcessOracle`
answers both kinds of query exactly from any finite HMM.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from sofic.exceptions import MixedStateExplosionError, StochasticValidationError
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.matrices import emission_tensors
from sofic.generators.mealy import MealyHMM
from sofic.generators.prob import as_prob, has_symbolic, is_symbolic, probs_equal, simplify_prob, zeros
from sofic.generators.process_equivalence import (
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    _HistoryFutureWordList,
    _sorted_words,
    is_equal_process,
)
from sofic.generators.stationary import stationary_distribution_from_transition
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph
from sofic.inference.cssr.process import _recurrent_states

__all__ = [
    "ProbabilityOracle",
    "ProcessOracle",
    "learn_epsilon_machine_active",
]

Word = tuple[Any, ...]

_MASS_ATOL = 1e-12


@runtime_checkable
class ProbabilityOracle(Protocol):
    """Answers the stationary probability of a finite word."""

    def word_probability(self, word: Sequence[Any]) -> Any: ...


class ProcessOracle:
    r"""Probability and equivalence oracle answering exactly from a finite HMM.

    ``word_probability(w)`` is :math:`\pi T^{(w_0)} \cdots T^{(w_{L-1})} \mathbf{1}`
    with :math:`\\pi` the stationary state law, in exact sympy arithmetic when the
    HMM's probabilities are symbolic. ``equivalent(machine, history=h)`` compares
    ``machine`` (from its initial distribution) with the target process conditioned
    on the history ``h`` using
    :func:`~sofic.generators.process_equivalence.is_equal_process`, and returns
    ``None`` or a shortest word among the history/future basis words of the two
    presentations whose probability differs.

    ``probability_queries`` and ``equivalence_queries`` count the calls.
    """

    def __init__(self, hmm: HiddenMarkovModel, *, rtol: float | None = None, atol: float | None = None) -> None:
        self._hmm = hmm
        self._start, self._matrices = emission_tensors(hmm, policy="stationary")
        self._rtol = _DEFAULT_RTOL if rtol is None else float(rtol)
        self._atol = _DEFAULT_ATOL if atol is None else float(atol)
        self.probability_queries = 0
        self.equivalence_queries = 0

    def _mass(self, word: Sequence[Any]) -> np.ndarray:
        mass = self._start
        n = len(mass)
        for symbol in word:
            matrix = self._matrices.get(symbol)
            if matrix is None:
                return zeros((n,))
            mass = mass @ matrix
        return mass

    def word_probability(self, word: Sequence[Any]) -> Any:
        self.probability_queries += 1
        mass = self._mass(tuple(word))
        if mass.dtype == object:
            return simplify_prob(sum(mass.tolist()))
        return float(mass.sum())

    def equivalent(self, machine: HiddenMarkovModel, history: Sequence[Any] = ()) -> Word | None:
        self.equivalence_queries += 1
        mass = np.asarray(self._mass(tuple(history)), dtype=float)
        total = float(mass.sum())
        if total <= 0.0:
            raise ValueError(f"history {tuple(history)!r} has zero probability")
        target = _float_presentation(self._hmm)
        start = dict(zip(target.reindex().states, mass / total, strict=True))
        hypothesis = _float_presentation(machine)
        if is_equal_process(hypothesis, target, start2=start, rtol=self._rtol, atol=self._atol):
            return None
        return _process_counterexample(hypothesis, target, start, rtol=self._rtol, atol=self._atol)


def _float_presentation(hmm: HiddenMarkovModel) -> MealyHMM:
    """``hmm`` as a Mealy presentation with float probabilities and the same state names.

    :func:`~sofic.generators.process_equivalence.is_equal_process` needs float
    matrices: its rank tests fail on sympy-valued ones.
    """
    mealy = hmm.to_mealy()
    transitions = list(mealy.transitions())
    if not has_symbolic(t.data.get(ATTR_PROB, 0.0) for t in transitions) and not has_symbolic(
        mealy.initial_distribution.values()
    ):
        return mealy
    graph = TransitionGraph()
    for state in mealy.reindex().states:
        graph.add_state(state)
    for transition in transitions:
        data = dict(transition.data)
        data[ATTR_PROB] = float(data[ATTR_PROB])
        graph.add_transition(transition.source, transition.target, **data)
    return MealyHMM(
        graph=graph,
        initial_distribution={state: float(mass) for state, mass in mealy.initial_distribution.items()},
        observation_alphabet=mealy.observation_alphabet,
    )


def _process_counterexample(
    machine: HiddenMarkovModel,
    target: HiddenMarkovModel,
    start: dict[Any, float],
    *,
    rtol: float,
    atol: float,
) -> Word | None:
    """Shortest word ``h + f`` over both presentations' basis words whose probability differs.

    :func:`~sofic.generators.process_equivalence.is_equal_process` compares future
    probabilities and conditionals ``P(f | h)`` over these words, so a difference it
    detects is a difference in ``P(h)`` or ``P(h + f)``.
    """
    left = _HistoryFutureWordList.from_hmm(machine)
    right = _HistoryFutureWordList.from_hmm(target, start=start)
    histories = set(left.history_word_list()) | set(right.history_word_list())
    futures = set(left.future_word_list()) | set(right.future_word_list())
    futures |= {(symbol, *word) for word in list(futures) for symbol in left.alphabet}
    best, best_gap = None, 0.0
    for word in _sorted_words({h + f for h in histories for f in futures}):
        p = float(left.start @ left.future_vector(word))
        q = float(right.start @ right.future_vector(word))
        if not np.isclose(p, q, rtol=rtol, atol=atol):
            return word
        if abs(p - q) > best_gap:
            best, best_gap = word, abs(p - q)
    return best


def _is_exact(value: Any) -> bool:
    return is_symbolic(value) or isinstance(value, (int, Fraction))


def _is_zero(value: Any, *, atol: float = 0.0) -> bool:
    if is_symbolic(value):
        return probs_equal(value, 0)
    if isinstance(value, (int, Fraction)):
        return value == 0
    return float(value) <= atol


def _divide(numerator: Any, denominator: Any) -> Any:
    if is_symbolic(numerator) or is_symbolic(denominator):
        import sympy as sp

        return simplify_prob(sp.sympify(numerator) / sp.sympify(denominator))
    if _is_exact(numerator) and _is_exact(denominator):
        return Fraction(numerator) / Fraction(denominator)
    return float(numerator) / float(denominator)


def _values_match(left: Any, right: Any, tol: float) -> bool:
    if _is_exact(left) and _is_exact(right):
        if is_symbolic(left) or is_symbolic(right):
            return probs_equal(left, right)
        return left == right
    return abs(float(left) - float(right)) <= tol


def _machine_prob(value: Any) -> Any:
    if isinstance(value, Fraction):
        import sympy as sp

        return sp.Rational(value.numerator, value.denominator)
    return as_prob(value)


class _PredictiveTable:
    """Observation table of histories (``seed + access``) by test suffixes.

    State rows are kept pairwise distinct, so the table is consistent by
    construction (the Maler-Pnueli treatment of counterexamples).
    """

    def __init__(self, oracle: ProbabilityOracle, alphabet: Sequence[Any], seed: Word, tol: float) -> None:
        self._oracle = oracle
        self._alphabet = tuple(alphabet)
        self._seed = seed
        self._tol = tol
        self._probabilities: dict[Word, Any] = {}
        self.states: list[Word] = [()]
        self.tests: list[Word] = [(symbol,) for symbol in self._alphabet]

    @property
    def exact(self) -> bool:
        return all(_is_exact(value) for value in self._probabilities.values())

    def probability(self, word: Word) -> Any:
        word = self._seed + word
        value = self._probabilities.get(word)
        if value is None:
            value = self._oracle.word_probability(word)
            self._probabilities[word] = value
        return value

    def conditional(self, history: Word, suffix: Word) -> Any:
        return _divide(self.probability(history + suffix), self.probability(history))

    def row(self, history: Word) -> tuple[Any, ...]:
        return tuple(self.conditional(history, test) for test in self.tests)

    def match(self, history: Word) -> int | None:
        target = self.row(history)
        for index, state in enumerate(self.states):
            if all(_values_match(a, b, self._tol) for a, b in zip(self.row(state), target, strict=True)):
                return index
        return None

    def successors(self, history: Word) -> list[tuple[Any, Any]]:
        """``(symbol, P(symbol | history))`` for each positive-probability next symbol."""
        out = []
        for symbol in self._alphabet:
            if _is_zero(self.probability(history + (symbol,))):
                continue
            prob = self.conditional(history, (symbol,))
            if not _is_zero(prob, atol=_MASS_ATOL):
                out.append((symbol, prob))
        return out

    def add_tests(self, counterexample: Word) -> None:
        for index in range(len(counterexample)):
            suffix = tuple(counterexample[index:])
            if suffix not in self.tests:
                self.tests.append(suffix)

    def close(self, max_states: int) -> None:
        """Add one-symbol extensions until every extension's row matches a state row."""
        index = 0
        while index < len(self.states):
            history = self.states[index]
            for symbol, _prob in self.successors(history):
                extension = history + (symbol,)
                if self.match(extension) is None:
                    self.states.append(extension)
                    if len(self.states) > max_states:
                        raise MixedStateExplosionError(
                            f"the predictive rows did not close within max_states={max_states}; the process "
                            "may not be exactly synchronizable from this history, or it has infinitely many "
                            "transient mixed states (seed a synchronizing `history`)"
                        )
            index += 1

    def transitions(self) -> dict[int, dict[Any, tuple[int, Any]]]:
        """``delta[state][symbol] = (target, probability)`` of the closed table."""
        delta: dict[int, dict[Any, tuple[int, Any]]] = {}
        for index, history in enumerate(self.states):
            delta[index] = {}
            for symbol, prob in self.successors(history):
                target = self.match(history + (symbol,))
                if target is None:  # pragma: no cover - guaranteed by close()
                    raise RuntimeError("observation table is not closed")
                delta[index][symbol] = (target, prob)
        return delta

    def presentation(self, delta: dict[int, dict[Any, tuple[int, Any]]]) -> MealyHMM:
        """The unifilar presentation of all state rows, started at the seed history's row."""
        return _build_machine(MealyHMM, sorted(delta), delta, {0: 1}, self._alphabet)


def _build_machine(
    machine_type: type[MealyHMM],
    keep: Sequence[int],
    delta: dict[int, dict[Any, tuple[int, Any]]],
    initial: dict[int, Any],
    alphabet: Sequence[Any],
) -> MealyHMM:
    labels = {state: f"s{rank}" for rank, state in enumerate(keep)}
    graph = TransitionGraph()
    for state in keep:
        graph.add_state(labels[state])
    for state in keep:
        for symbol, (target, prob) in delta[state].items():
            graph.add_transition(
                labels[state], labels[target], **{ATTR_PROB: _machine_prob(prob), ATTR_EMISSION: symbol}
            )
    machine = machine_type(
        graph=graph,
        initial_distribution={labels[state]: _machine_prob(mass) for state, mass in initial.items()},
        observation_alphabet=frozenset(alphabet),
    )
    machine.validate()
    return machine


def _recurrent_epsilon_machine(delta: dict[int, dict[Any, tuple[int, Any]]], alphabet: Sequence[Any]) -> EpsilonMachine:
    """The ε-machine on the presentation's unique closed class, with its stationary law."""
    classes = _recurrent_states(
        {state: {symbol: (target,) for symbol, (target, _) in edges.items()} for state, edges in delta.items()}
    )
    if len(classes) != 1:
        raise StochasticValidationError(
            f"the learned presentation has {len(classes)} closed classes; the process is not ergodic"
        )
    keep = sorted(classes[0])
    position = {state: i for i, state in enumerate(keep)}
    probs = [_machine_prob(prob) for state in keep for _, prob in delta[state].values()]
    transition = zeros((len(keep), len(keep)), symbolic=has_symbolic(probs))
    for state in keep:
        for target, prob in delta[state].values():
            transition[position[state], position[target]] += _machine_prob(prob)
    pi = stationary_distribution_from_transition(transition)
    initial = {state: pi[position[state]] for state in keep}
    return cast(EpsilonMachine, _build_machine(EpsilonMachine, keep, delta, initial, alphabet))


def _presentation_probability(delta: dict[int, dict[Any, tuple[int, Any]]], word: Word) -> Any:
    state, prob = 0, 1
    for symbol in word:
        edge = delta[state].get(symbol)
        if edge is None:
            return 0
        state = edge[0]
        prob = prob * edge[1]
    return prob


def _bounded_counterexample(
    table: _PredictiveTable,
    delta: dict[int, dict[Any, tuple[int, Any]]],
    alphabet: Sequence[Any],
    *,
    max_length: int,
    tol: float,
) -> Word | None:
    """Shortest word up to ``max_length`` where ``P(w | seed)`` and the presentation disagree."""
    frontier: list[Word] = [()]
    for _ in range(max_length):
        extended = []
        for word in frontier:
            for symbol in alphabet:
                candidate = word + (symbol,)
                target = table.conditional((), candidate)
                hypothesis = _presentation_probability(delta, candidate)
                if not _values_match(target, hypothesis, tol):
                    return candidate
                if not _is_zero(target):
                    extended.append(candidate)
        frontier = extended
    return None


def learn_epsilon_machine_active(
    oracle: ProbabilityOracle,
    alphabet: Iterable[Any],
    *,
    tol: float = 1e-9,
    max_states: int = 1000,
    max_rounds: int = 100,
    history: Sequence[Any] = (),
    max_counterexample_length: int = 8,
) -> EpsilonMachine:
    r"""Learn an ε-machine from probability (and equivalence) queries, L\*-style :cite:`Angluin1987`.

    The observation table has histories ``history + u`` as rows and test suffixes
    ``s`` as columns, with entries :math:`P(s \mid \text{history}\, u)` from
    ``oracle.word_probability``. Tests start as the single symbols. The table is
    kept *closed* -- every positive-probability one-symbol extension of a state row
    matches some state row -- and *consistent*, since state rows are pairwise
    distinct and adding tests only refines them. The hypothesis has one state per
    distinct row and the transition :math:`P(x \mid h)` from row :math:`h` on
    :math:`x`, so it is unifilar by construction; started at the row of
    ``history`` it is a candidate for the conditional process
    :math:`P(\cdot \mid \text{history})`. It is checked with
    ``oracle.equivalent(hypothesis)`` (``oracle.equivalent(hypothesis,
    history=history)`` when seeded), or, if the oracle has no ``equivalent``,
    against every word of length at most ``max_counterexample_length``. Every
    suffix of a counterexample becomes a test, and the loop repeats. The returned
    ε-machine :cite:`Shalizi2001` is the hypothesis restricted to its closed class,
    started at its stationary distribution.

    Parameters
    ----------
    oracle
        Has ``word_probability(word)`` (stationary probability) and optionally
        ``equivalent(machine[, history])``; see :class:`ProcessOracle`.
    alphabet
        The process alphabet.
    tol
        Rows match when entries differ by at most ``tol``. Exact answers
        (``int``, :class:`~fractions.Fraction`, sympy) are compared exactly.
    max_states
        Cap on table rows; exceeding it raises
        :class:`~sofic.exceptions.MixedStateExplosionError`.
    max_rounds
        Cap on equivalence queries; exceeding it raises :class:`RuntimeError`.
    history
        Seed history prepended to every row (default empty).
    max_counterexample_length
        Word length searched when the oracle has no ``equivalent``.

    Notes
    -----
    With exact answers and an exact equivalence oracle, let :math:`N` be the number
    of distinct predictive distributions :math:`P(\cdot \mid \text{history}\, u)` over
    positive-probability ``u``. If :math:`N` is finite, the learner stops after at
    most :math:`N` equivalence queries (a counterexample whose suffixes are all tests
    is predicted correctly from a closed table by the chain rule, so each one adds a
    state) with exactly :math:`N` hypothesis states, and every table entry costs one
    or two probability queries. For a stationary ergodic process whose ε-machine is
    finite and exactly synchronizable :cite:`Travers2010`, every predictive state
    reaches a causal state, so the closed class is the ε-machine: same number of
    states, same process, same :math:`C_\mu` and :math:`h_\mu`.

    :math:`N` is finite when ``history`` is a synchronizing word (then all rows are
    causal states) or when the mixed-state presentation :cite:`Ellison2009` from the
    stationary law is finite. A finite ε-machine that is not exactly synchronizable
    has infinitely many mixed states, as can an exactly synchronizable one along a
    non-synchronizing transient; those exceed ``max_states`` unless seeded with a
    synchronizing ``history``. With float answers the result is approximate: row
    matching within ``tol`` is not transitive, and a counterexample that separates
    no rows at ``tol`` ends learning with the current hypothesis.

    Examples
    --------
    >>> from sofic.examples import even_process
    >>> from sofic.inference.active import ProcessOracle, learn_epsilon_machine_active
    >>> machine = learn_epsilon_machine_active(ProcessOracle(even_process(0.5)), ("0", "1"))
    >>> len(list(machine.states()))
    2
    """
    symbols = tuple(sorted(set(alphabet), key=repr))
    seed = tuple(history)
    table = _PredictiveTable(oracle, symbols, seed, float(tol))
    if _is_zero(table.probability(())):
        raise ValueError(f"history {seed!r} has zero probability")
    equivalent = getattr(oracle, "equivalent", None)

    table.close(max_states)
    for _ in range(max_rounds):
        delta = table.transitions()
        if equivalent is None:
            counterexample = _bounded_counterexample(
                table, delta, symbols, max_length=max_counterexample_length, tol=float(tol)
            )
        else:
            hypothesis = table.presentation(delta)
            counterexample = equivalent(hypothesis, history=seed) if seed else equivalent(hypothesis)
        if counterexample is None:
            return _recurrent_epsilon_machine(delta, symbols)
        size = len(table.states)
        table.add_tests(tuple(counterexample))
        table.close(max_states)
        if len(table.states) == size:
            if table.exact:
                raise RuntimeError(
                    f"counterexample {tuple(counterexample)!r} separates no rows; the oracles are inconsistent"
                )
            return _recurrent_epsilon_machine(delta, symbols)

    raise RuntimeError("active ε-machine learning did not converge within max_rounds; check the equivalence oracle")
