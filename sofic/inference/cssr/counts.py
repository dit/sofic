"""Empirical history counts for the CSSR family.

:class:`SuffixCounts` counts suffixes of a single process, :class:`JointSuffixCounts`
joint ``(input, output)`` pasts of a channel, and :class:`StackSuffixCounts`
``(suffix, stack)`` configurations of a visibly pushdown process.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from sofic.automata.learning.papni import DyckAlphabet

History = tuple[Any, ...]

JointHistory = tuple[tuple[Any, Any], ...]

ConfigurationHistory = tuple[tuple[Any, ...], tuple[Any, ...]]


def infer_alphabet(tokens: Sequence[Any], alphabet: Sequence[Any] | None) -> tuple[Any, ...]:
    """``alphabet`` as a tuple, or the distinct ``tokens`` sorted by ``repr``."""
    return tuple(sorted(set(tokens), key=repr)) if alphabet is None else tuple(alphabet)


def iter_suffixes(tokens: tuple[Any, ...], max_length: int) -> Iterator[tuple[int, tuple[Any, ...]]]:
    """Yield ``(t, tokens[t - L : t])`` for each position ``t`` and each ``L`` in ``0..min(t, max_length)``.

    These are the pasts, up to ``max_length`` long, that precede the token at ``t``.
    """
    for t in range(len(tokens)):
        for length in range(min(t, max_length) + 1):
            yield t, tokens[t - length : t]


@dataclass
class SuffixCounts:
    """Empirical counts of histories and following symbols in a sequence."""

    alphabet: tuple[Any, ...]
    history_counts: Counter[History] = field(default_factory=Counter)
    next_counts: dict[History, Counter[Any]] = field(default_factory=lambda: defaultdict(Counter))

    #: History key used as the fallback for an empty history set (overridden by stack counts).
    empty_history: ClassVar[History] = ()

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Any],
        *,
        alphabet: Sequence[Any] | None = None,
        max_length: int | None = None,
    ) -> SuffixCounts:
        seq = tuple(sequence)
        if not seq:
            raise ValueError("sequence must be non-empty")
        alphabet = infer_alphabet(seq, alphabet)
        unknown = set(seq) - set(alphabet)
        if unknown:
            raise ValueError(f"symbols {unknown!r} not in alphabet")
        counts = cls(alphabet=alphabet)
        for t, history in iter_suffixes(seq, max_length if max_length is not None else len(seq)):
            counts.history_counts[history] += 1
            counts.next_counts[history][seq[t]] += 1
        return counts

    def morph(self, history: History, *, smoothing: float = 0.0) -> dict[Any, float]:
        """MLE (optional additive smoothing) of P(next symbol | history)."""
        counts = self.next_counts.get(history, Counter())
        total = sum(counts.values())
        if total == 0:
            uniform = 1.0 / len(self.alphabet)
            return dict.fromkeys(self.alphabet, uniform)
        denom = total + smoothing * len(self.alphabet)
        return {symbol: (counts.get(symbol, 0) + smoothing) / denom for symbol in self.alphabet}

    def state_morph(self, histories: set[History], *, smoothing: float = 0.0) -> dict[Any, float]:
        """Weighted average of history morphs with weights from occurrence counts."""
        weights = {history: float(self.history_counts.get(history, 0)) for history in histories}
        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            return self.morph(self.empty_history, smoothing=smoothing)
        result = dict.fromkeys(self.alphabet, 0.0)
        for history, weight in weights.items():
            morph = self.morph(history, smoothing=smoothing)
            for symbol in self.alphabet:
                result[symbol] += weight * morph[symbol]
        return {symbol: prob / total_weight for symbol, prob in result.items()}

    def marginal_morph(self) -> dict[Any, float]:
        """Global next-symbol distribution (IID morph at L=0)."""
        counts = Counter()
        for _history, counter in self.next_counts.items():
            counts.update(counter)
        grand = sum(counts.values())
        if grand == 0:
            uniform = 1.0 / len(self.alphabet)
            return dict.fromkeys(self.alphabet, uniform)
        return {symbol: counts.get(symbol, 0) / grand for symbol in self.alphabet}

    def restricted_to(self, histories: set[History]) -> SuffixCounts:
        """Return a plain :class:`SuffixCounts` proxy limited to ``histories``.

        The morph/comparison helpers only read the history sets handed to them, so
        stack inference can reuse them by projecting its configuration counts onto a
        flat proxy without changing any results.
        """
        proxy = SuffixCounts(alphabet=self.alphabet)
        proxy.history_counts = Counter({h: self.history_counts.get(h, 0) for h in histories})
        proxy.next_counts = defaultdict(Counter)
        for history in histories:
            proxy.next_counts[history] = self.next_counts.get(history, Counter())
        return proxy


@dataclass
class JointSuffixCounts:
    """Empirical counts of joint pasts and following input-conditioned outputs."""

    input_alphabet: tuple[Any, ...]
    output_alphabet: tuple[Any, ...]
    history_counts: Counter[JointHistory] = field(default_factory=Counter)
    #: ``next_counts[history][input]`` is a Counter over following output symbols.
    next_counts: dict[JointHistory, dict[Any, Counter[Any]]] = field(default_factory=dict)

    @classmethod
    def from_sequences(
        cls,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        *,
        input_alphabet: Sequence[Any] | None = None,
        output_alphabet: Sequence[Any] | None = None,
        max_length: int,
    ) -> JointSuffixCounts:
        xs = tuple(inputs)
        ys = tuple(outputs)
        if len(xs) != len(ys):
            raise ValueError("inputs and outputs must have equal length")
        if not xs:
            raise ValueError("sequences must be non-empty")
        counts = cls(
            input_alphabet=infer_alphabet(xs, input_alphabet), output_alphabet=infer_alphabet(ys, output_alphabet)
        )
        for t, history in iter_suffixes(tuple(zip(xs, ys, strict=True)), max_length):
            counts.history_counts[history] += 1
            counts.next_counts.setdefault(history, {}).setdefault(xs[t], Counter())[ys[t]] += 1
        return counts

    def output_counts(self, histories: set[JointHistory], input_symbol: Any) -> Counter[Any]:
        observed: Counter[Any] = Counter()
        for history in histories:
            by_input = self.next_counts.get(history)
            if by_input is None:
                continue
            counter = by_input.get(input_symbol)
            if counter is not None:
                observed.update(counter)
        return observed

    def state_morph(self, histories: set[JointHistory], input_symbol: Any) -> dict[Any, float]:
        """Return ``P(output | histories, input_symbol)``."""
        observed = self.output_counts(histories, input_symbol)
        total = sum(observed.values())
        if total == 0:
            return {}
        return {symbol: observed.get(symbol, 0) / total for symbol in self.output_alphabet}


#: Aggregated output counts of a state: ``agg[input_symbol]`` is a Counter over outputs.
StateAggregate = dict[Any, Counter[Any]]


def _history_aggregate(counts: JointSuffixCounts, history: JointHistory) -> StateAggregate:
    return {input_symbol: Counter(counter) for input_symbol, counter in counts.next_counts.get(history, {}).items()}


def _merge_aggregate(target: StateAggregate, source: StateAggregate) -> None:
    for input_symbol, counter in source.items():
        target.setdefault(input_symbol, Counter()).update(counter)


def _state_aggregate(counts: JointSuffixCounts, histories: Iterable[JointHistory]) -> StateAggregate:
    aggregate: StateAggregate = {}
    for history in histories:
        _merge_aggregate(aggregate, _history_aggregate(counts, history))
    return aggregate


class StackSuffixCounts(SuffixCounts):
    """Empirical counts of (suffix, stack) histories and following symbols.

    Shares the morph / comparison machinery of :class:`SuffixCounts`; only the
    empty-history key and the sequence-scanning constructor differ.
    """

    empty_history: ClassVar[History] = ((), ())

    def __init__(
        self,
        alphabet: tuple[Any, ...],
        history_counts: Counter[ConfigurationHistory] | None = None,
        next_counts: dict[ConfigurationHistory, Counter[Any]] | None = None,
    ) -> None:
        super().__init__(
            alphabet=alphabet,
            history_counts=history_counts if history_counts is not None else Counter(),
            next_counts=next_counts if next_counts is not None else defaultdict(Counter),
        )

    @classmethod
    def from_sequence(  # type: ignore[override]
        cls,
        sequence: Sequence[Any],
        *,
        alphabet: DyckAlphabet,
        max_length: int | None = None,
        max_stack_depth: int = 8,
    ) -> StackSuffixCounts:
        seq = tuple(sequence)
        if not seq:
            raise ValueError("sequence must be non-empty")
        stacks: list[tuple[Any, ...]] = []
        stack: tuple[Any, ...] = ()
        for symbol in seq:
            if symbol not in alphabet.symbol_alphabet:
                raise ValueError(f"symbol {symbol!r} not in Dyck alphabet")
            stacks.append(stack)
            stack = _push(stack, symbol, alphabet=alphabet, max_stack_depth=max_stack_depth)
        counts = cls(alphabet=infer_alphabet(alphabet.symbol_alphabet, None))
        for t, suffix in iter_suffixes(seq, max_length if max_length is not None else len(seq)):
            history = (suffix, stacks[t])
            counts.history_counts[history] += 1
            counts.next_counts[history][seq[t]] += 1
        return counts


def _push(stack: tuple[Any, ...], symbol: Any, *, alphabet: DyckAlphabet, max_stack_depth: int) -> tuple[Any, ...]:
    """The stack after ``symbol``: calls push (dropping the bottom past ``max_stack_depth``), returns pop."""
    if symbol in alphabet.call_alphabet:
        return (*(stack[1:] if len(stack) >= max_stack_depth else stack), symbol)
    if symbol in alphabet.return_alphabet:
        return stack[:-1]
    return stack
