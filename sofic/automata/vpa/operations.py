r"""Concrete constructions and decision procedures for visibly pushdown automata.

Every construction works on a *normalized* form of a VPA: the bottom of the
stack is an explicit symbol ``⊥``, and wildcard returns are expanded into
guarded returns. A return guarded by ``⊥`` is a *pending* return: it fires on
the empty stack and leaves it empty. A VPA without ``⊥``-guarded returns rejects
pending returns. Acceptance is by final state, whatever the stack holds, so
pending calls are allowed :cite:`AlurMadhusudan2009`.

* :func:`determinize` is the summary construction of Alur and Madhusudan: a
  deterministic state is a pair ``(S, R)`` where ``S`` relates the state after
  the last pending call to the current state along well-matched factors, and
  ``R`` is the set of current states. The result is complete, so
  :func:`complement` flips its accepting states.
* :func:`concat` and :func:`kleene_star` track, in the finite control, whether
  the current factor's stack is empty, and push that bit with every symbol. A
  return seen while the bit is set is a pending return of the current factor,
  whatever lies below on the physical stack.
* :func:`is_empty` saturates the relation of well-matched summaries and then
  explores states reachable with pending returns (stack empty) or pending calls.
* :func:`to_single_entry` and :func:`to_multiple_entry` convert a VPA of a
  well-matched language into a modular form following the summary construction
  of :cite:`AlurKumarMadhusudanViswanathan2005`: on a call the state resets to
  the entry of the call's module and the caller is pushed.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sofic.exceptions import NonWellMatchedLanguageError
from sofic.graph import ATTR_KIND, ATTR_STACK_SYMBOL, ATTR_SYMBOL, KIND_CALL, KIND_INTERNAL, KIND_RETURN

if TYPE_CHECKING:
    from sofic.automata.vpa.base import VisiblyPushdownAutomaton
    from sofic.automata.vpa.deterministic import DeterministicVisiblyPushdownAutomaton
    from sofic.automata.vpa.modular import MultipleEntryVisiblyPushdownAutomaton, SingleEntryVisiblyPushdownAutomaton

#: Bottom-of-stack symbol used by constructed VPAs.
BOTTOM = "⊥"

Word = tuple[Any, ...]


@dataclass
class NormalVPA:
    """VPA with an explicit bottom symbol and only guarded returns."""

    call_alphabet: frozenset[Any]
    return_alphabet: frozenset[Any]
    internal_alphabet: frozenset[Any]
    states: set[Hashable] = field(default_factory=set)
    initial: set[Hashable] = field(default_factory=set)
    accepting: set[Hashable] = field(default_factory=set)
    calls: set[tuple[Hashable, Any, Hashable, Any]] = field(default_factory=set)
    internals: set[tuple[Hashable, Any, Hashable]] = field(default_factory=set)
    returns: set[tuple[Hashable, Any, Any, Hashable]] = field(default_factory=set)

    @property
    def stack_symbols(self) -> set[Any]:
        pushed = {g for _q, _a, _t, g in self.calls}
        guarded = {g for _q, _a, g, _t in self.returns if g != BOTTOM}
        return pushed | guarded

    def call_map(self) -> dict[tuple[Hashable, Any], set[tuple[Hashable, Any]]]:
        result: dict[tuple[Hashable, Any], set[tuple[Hashable, Any]]] = defaultdict(set)
        for q, a, t, g in self.calls:
            result[(q, a)].add((t, g))
        return result

    def internal_map(self) -> dict[tuple[Hashable, Any], set[Hashable]]:
        result: dict[tuple[Hashable, Any], set[Hashable]] = defaultdict(set)
        for q, a, t in self.internals:
            result[(q, a)].add(t)
        return result

    def return_map(self) -> dict[tuple[Hashable, Any], set[tuple[Any, Hashable]]]:
        result: dict[tuple[Hashable, Any], set[tuple[Any, Hashable]]] = defaultdict(set)
        for q, a, g, t in self.returns:
            result[(q, a)].add((g, t))
        return result

    def copy_outgoing(self, source: Hashable, new_source: Hashable, map_target=lambda t: t) -> None:
        """Give ``new_source`` copies of ``source``'s outgoing transitions."""
        self.calls |= {(new_source, a, map_target(t), g) for q, a, t, g in list(self.calls) if q == source}
        self.internals |= {(new_source, a, map_target(t)) for q, a, t in list(self.internals) if q == source}
        self.returns |= {(new_source, a, g, map_target(t)) for q, a, g, t in list(self.returns) if q == source}


def normalize(vpa: VisiblyPushdownAutomaton) -> NormalVPA:
    """Return the normalized form of ``vpa``.

    A wildcard return (no stack symbol) fires on every stack symbol, and also on
    the empty stack when ``vpa`` has a bottom symbol; a return guarded by the
    bottom symbol fires only on the empty stack. A call without a stack symbol
    never fires and is dropped.
    """
    bottom = vpa.bottom_stack_symbol
    gamma = {g for g in vpa.stack_alphabet if g != bottom}
    for transition in vpa.transitions():
        if transition.data.get(ATTR_KIND) == KIND_CALL and transition.data.get(ATTR_STACK_SYMBOL) is not None:
            gamma.add(transition.data.get(ATTR_STACK_SYMBOL))
    result = NormalVPA(
        call_alphabet=frozenset(vpa.call_alphabet),
        return_alphabet=frozenset(vpa.return_alphabet),
        internal_alphabet=frozenset(vpa.internal_alphabet),
        states=set(vpa.states()),
        initial=set() if vpa.initial_state is None else {vpa.initial_state},
        accepting=set(vpa.accepting_states),
    )
    for transition in vpa.transitions():
        data = transition.data
        kind, symbol = data.get(ATTR_KIND), data.get(ATTR_SYMBOL)
        if symbol is None:
            continue
        source, target = transition.source, transition.target
        if kind == KIND_CALL:
            if data.get(ATTR_STACK_SYMBOL) is not None:
                result.calls.add((source, symbol, target, data.get(ATTR_STACK_SYMBOL)))
        elif kind == KIND_INTERNAL:
            result.internals.add((source, symbol, target))
        elif kind == KIND_RETURN:
            guard = data.get(ATTR_STACK_SYMBOL)
            if guard is None:
                guards = set(gamma) | ({BOTTOM} if bottom is not None else set())
            elif bottom is not None and guard == bottom:
                guards = {BOTTOM}
            else:
                guards = {guard}
            result.returns |= {(source, symbol, g, target) for g in guards}
    return result


def _merge_alphabets(*machines: NormalVPA) -> tuple[frozenset[Any], frozenset[Any], frozenset[Any]]:
    calls = frozenset().union(*(m.call_alphabet for m in machines))
    returns = frozenset().union(*(m.return_alphabet for m in machines))
    internals = frozenset().union(*(m.internal_alphabet for m in machines))
    if len(calls) + len(returns) + len(internals) != len(calls | returns | internals):
        raise ValueError("operands disagree on which symbols are calls, returns, and internals")
    return calls, returns, internals


def _with_alphabets(machine: NormalVPA, calls, returns, internals) -> NormalVPA:
    machine.call_alphabet, machine.return_alphabet, machine.internal_alphabet = calls, returns, internals
    return machine


def _forward_trim(machine: NormalVPA) -> NormalVPA:
    """Drop states unreachable from the initial states (ignoring the stack)."""
    successors: dict[Hashable, set[Hashable]] = defaultdict(set)
    for q, _a, t, _g in machine.calls:
        successors[q].add(t)
    for q, _a, t in machine.internals:
        successors[q].add(t)
    for q, _a, _g, t in machine.returns:
        successors[q].add(t)
    keep = set(machine.initial)
    stack = list(keep)
    while stack:
        for target in successors[stack.pop()]:
            if target not in keep:
                keep.add(target)
                stack.append(target)
    machine.states &= keep
    machine.accepting &= keep
    machine.calls = {c for c in machine.calls if c[0] in keep}
    machine.internals = {i for i in machine.internals if i[0] in keep}
    machine.returns = {r for r in machine.returns if r[0] in keep}
    return machine


def denormalize(machine: NormalVPA, cls: type | None = None, **extra: Any) -> Any:
    """Build a VPA of class ``cls`` from a normalized form, relabeling states and stack symbols as integers."""
    from sofic.automata.vpa.base import VisiblyPushdownAutomaton

    cls = cls or VisiblyPushdownAutomaton
    machine = _forward_trim(machine)
    if len(machine.initial) > 1:
        start = ("start", len(machine.states))
        machine.states.add(start)
        for state in sorted(machine.initial, key=repr):
            machine.copy_outgoing(state, start)
        if machine.initial & machine.accepting:
            machine.accepting.add(start)
        machine.initial = {start}

    state_name = {state: index for index, state in enumerate(sorted(machine.states, key=repr))}
    stack_name = {symbol: index for index, symbol in enumerate(sorted(machine.stack_symbols, key=repr))}
    has_bottom = any(g == BOTTOM for _q, _a, g, _t in machine.returns)
    stack_alphabet = set(stack_name.values()) | ({BOTTOM} if has_bottom else set())
    result = cls(
        call_alphabet=machine.call_alphabet,
        return_alphabet=machine.return_alphabet,
        internal_alphabet=machine.internal_alphabet,
        stack_alphabet=frozenset(stack_alphabet),
        bottom_stack_symbol=BOTTOM if has_bottom else None,
        initial_state=state_name[next(iter(machine.initial))] if machine.initial else None,
        accepting_states=frozenset(state_name[q] for q in machine.accepting),
        **extra,
    )
    for state in state_name.values():
        result.graph.add_state(state)
    for q, a, t, g in sorted(machine.calls, key=repr):
        result.graph.add_transition(
            state_name[q], state_name[t], **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: a, ATTR_STACK_SYMBOL: stack_name[g]}
        )
    for q, a, t in sorted(machine.internals, key=repr):
        result.graph.add_transition(state_name[q], state_name[t], **{ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: a})
    for q, a, g, t in sorted(machine.returns, key=repr):
        guard = BOTTOM if g == BOTTOM else stack_name[g]
        result.graph.add_transition(
            state_name[q], state_name[t], **{ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: a, ATTR_STACK_SYMBOL: guard}
        )
    return result


# --------------------------------------------------------------------- boolean operations


def union(left: NormalVPA, right: NormalVPA) -> NormalVPA:
    """Disjoint sum of two normalized VPAs."""
    alphabets = _merge_alphabets(left, right)
    result = NormalVPA(*alphabets)
    for tag, machine in ((0, left), (1, right)):

        def lift(g: Any, tag: int = tag) -> Any:
            return BOTTOM if g == BOTTOM else (tag, g)

        result.states |= {(tag, q) for q in machine.states}
        result.initial |= {(tag, q) for q in machine.initial}
        result.accepting |= {(tag, q) for q in machine.accepting}
        result.calls |= {((tag, q), a, (tag, t), (tag, g)) for q, a, t, g in machine.calls}
        result.internals |= {((tag, q), a, (tag, t)) for q, a, t in machine.internals}
        result.returns |= {((tag, q), a, lift(g), (tag, t)) for q, a, g, t in machine.returns}
    return result


def intersection(left: NormalVPA, right: NormalVPA) -> NormalVPA:
    """Synchronized product of two normalized VPAs (reachable part)."""
    alphabets = _merge_alphabets(left, right)
    result = NormalVPA(*alphabets)
    lc, li, lr = left.call_map(), left.internal_map(), left.return_map()
    rc, ri, rr = right.call_map(), right.internal_map(), right.return_map()
    frontier = [(p, q) for p in left.initial for q in right.initial]
    result.initial = set(frontier)
    result.states = set(frontier)
    while frontier:
        state = frontier.pop()
        p, q = state
        if p in left.accepting and q in right.accepting:
            result.accepting.add(state)
        successors = []
        for a in alphabets[2]:
            for t1 in li.get((p, a), ()):
                for t2 in ri.get((q, a), ()):
                    result.internals.add((state, a, (t1, t2)))
                    successors.append((t1, t2))
        for a in alphabets[0]:
            for t1, g1 in lc.get((p, a), ()):
                for t2, g2 in rc.get((q, a), ()):
                    result.calls.add((state, a, (t1, t2), (g1, g2)))
                    successors.append((t1, t2))
        for a in alphabets[1]:
            for g1, t1 in lr.get((p, a), ()):
                for g2, t2 in rr.get((q, a), ()):
                    if (g1 == BOTTOM) != (g2 == BOTTOM):
                        continue
                    guard = BOTTOM if g1 == BOTTOM else (g1, g2)
                    result.returns.add((state, a, guard, (t1, t2)))
                    successors.append((t1, t2))
        for successor in successors:
            if successor not in result.states:
                result.states.add(successor)
                frontier.append(successor)
    return result


def determinize(machine: NormalVPA, alphabets: tuple[frozenset, frozenset, frozenset] | None = None) -> NormalVPA:
    """Alur-Madhusudan summary construction; the result is deterministic and complete."""
    calls_a, returns_a, internals_a = alphabets or (
        machine.call_alphabet,
        machine.return_alphabet,
        machine.internal_alphabet,
    )
    cmap, imap, rmap = machine.call_map(), machine.internal_map(), machine.return_map()
    identity = frozenset((q, q) for q in machine.states)
    start = (identity, frozenset(machine.initial))
    result = NormalVPA(calls_a, returns_a, internals_a, states={start}, initial={start})

    def after_internal(pairs, a):
        return frozenset((p, t) for p, q in pairs for t in imap.get((q, a), ()))

    def after_pending_return(pairs, a):
        return frozenset((p, t) for p, q in pairs for g, t in rmap.get((q, a), ()) if g == BOTTOM)

    def after_matched_return(caller_pairs, c, pairs, a):
        inner: dict[Hashable, set[Hashable]] = defaultdict(set)
        for q2, q3 in pairs:
            inner[q2].add(q3)
        out = set()
        for p, q1 in caller_pairs:
            for q2, g in cmap.get((q1, c), ()):
                for q3 in inner.get(q2, ()):
                    out |= {(p, t) for guard, t in rmap.get((q3, a), ()) if guard == g}
        return frozenset(out)

    stack_symbols: list[tuple[frozenset, frozenset, Any]] = []
    known_stack: set[tuple[frozenset, frozenset, Any]] = set()
    done: set[tuple[Any, Any]] = set()
    pending = [start]
    while True:
        while pending:
            state = pending.pop()
            pairs, current = state
            if current & machine.accepting:
                result.accepting.add(state)
            successors = []
            for a in internals_a:
                target = (after_internal(pairs, a), frozenset(t for q in current for t in imap.get((q, a), ())))
                successors.append(target)
                result.internals.add((state, a, target))
            for c in calls_a:
                entered = frozenset(t for q in current for t, _g in cmap.get((q, c), ()))
                target = (identity, entered)
                symbol = (pairs, current, c)
                if symbol not in known_stack:
                    known_stack.add(symbol)
                    stack_symbols.append(symbol)
                successors.append(target)
                result.calls.add((state, c, target, symbol))
            for a in returns_a:
                moved = after_pending_return(pairs, a)
                current_after = frozenset(t for q in current for g, t in rmap.get((q, a), ()) if g == BOTTOM)
                target = (moved, current_after)
                successors.append(target)
                result.returns.add((state, a, BOTTOM, target))
            for successor in successors:
                if successor not in result.states:
                    result.states.add(successor)
                    pending.append(successor)
        new_work = False
        for state in list(result.states):
            pairs, _current = state
            for symbol in list(stack_symbols):
                if (state, symbol) in done:
                    continue
                done.add((state, symbol))
                caller_pairs, caller_current, c = symbol
                for a in returns_a:
                    moved = after_matched_return(caller_pairs, c, pairs, a)
                    reached = after_matched_return(frozenset((q, q) for q in caller_current), c, pairs, a)
                    target = (moved, frozenset(t for _p, t in reached))
                    result.returns.add((state, a, symbol, target))
                    if target not in result.states:
                        result.states.add(target)
                        pending.append(target)
                        new_work = True
        if not new_work and not pending:
            return result


def complement(machine: NormalVPA, alphabets=None) -> NormalVPA:
    """Complement over the visible alphabet (or ``alphabets``)."""
    deterministic = determinize(machine, alphabets)
    deterministic.accepting = deterministic.states - deterministic.accepting
    return deterministic


def difference(left: NormalVPA, right: NormalVPA) -> NormalVPA:
    alphabets = _merge_alphabets(left, right)
    return intersection(_with_alphabets(left, *alphabets), complement(right, alphabets))


# ----------------------------------------------------------------- structural operations


def concat(left: NormalVPA, right: NormalVPA) -> NormalVPA:
    """Concatenation; the right factor starts with an empty stack of its own."""
    alphabets = _merge_alphabets(left, right)
    result = NormalVPA(*alphabets)
    left_tops = [("A", g) for g in left.stack_symbols] + [BOTTOM]

    for q in left.states:
        result.states.add(("A", q))
    result.initial = {("A", q) for q in left.initial}
    result.calls |= {(("A", q), a, ("A", t), ("A", g)) for q, a, t, g in left.calls}
    result.internals |= {(("A", q), a, ("A", t)) for q, a, t in left.internals}
    result.returns |= {(("A", q), a, g if g == BOTTOM else ("A", g), ("A", t)) for q, a, g, t in left.returns}

    for q in right.states:
        for empty in (True, False):
            state = ("B", q, empty)
            result.states.add(state)
            result.internals |= {(state, a, ("B", t, empty)) for p, a, t in right.internals if p == q}
            result.calls |= {(state, a, ("B", t, False), ("B", g, empty)) for p, a, t, g in right.calls if p == q}
            for p, a, g, t in right.returns:
                if p != q:
                    continue
                if empty and g == BOTTOM:
                    result.returns |= {(state, a, top, ("B", t, True)) for top in left_tops}
                elif not empty and g != BOTTOM:
                    result.returns |= {(state, a, ("B", g, bit), ("B", t, bit)) for bit in (True, False)}
            if q in right.accepting:
                result.accepting.add(state)

    epsilon_in_right = bool(right.initial & right.accepting)
    for q in left.accepting:
        for start in right.initial:
            result.copy_outgoing(("B", start, True), ("A", q))
        if epsilon_in_right:
            result.accepting.add(("A", q))
    return result


def kleene_star(machine: NormalVPA) -> NormalVPA:
    """Kleene star; each factor starts with an empty stack of its own."""
    result = NormalVPA(machine.call_alphabet, machine.return_alphabet, machine.internal_alphabet)
    tops = [(g, bit) for g in machine.stack_symbols for bit in (True, False)] + [BOTTOM]
    for q in machine.states:
        for empty in (True, False):
            state = (q, empty)
            result.states.add(state)
            result.internals |= {(state, a, (t, empty)) for p, a, t in machine.internals if p == q}
            result.calls |= {(state, a, (t, False), (g, empty)) for p, a, t, g in machine.calls if p == q}
            for p, a, g, t in machine.returns:
                if p != q:
                    continue
                if empty and g == BOTTOM:
                    result.returns |= {(state, a, top, (t, True)) for top in tops}
                elif not empty and g != BOTTOM:
                    result.returns |= {(state, a, (g, bit), (t, bit)) for bit in (True, False)}
            if q in machine.accepting:
                result.accepting.add(state)
    start = ("star-start",)
    result.states.add(start)
    result.initial = {start}
    result.accepting.add(start)
    restart_sources = [start] + [(q, empty) for q in machine.accepting for empty in (True, False)]
    for source in restart_sources:
        for initial in machine.initial:
            result.copy_outgoing((initial, True), source)
    return result


# --------------------------------------------------------------------- decision procedures


def well_matched_summaries(machine: NormalVPA) -> dict[tuple[Hashable, Hashable], Word]:
    """Pairs ``(p, q)`` joined by a well-matched word, each with a witness word."""
    witness: dict[tuple[Hashable, Hashable], Word] = {(q, q): () for q in machine.states}
    rmap = machine.return_map()
    changed = True
    while changed:
        changed = False
        updates: dict[tuple[Hashable, Hashable], Word] = {}
        by_source: dict[Hashable, list[tuple[Hashable, Word]]] = defaultdict(list)
        for (p, q), word in witness.items():
            by_source[p].append((q, word))
        for (p, q), word in witness.items():
            for source, a, target in machine.internals:
                if source == q:
                    updates.setdefault((p, target), (*word, a))
            for r, word2 in by_source.get(q, ()):
                updates.setdefault((p, r), word + word2)
        for p0, c, p, g in machine.calls:
            for q, word in by_source.get(p, ()):
                for a in machine.return_alphabet:
                    for guard, target in rmap.get((q, a), ()):
                        if guard == g:
                            updates.setdefault((p0, target), (c, *word, a))
        for pair, word in updates.items():
            if pair not in witness or len(word) < len(witness[pair]):
                if pair not in witness:
                    changed = True
                witness[pair] = word
    return witness


def accepted_word(machine: NormalVPA) -> Word | None:
    """Return a short accepted word, or ``None`` when the language is empty."""
    summaries = well_matched_summaries(machine)
    by_source: dict[Hashable, list[tuple[Hashable, Word]]] = defaultdict(list)
    for (p, q), word in summaries.items():
        if word:
            by_source[p].append((q, word))
    queue: list[tuple[int, int, Hashable, bool, Word]] = []
    counter = 0
    for q in machine.initial:
        heapq.heappush(queue, (0, counter, q, True, ()))
        counter += 1
    seen: set[tuple[Hashable, bool]] = set()
    while queue:
        _length, _tie, state, stack_empty, word = heapq.heappop(queue)
        if (state, stack_empty) in seen:
            continue
        seen.add((state, stack_empty))
        if state in machine.accepting:
            return word
        moves: list[tuple[Hashable, bool, Word]] = [(t, stack_empty, word + w) for t, w in by_source.get(state, ())]
        moves += [(t, False, (*word, a)) for q, a, t, _g in machine.calls if q == state]
        if stack_empty:
            moves += [(t, True, (*word, a)) for q, a, g, t in machine.returns if q == state and g == BOTTOM]
        for target, empty, extended in moves:
            if (target, empty) not in seen:
                heapq.heappush(queue, (len(extended), counter, target, empty, extended))
                counter += 1
    return None


def is_empty(machine: NormalVPA) -> bool:
    return accepted_word(machine) is None


def _unmatched_tracker(alphabets: tuple[frozenset, frozenset, frozenset]) -> NormalVPA:
    """Deterministic VPA accepting the words with a pending call or a pending return."""
    tracker = NormalVPA(*alphabets)
    for pending_return in (False, True):
        for empty in (True, False):
            state = (pending_return, empty)
            tracker.states.add(state)
            if pending_return or not empty:
                tracker.accepting.add(state)
            tracker.internals |= {(state, a, state) for a in alphabets[2]}
            tracker.calls |= {(state, c, (pending_return, False), empty) for c in alphabets[0]}
            for a in alphabets[1]:
                if empty:
                    tracker.returns.add((state, a, BOTTOM, (True, True)))
                else:
                    tracker.returns |= {(state, a, bit, (pending_return, bit)) for bit in (True, False)}
    tracker.initial = {(False, True)}
    return tracker


def has_unmatched_word(machine: NormalVPA) -> bool:
    """Whether the language contains a word with a pending call or a pending return."""
    alphabets = (machine.call_alphabet, machine.return_alphabet, machine.internal_alphabet)
    return not is_empty(intersection(machine, _unmatched_tracker(alphabets)))


# ------------------------------------------------------------------------ modular forms


def _modular_conversion(
    vpa: VisiblyPushdownAutomaton,
    call_partition: Mapping[Any, Hashable] | None,
    *,
    multiple_entry: bool,
) -> Any:
    from sofic.automata.vpa.modular import MultipleEntryVisiblyPushdownAutomaton, SingleEntryVisiblyPushdownAutomaton

    machine = normalize(vpa)
    if has_unmatched_word(machine):
        raise NonWellMatchedLanguageError(
            "modular (single- or multiple-entry) forms are built here for well-matched languages only"
        )
    base = "base"
    partition = dict(call_partition) if call_partition is not None else {c: ("module", c) for c in vpa.call_alphabet}
    if base in partition.values():
        raise ValueError("calls cannot target the base module")
    missing = set(vpa.call_alphabet) - set(partition)
    if missing:
        raise ValueError(f"call_partition is missing calls {sorted(missing, key=repr)!r}")

    cmap, imap, rmap = machine.call_map(), machine.internal_map(), machine.return_map()
    identity = frozenset((q, q) for q in machine.states)

    def compose_internal(pairs, a):
        return frozenset((p, t) for p, q in pairs for t in imap.get((q, a), ()))

    def wrap(caller_pairs, c, pairs, a):
        inner: dict[Hashable, set[Hashable]] = defaultdict(set)
        for q2, q3 in pairs:
            inner[q2].add(q3)
        out = set()
        for p, q1 in caller_pairs:
            for q2, g in cmap.get((q1, c), ()):
                for q3 in inner.get(q2, ()):
                    out |= {(p, t) for guard, t in rmap.get((q3, a), ()) if guard == g}
        return frozenset(out)

    # A raw state is (module, entering_call, summary); entering_call is only kept
    # for the multiple-entry form, where it replaces the call symbol on the stack.
    def entry(c):
        return (partition[c], c if multiple_entry else None, identity)

    start = (base, None, identity)
    label: dict[tuple, int] = {start: 0}
    order = [start]
    internals: set[tuple[int, Any, int]] = set()
    calls: set[tuple[int, Any, int, Any]] = set()
    returns: set[tuple[int, Any, Any, int]] = set()
    pushes: list[tuple[int, Any]] = []
    done: set[tuple[int, Any]] = set()

    def name(raw):
        if raw not in label:
            label[raw] = len(order)
            order.append(raw)
        return label[raw]

    index = 0
    while True:
        while index < len(order):
            raw = order[index]
            module, entered, pairs = raw
            source = label[raw]
            for a in vpa.internal_alphabet:
                internals.add((source, a, name((module, entered, compose_internal(pairs, a)))))
            for c in vpa.call_alphabet:
                push = source if multiple_entry else (source, c)
                calls.add((source, c, name(entry(c)), push))
                if (source, c) not in pushes:
                    pushes.append((source, c))
            index += 1
        grew = False
        for raw in list(order):
            module, entered, pairs = raw
            source = label[raw]
            for caller, c in list(pushes):
                if (source, (caller, c)) in done:
                    continue
                if multiple_entry and entered != c:
                    continue
                if not multiple_entry and module != partition[c]:
                    continue
                done.add((source, (caller, c)))
                caller_module, caller_entered, caller_pairs = order[caller]
                guard = caller if multiple_entry else (caller, c)
                for a in vpa.return_alphabet:
                    before = len(order)
                    target = name((caller_module, caller_entered, wrap(caller_pairs, c, pairs, a)))
                    returns.add((source, a, guard, target))
                    grew = grew or len(order) > before
        if not grew and index >= len(order):
            break

    modules: dict[Hashable, set[int]] = defaultdict(set)
    for raw, state in label.items():
        modules[raw[0]].add(state)
    accepting = {
        label[raw]
        for raw in order
        if raw[0] == base and any(p in machine.initial and q in machine.accepting for p, q in raw[2])
    }
    used_partition = {c: m for c, m in partition.items() if c in vpa.call_alphabet}
    call_entries = {c: label[entry(c)] for c in vpa.call_alphabet}
    common = {
        "call_alphabet": vpa.call_alphabet,
        "return_alphabet": vpa.return_alphabet,
        "internal_alphabet": vpa.internal_alphabet,
        "stack_alphabet": frozenset(push for _s, _c, _t, push in calls),
        "initial_state": 0,
        "accepting_states": frozenset(accepting),
        "modules": {m: frozenset(states) for m, states in modules.items()},
        "base_module": base,
        "call_partition": used_partition,
        "call_entries": call_entries,
    }
    if multiple_entry:
        entries: dict[Hashable, set[int]] = defaultdict(set)
        entries[base].add(0)
        for c, state in call_entries.items():
            entries[partition[c]].add(state)
        result = MultipleEntryVisiblyPushdownAutomaton(
            entry_states={m: frozenset(s) for m, s in entries.items()}, **common
        )
    else:
        result = SingleEntryVisiblyPushdownAutomaton(
            entry_states={partition[c]: state for c, state in call_entries.items()}, **common
        )
    for state in range(len(order)):
        result.graph.add_state(state)
    for s, c, t, push in sorted(calls, key=repr):
        result.add_call_transition(s, t, c, push)
    for s, a, t in sorted(internals, key=repr):
        result.add_internal_transition(s, t, a)
    for s, a, guard, t in sorted(returns, key=repr):
        result.add_return_transition(s, t, a, guard)
    result.validate()
    return result


def to_single_entry(
    vpa: VisiblyPushdownAutomaton, call_partition: Mapping[Any, Hashable] | None = None
) -> SingleEntryVisiblyPushdownAutomaton:
    """Convert a VPA of a well-matched language into a single-entry modular VPA.

    ``call_partition`` maps each call symbol to a module (default: one module
    per call symbol). Raises
    :class:`~sofic.exceptions.NonWellMatchedLanguageError` when the language has
    a word with a pending call or return, which a single-entry automaton that
    forgets its caller cannot accept.
    """
    return _modular_conversion(vpa, call_partition, multiple_entry=False)


def to_multiple_entry(
    vpa: VisiblyPushdownAutomaton, call_partition: Mapping[Any, Hashable] | None = None
) -> MultipleEntryVisiblyPushdownAutomaton:
    """Convert a VPA of a well-matched language into a multiple-entry modular VPA.

    Each module has one entry per call symbol assigned to it, and calls push
    only the caller state.
    """
    return _modular_conversion(vpa, call_partition, multiple_entry=True)


def determinize_vpa(vpa: VisiblyPushdownAutomaton) -> DeterministicVisiblyPushdownAutomaton:
    from sofic.automata.vpa.deterministic import DeterministicVisiblyPushdownAutomaton

    return denormalize(determinize(normalize(vpa)), DeterministicVisiblyPushdownAutomaton)
