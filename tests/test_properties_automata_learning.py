"""Property-based and metamorphic tests for automaton learners and observation tables."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable
from itertools import combinations
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from sofic.automata import (
    DFA,
    AutomatonEquivalenceOracle,
    CanonicalRFSA,
    LanguageMembershipOracle,
    ObservationTable,
    equivalent,
    learn_dfa_edsm,
    learn_dfa_lstar,
    learn_dfa_rpni,
    learn_dfa_ttt,
    learn_rfsa_nlstar,
)
from sofic.testing import dfas
from tests import oracles
from tests.test_properties_automata import relabel

AB = ("0", "1")
PRESERVING = {"0": "a", "1": "b"}
REVERSING = {"0": "b", "1": "a"}

Word = tuple[Any, ...]


def lang(aut: Any, alphabet: tuple[Any, ...] = AB, n: int = 5) -> frozenset[Word]:
    return frozenset(w for w in oracles.words(alphabet, n) if aut.recognizes(w))


def mapped(words: frozenset[Word], symbol_map: dict[str, str]) -> frozenset[Word]:
    return frozenset(tuple(symbol_map[c] for c in w) for w in words)


def size(aut: Any) -> int:
    return len(list(aut.states()))


def nerode(dfa: DFA) -> int:
    return oracles.nerode_class_count(lambda w: oracles.nfa_accepts(dfa, w), AB, 3)


def active(learner: Callable[..., Any], target: Any, alphabet: tuple[str, ...]) -> Any:
    return learner(alphabet, LanguageMembershipOracle(target), AutomatonEquivalenceOracle(target, alphabet))


# --------------------------------------------------------------------------- active learners


@given(dfas(), st.sampled_from([learn_dfa_lstar, learn_dfa_ttt]))
def test_active_dfa_learners_find_the_minimal_dfa(target: DFA, learner: Callable[..., DFA]) -> None:
    learned = active(learner, target, AB)
    assert equivalent(learned, target)
    assert size(learned) <= nerode(target)


@given(dfas(), st.sampled_from([learn_dfa_lstar, learn_dfa_ttt]), st.sampled_from([PRESERVING, REVERSING]))
def test_active_dfa_learners_are_relabeling_invariant(
    target: DFA, learner: Callable[..., DFA], symbol_map: dict[str, str]
) -> None:
    renamed = relabel(target, lambda s: ("r", s), symbol_map.__getitem__)
    original = active(learner, target, AB)
    learned = active(learner, renamed, ("a", "b"))
    assert size(learned) == size(original)
    assert lang(learned, ("a", "b")) == mapped(lang(original), symbol_map)


@given(dfas(), st.sampled_from([PRESERVING, REVERSING]))
def test_nlstar_learns_canonical_rfsa_invariantly(target: DFA, symbol_map: dict[str, str]) -> None:
    learned = active(learn_rfsa_nlstar, target, AB)
    assert lang(learned) == lang(target)
    assert size(learned) == size(CanonicalRFSA.from_language(target))
    renamed = active(learn_rfsa_nlstar, relabel(target, lambda s: ("r", s), symbol_map.__getitem__), ("a", "b"))
    assert size(renamed) == size(learned)
    assert lang(renamed, ("a", "b")) == mapped(lang(learned), symbol_map)


# --------------------------------------------------------------------------- passive learners


@given(
    dfas(),
    st.sets(st.sampled_from(oracles.words(AB, 3)), min_size=1),
    st.sampled_from([learn_dfa_rpni, learn_dfa_edsm]),
)
def test_passive_learners_are_consistent_and_relabeling_invariant(
    target: DFA, sample: set[Word], learner: Callable[..., DFA]
) -> None:
    ordered = sorted(sample, key=lambda w: (len(w), w))
    positive = [w for w in ordered if target.recognizes(w)]
    negative = [w for w in ordered if not target.recognizes(w)]
    learned = learner(positive, negative)
    assert all(learned.recognizes(w) for w in positive)
    assert not any(learned.recognizes(w) for w in negative)

    def rename(words: list[Word]) -> list[Word]:
        return [tuple(PRESERVING[c] for c in w) for w in words]

    renamed = learner(rename(positive), rename(negative))
    assert size(renamed) == size(learned)
    assert lang(renamed, ("a", "b"), 4) == mapped(lang(learned, AB, 4), PRESERVING)


# --------------------------------------------------------------------------- observation tables


def union_witness(target: DFA, state: Hashable, others: frozenset[Hashable]) -> Word | None:
    """Shortest word in the right language of ``state`` but of none of ``others`` (complete DFA)."""
    start = (state, others)
    queue, seen = deque([(start, ())]), {start}
    while queue:
        (p, qs), w = queue.popleft()
        if p in target.accepting_states and not qs & target.accepting_states:
            return w
        for a in AB:
            nxt = (next(iter(target.delta(p, a))), frozenset(next(iter(target.delta(q, a))) for q in qs))
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, (*w, a)))
    return None


def full_table(target: DFA, n: int = 3) -> ObservationTable:
    # Extraction reads transitions u -> ua only among access words, so access must hold every word < n and
    # its one-symbol extensions. Experiments must witness every residual non-inclusion L_p not in the union
    # of L_q (q in Q), or the table is closed and consistent but not RFSA-closed and RFSA-consistent.
    access = frozenset(oracles.words(AB, n))
    states = list(target.states())
    witnesses = {
        union_witness(target, p, frozenset(others))
        for p in states
        for k in range(len(states))
        for others in combinations([q for q in states if q != p], k)
    }
    experiments = frozenset(oracles.words(AB, 2)) | {w[i:] for w in witnesses if w is not None for i in range(len(w))}
    membership = {
        u + a + e: oracles.nfa_accepts(target, u + a + e)
        for u in access
        for a in ((), *((s,) for s in AB))
        for e in experiments
    }
    return ObservationTable(access_words=access, experiments=experiments, membership=membership)


@given(dfas())
def test_observation_table_extractions(target: DFA) -> None:
    table = full_table(target)
    dfa = table.to_minimal_dfa()
    assert lang(dfa) == lang(target)
    assert size(dfa) == nerode(target)
    rfsa = table.to_canonical_rfsa()
    assert lang(rfsa) == lang(target)
    assert size(rfsa) == size(CanonicalRFSA.from_language(target))
    assert lang(table.to_atomaton()) == lang(target)
