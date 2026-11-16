"""Incomplete accessible DFA strings (:mod:`sofic.automata.enumeration.idfa`)."""

from __future__ import annotations

import pytest

from sofic.automata.enumeration import idfa
from sofic.automata.enumeration.idfa import MISSING_TRANSITION as MISSING
from sofic.automata.enumeration.idfa import IDFAEnumerationError


@pytest.mark.parametrize(("n", "k"), [(1, 2), (2, 2), (3, 2), (2, 3)])
def test_enumeration_is_counted_ranked_and_bounded(n, k):
    strings = list(idfa.iter_idfa_strings(k, n))
    assert len(strings) == len(set(strings)) == idfa.count_accessible_idfa(k, n)
    assert strings[0] == idfa.first_idfa_string(n=n, k=k)
    assert strings[-1] == idfa.last_idfa_string(n=n, k=k)
    for rank in (0, len(strings) // 2, len(strings) - 1):
        assert idfa.unrank_idfa_string(rank, n=n, k=k) == strings[rank]
        assert idfa.rank_idfa_string(strings[rank], n=n, k=k) == rank


def test_unrank_rejects_out_of_range_ranks():
    with pytest.raises(IDFAEnumerationError, match="nonnegative"):
        idfa.unrank_idfa_string(-1, n=2, k=2)
    with pytest.raises(IDFAEnumerationError, match="out of range"):
        idfa.unrank_idfa_string(4, n=1, k=2)
    with pytest.raises(IDFAEnumerationError, match="out of range"):
        idfa.unrank_idfa_string(idfa.count_accessible_idfa(2, 2), n=2, k=2)


def test_validation_rules():
    with pytest.raises(IDFAEnumerationError, match="expected length"):
        idfa.validate_idfa_string((0,), n=2, k=2)
    with pytest.raises(IDFAEnumerationError, match="not in"):
        idfa.validate_idfa_string((1, 5, MISSING, MISSING), n=2, k=2)
    with pytest.raises(IDFAEnumerationError, match="appears before"):
        idfa.validate_idfa_string((2, 1, MISSING, MISSING, MISSING, MISSING), n=3, k=2)
    with pytest.raises(IDFAEnumerationError, match="does not appear"):
        idfa.validate_idfa_string((0, 0, 1, MISSING), n=2, k=2)


def test_advancing_past_the_upper_bound_is_an_error():
    assert idfa._advance_idfa_value(MISSING, -1) == (MISSING, True)
    assert idfa._advance_idfa_value(MISSING, 1) == (0, False)
    assert idfa._advance_idfa_value(1, 1) == (MISSING, True)
    with pytest.raises(IDFAEnumerationError, match="invalid transition value"):
        idfa._advance_idfa_value(3, 1)


def test_decoding_to_a_topological_graph():
    transitions = (1, MISSING, 0, 1)
    graph = idfa.idfa_string_to_topological_graph(transitions, n=2, k=2, alphabet=("a", "b"))
    assert graph.transitions == {(0, "a"): 1, (1, "a"): 0, (1, "b"): 1}
    assert idfa.idfa_string_to_topological_graph(transitions, n=2, k=2).alphabet == frozenset({0, 1})
    with pytest.raises(IDFAEnumerationError, match="alphabet length"):
        idfa.idfa_string_to_topological_graph(transitions, n=2, k=2, alphabet=("a",))


def test_rerooting():
    rerooted = idfa.reroot_idfa_string((1, MISSING, 0, 1), new_root=1, n=2, k=2)
    assert rerooted is not None
    idfa.validate_idfa_string(rerooted, n=2, k=2)
    # State 0 cannot be reached from state 1.
    assert idfa.reroot_idfa_string((1, MISSING, MISSING, 1), new_root=1, n=2, k=2) is None
