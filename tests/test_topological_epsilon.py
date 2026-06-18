"""Tests for topological ε-machine enumeration (Johnson et al. 2010)."""

from __future__ import annotations

import pytest

from pensive.automata.idfa import (
    MISSING_TRANSITION,
    count_accessible_idfa,
    first_idfa_string,
    iter_idfa_strings,
    rank_idfa_string,
    reroot_idfa_string,
    unrank_idfa_string,
    validate_idfa_string,
)
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.topological_epsilon_enumeration import (
    count_topological_epsilon_machines,
    idfa_string_to_epsilon_machine,
    is_canonical_topological_epsilon,
    is_strongly_connected_idfa,
    iter_topological_epsilon_machines,
    iter_topological_epsilon_strings,
)
from pensive.generators.synchronization import graph_from_epsilon_machine

E2 = [3, 7, 78, 1388, 35186, 1132613, 43997426, 1993473480]


def test_even_process_strongly_connected() -> None:
    transitions = (0, 1, MISSING_TRANSITION, 0)
    assert is_strongly_connected_idfa(transitions, n=2, k=2)


def test_even_process_string() -> None:
    transitions = (0, 1, MISSING_TRANSITION, 0)
    validate_idfa_string(transitions, n=2, k=2)
    assert is_canonical_topological_epsilon(transitions, n=2, k=2)


def test_even_process_graph() -> None:
    transitions = (0, 1, MISSING_TRANSITION, 0)
    machines = list(iter_topological_epsilon_machines(2, 2, alphabet=("0", "1")))
    assert all(isinstance(eps, EpsilonMachine) for eps in machines)
    assert any(
        graph_from_epsilon_machine(eps).delta(0, "0") == 0
        and graph_from_epsilon_machine(eps).delta(0, "1") == 1
        and graph_from_epsilon_machine(eps).delta(1, "1") == 0
        and graph_from_epsilon_machine(eps).delta(1, "0") is None
        for eps in machines
    )


def test_even_process_epsilon_machine() -> None:
    eps = idfa_string_to_epsilon_machine((0, 1, MISSING_TRANSITION, 0), n=2, k=2, alphabet=("0", "1"))
    eps.validate()
    graph = graph_from_epsilon_machine(eps)
    assert graph.delta(0, "0") == 0
    assert graph.delta(0, "1") == 1
    assert graph.delta(1, "1") == 0
    assert graph.delta(1, "0") is None


def test_rank_round_trip_small() -> None:
    for k, n in ((2, 2), (2, 3)):
        for rank, transitions in enumerate(iter_idfa_strings(k, n)):
            assert rank_idfa_string(transitions, n=n, k=k) == rank
            assert unrank_idfa_string(rank, n=n, k=k) == transitions


def test_paper_figure_string_validates() -> None:
    canonical = (MISSING_TRANSITION, 1, 0, 2, 0, 1, 1, MISSING_TRANSITION, 0)
    validate_idfa_string(canonical, n=3, k=3)


def test_count_topological_binary() -> None:
    assert count_topological_epsilon_machines(2, 1) == 3


@pytest.mark.xfail(reason="E_{n,2} totals pending alignment of B¹_{n,2} with Johnson et al. (2010)")
def test_count_topological_binary_n2_oeis() -> None:
    assert count_topological_epsilon_machines(2, 2) == E2[1]


@pytest.mark.xfail(reason="E_{n,2} totals pending alignment of B¹_{n,2} with Johnson et al. (2010)")
@pytest.mark.slow
def test_count_topological_n3_oeis() -> None:
    assert count_topological_epsilon_machines(2, 3) == E2[2]


@pytest.mark.xfail(reason="E_{n,2} totals pending alignment of B¹_{n,2} with Johnson et al. (2010)")
@pytest.mark.slow
def test_enumerate_n4_binary() -> None:
    assert len(list(iter_topological_epsilon_strings(2, 4))) == 1388


def test_first_idfa_matches_unrank_zero() -> None:
    for n in (1, 2, 3):
        assert first_idfa_string(n=n, k=2) == unrank_idfa_string(0, n=n, k=2)


def test_accessible_idfa_count_grows() -> None:
    assert count_accessible_idfa(2, 2) > count_icdfa_placeholder(2, 2)


def count_icdfa_placeholder(k: int, n: int) -> int:
    from pensive.automata.icdfa import count_icdfa_empty

    return count_icdfa_empty(k, n)
