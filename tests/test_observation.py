"""Tests for observation tables."""

import pytest

from pensive.automata.languages.base import AutomatonLanguage
from pensive.automata.observation import ObservationTable


def test_defaults():
    table = ObservationTable()
    assert () in table.access_words
    assert () in table.experiments


def test_observation_extractors():
    table = ObservationTable(
        access_words=frozenset({(), ("a",)}),
        experiments=frozenset({(), ("b",)}),
        membership={
            (): False,
            ("a",): True,
            ("b",): False,
            ("a", "b"): True,
        },
    )
    dfa = table.to_minimal_dfa()
    assert ("a",) in AutomatonLanguage(dfa)
