"""Sequential capital-letter state labels."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sofic.states import letter_label_index, next_sequential_label_index, sequential_labels


def test_labels_run_through_one_two_and_three_letters() -> None:
    labels = sequential_labels(26 + 26**2 + 26**3 + 1)
    assert labels[:3] == ("A", "B", "C")
    assert labels[25:28] == ("Z", "AA", "AB")
    assert labels[26 + 25] == "AZ"
    assert labels[26 + 26] == "BA"
    assert labels[26 + 26**2 - 1] == "ZZ"
    assert labels[26 + 26**2] == "AAA"
    assert labels[26 + 26**2 + 26**3 - 1] == "ZZZ"
    assert labels[-1] == "AAAA"
    assert len(set(labels)) == len(labels)


def test_start_offsets_the_sequence() -> None:
    assert sequential_labels(3, start=25) == ("Z", "AA", "AB")
    assert sequential_labels(0, start=5) == ()


@given(st.integers(0, 10**6))
def test_label_index_round_trip(index: int) -> None:
    (label,) = sequential_labels(1, start=index)
    assert letter_label_index(label) == index


@pytest.mark.parametrize("label", ["", "a", "A1", "Ab", 3, ("A",), None])
def test_non_letter_labels_have_no_index(label) -> None:
    assert letter_label_index(label) is None


def test_next_index_follows_the_largest_letter_label() -> None:
    assert next_sequential_label_index([]) == 0
    assert next_sequential_label_index(["A", "C", 7, "x"]) == 3
    assert next_sequential_label_index(["Z"]) == 26
    assert next_sequential_label_index(["B", "AA"]) == 27


def test_negative_arguments_raise() -> None:
    with pytest.raises(ValueError):
        sequential_labels(-1)
    with pytest.raises(ValueError):
        sequential_labels(1, start=-1)
