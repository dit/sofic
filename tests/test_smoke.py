"""Smoke tests for the sofic package."""

import sofic


def test_import():
    assert sofic.__version__
