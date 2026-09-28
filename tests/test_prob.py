"""Tests for probability scalar helpers."""

from __future__ import annotations

import builtins

from sofic.generators import prob


def test_numeric_checks_never_import_sympy(monkeypatch):
    """Regression: without sympy installed, every check retried the failed import,
    which made entropy_rate() on a 127-state machine take seconds."""
    calls = []
    real_import = builtins.__import__

    def spy(name, *args, **kwargs):
        if name == "sympy":
            calls.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy)
    assert not prob.is_symbolic(0.5)
    assert not prob.has_symbolic([0.1, 0.9])
    assert prob.as_prob(0.25) == 0.25
    assert calls == []
