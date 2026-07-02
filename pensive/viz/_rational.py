"""Shared rational-number helper for viz label formatting."""

from __future__ import annotations

from fractions import Fraction

_TWO_DIGIT_RATIONAL_ATOL = 1e-9
_MAX_TWO_DIGIT_RATIONAL = 99


def two_digit_rational(value: float, *, atol: float = _TWO_DIGIT_RATIONAL_ATOL) -> Fraction | None:
    """Return a reduced rational with 1 <= p, q <= 99 when ``value`` matches exactly."""
    if value <= 0.0 or value >= 1.0:
        return None
    frac = Fraction(value).limit_denominator(_MAX_TWO_DIGIT_RATIONAL)
    if abs(float(frac) - value) >= atol:
        return None
    if not (1 <= frac.numerator <= _MAX_TWO_DIGIT_RATIONAL and 1 <= frac.denominator <= _MAX_TWO_DIGIT_RATIONAL):
        return None
    return frac
