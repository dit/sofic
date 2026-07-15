"""Probability scalars: floats or sympy expressions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

Prob = Any  # float | sympy.Expr


def _sympy():
    try:
        import sympy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Symbolic probabilities require sympy. Install with: pip install pensive[symbolic]"
        ) from exc
    return sympy


def is_symbolic(value: Any) -> bool:
    """Return whether ``value`` is an exact sympy expression (not a Python float).

    Exact rationals / integers and expressions with free symbols are symbolic.
    ``sympy.Float`` is treated as numeric and will be coerced to ``float``.
    """
    try:
        import sympy
    except ImportError:
        return False
    if not isinstance(value, sympy.Expr):
        return False
    return not isinstance(value, sympy.Float)


def has_symbolic(values: Iterable[Any]) -> bool:
    """Return whether any entry in ``values`` is symbolic."""
    return any(is_symbolic(value) for value in values)


def as_prob(value: Any) -> Prob:
    """Coerce ``value`` to a stored probability: sympy Expr pass-through, else float.

    Plain Python ``int`` values are left unchanged so they compose with sympy
    without introducing ``Float`` coefficients (``0 + a`` stays exact).
    """
    if is_symbolic(value):
        return value
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return value
    try:
        import sympy

        if isinstance(value, sympy.Basic):
            return float(value)
    except ImportError:
        pass
    return float(value)


def simplify_prob(value: Prob) -> Prob:
    """Simplify a symbolic probability; return floats/ints unchanged."""
    if not is_symbolic(value):
        return value
    sp = _sympy()
    return sp.simplify(value)


def is_zero(value: Prob, *, atol: float = 1e-15) -> bool:
    """Return whether ``value`` is (numerically or symbolically) zero."""
    if is_symbolic(value):
        sp = _sympy()
        return sp.simplify(value) == 0
    if isinstance(value, int):
        return value == 0
    try:
        return abs(float(value)) <= atol
    except (TypeError, ValueError):
        return False


def is_positive_mass(value: Prob, *, atol: float = 1e-15) -> bool:
    """Return whether ``value`` carries positive probability mass.

    Symbolic expressions with free symbols are treated as positive unless they
    simplify to a non-positive number (matching dit's free-symbol convention).
    """
    if is_zero(value, atol=atol):
        return False
    if is_symbolic(value):
        sp = _sympy()
        simplified = sp.simplify(value)
        if simplified.free_symbols:
            return simplified.is_nonpositive is not True
        try:
            return float(simplified) > atol
        except (TypeError, ValueError):
            return simplified != 0
    return float(value) > atol


def probs_equal(
    left: Prob,
    right: Prob,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    constraints: SymbolConstraints | None = None,
) -> bool:
    """Equality for probability scalars (exact sympy simplify, else numeric close).

    When ``constraints`` are supplied and either operand is symbolic, equality is
    tested modulo the constraint ideal (see :class:`SymbolConstraints`), so
    expressions that coincide only under a parameter's minimal polynomial are
    recognized as equal.
    """
    if is_symbolic(left) or is_symbolic(right):
        sp = _sympy()
        if constraints is not None:
            return constraints.equal(left, right)
        return sp.simplify(sp.sympify(left) - sp.sympify(right)) == 0
    return bool(np.isclose(float(left), float(right), rtol=rtol, atol=atol))


def sum_probs(values: Iterable[Prob]) -> Prob:
    """Sum probability scalars, preserving sympy when any term is symbolic."""
    items = list(values)
    if not items:
        return 0
    if has_symbolic(items):
        sp = _sympy()
        total = sp.Integer(0)
        for value in items:
            total += sp.sympify(value)
        return simplify_prob(total)
    return float(sum(float(value) for value in items))


def row_sums_to_one(probs: Sequence[Prob], *, atol: float = 1e-9) -> bool:
    """Return whether ``probs`` sum to one (exact sympy or numeric close)."""
    if not probs:
        return True
    total = sum_probs(probs)
    if is_symbolic(total) or has_symbolic(probs):
        sp = _sympy()
        # Free symbols: accept rows that are identically 1, else skip (dit-style).
        if getattr(total, "free_symbols", None):
            return True
        return sp.simplify(sp.sympify(total) - 1) == 0
    return bool(np.isclose(float(total), 1.0, atol=atol))


def zeros(shape: tuple[int, ...], *, symbolic: bool = False) -> np.ndarray:
    """Allocate a probability array (object dtype when ``symbolic``)."""
    if symbolic:
        sp = _sympy()
        array = np.empty(shape, dtype=object)
        array.fill(sp.Integer(0))
        return array
    return np.zeros(shape, dtype=float)


def array_sum(array: np.ndarray) -> Prob:
    """Sum an array, using Python reduction for object (sympy) arrays."""
    flat = np.asarray(array).ravel()
    if flat.dtype == object or has_symbolic(flat):
        return sum_probs(flat.tolist())
    return float(flat.sum())


def matvec(row: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-vector times matrix, supporting object-dtype sympy entries."""
    if row.dtype == object or matrix.dtype == object or has_symbolic(row) or has_symbolic(matrix.ravel()):
        sp = _sympy()
        n = matrix.shape[1]
        out = np.empty(n, dtype=object)
        for j in range(n):
            total = sp.Integer(0)
            for i in range(matrix.shape[0]):
                total += sp.sympify(row[i]) * sp.sympify(matrix[i, j])
            out[j] = simplify_prob(total)
        return out
    return row @ matrix


def canonical_prob_key(value: Prob, constraints: SymbolConstraints | None = None) -> Any:
    """Hash-stable key for partition signatures (simplified sympy or float).

    With ``constraints`` the key is taken in the constraint residue field, so
    two probabilities that are equal modulo the constraints share a key.
    """
    if constraints is not None:
        return constraints.key(value)
    if is_symbolic(value):
        sp = _sympy()
        return sp.srepr(sp.simplify(value))
    return float(value)


class SymbolConstraints:
    """Algebraic side-relations (each ``expr == 0``) over probability symbols.

    Used to compare symbolic probabilities modulo the ideal they generate, so
    that states which coincide only under a parameter's minimal polynomial (for
    example the Misiurewicz root ``a**3 - 2*a - 2 == 0``) are recognized as
    equal when de-duplicating beliefs or merging causal states.

    When the relations reduce to a single univariate polynomial in one symbol,
    equality and keys are computed exactly in the residue field
    ``Q[s] / (minpoly)`` via ``p * q^{-1} mod minpoly``.  Otherwise a
    Groebner-basis ideal-membership test (with a plain ``simplify`` fallback) is
    used for equality, and the simplified ``srepr`` for keys.
    """

    def __init__(self, relations: Iterable[Any]) -> None:
        sp = _sympy()
        rels = tuple(sp.sympify(relation) for relation in relations)
        self.relations = rels
        symbols: set[Any] = set()
        for relation in rels:
            symbols |= relation.free_symbols
        self._symbols = tuple(sorted(symbols, key=str))
        self._var: Any | None = None
        self._minpoly: Any | None = None
        if len(rels) == 1 and len(self._symbols) == 1:
            var = self._symbols[0]
            try:
                self._minpoly = sp.Poly(rels[0], var, domain=sp.QQ)
                self._var = var
            except sp.PolynomialError:
                self._minpoly = None
                self._var = None

    def __repr__(self) -> str:
        return f"SymbolConstraints({list(self.relations)!r})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SymbolConstraints) and self.relations == other.relations

    def __hash__(self) -> int:
        return hash(self.relations)

    def _residue_key(self, value: Prob) -> tuple[Any, ...] | None:
        """Residue-field representative coefficients, or ``None`` if unavailable."""
        if self._minpoly is None or self._var is None:
            return None
        if value.free_symbols - {self._var}:
            return None
        sp = _sympy()
        from sympy.polys.polyerrors import NotInvertible

        try:
            num, den = sp.fraction(sp.together(value))
            num_poly = sp.Poly(num, self._var, domain=sp.QQ)
            den_poly = sp.Poly(den, self._var, domain=sp.QQ)
            den_inv = sp.invert(den_poly, self._minpoly)
            rep = sp.rem((num_poly * den_inv).as_expr(), self._minpoly.as_expr(), self._var)
            coeffs = sp.Poly(rep, self._var, domain=sp.QQ).all_coeffs()
        except (sp.PolynomialError, NotInvertible, ValueError, ZeroDivisionError):
            return None
        return tuple(coeffs)

    def key(self, value: Prob) -> Any:
        """Canonical, hashable key equal for probabilities equal modulo the ideal."""
        if not is_symbolic(value):
            return float(value)
        residue = self._residue_key(value)
        if residue is not None:
            return ("mod", residue)
        return canonical_prob_key(value)

    def equal(self, left: Prob, right: Prob) -> bool:
        """Return whether ``left`` and ``right`` are equal modulo the constraints."""
        sp = _sympy()
        left_residue = self._residue_key(sp.sympify(left)) if is_symbolic(left) else None
        right_residue = self._residue_key(sp.sympify(right)) if is_symbolic(right) else None
        if left_residue is not None and right_residue is not None:
            return left_residue == right_residue
        diff = sp.sympify(left) - sp.sympify(right)
        if self.relations:
            try:
                num = sp.numer(sp.together(diff))
                basis = sp.groebner(self.relations, *self._symbols, order="lex")
                _, remainder = sp.reduced(sp.expand(num), basis, *self._symbols)
                if sp.simplify(remainder) == 0:
                    return True
            except (sp.PolynomialError, ValueError, ZeroDivisionError):
                pass
        return sp.simplify(diff) == 0
