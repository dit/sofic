"""Varieties of regular languages, decided on the syntactic monoid.

Eilenberg's variety theorem matches classes of regular languages closed
under Boolean operations and quotients with pseudovarieties of finite monoids
(or semigroups), so membership is decided by checking an identity on the
syntactic monoid or semigroup :cite:`Eilenberg1976` :cite:`Pin1986`.
Products here follow word order: ``s * t`` is the image of ``uv`` when ``s``
and ``t`` are the images of ``u`` and ``v``.

Each test accepts a language (anything
:func:`~sofic.algebra.syntactic.syntactic_monoid` accepts, plus an
``alphabet`` keyword) or an already-built
:class:`~sofic.algebra.transformation.TransformationSemigroup`, which is then
used as is.
"""

from __future__ import annotations

import itertools
from typing import Any

from sofic.algebra.transformation import TransformationSemigroup

__all__ = [
    "is_commutative_language",
    "is_definite",
    "is_group_language",
    "is_j_trivial",
    "is_l_trivial",
    "is_locally_testable",
    "is_piecewise_testable",
    "is_r_trivial",
    "is_reverse_definite",
]


def _monoid(language: Any, **kwargs: Any) -> TransformationSemigroup:
    if isinstance(language, TransformationSemigroup):
        return language
    from sofic.algebra.syntactic import syntactic_monoid

    return syntactic_monoid(language, **kwargs)


def _semigroup(language: Any, **kwargs: Any) -> TransformationSemigroup:
    if isinstance(language, TransformationSemigroup):
        return language
    from sofic.algebra.syntactic import syntactic_monoid

    return syntactic_monoid(language, monoid=False, **kwargs)


def is_j_trivial(language: Any, **kwargs: Any) -> bool:
    """Return whether every J-class of the syntactic monoid is a singleton."""
    return all(d.size == 1 for d in _monoid(language, **kwargs).green().d_classes)


def is_r_trivial(language: Any, **kwargs: Any) -> bool:
    """Return whether every R-class of the syntactic monoid is a singleton."""
    return all(len(r) == 1 for r in _monoid(language, **kwargs).green().r_classes)


def is_l_trivial(language: Any, **kwargs: Any) -> bool:
    """Return whether every L-class of the syntactic monoid is a singleton."""
    return all(len(c) == 1 for c in _monoid(language, **kwargs).green().l_classes)


def is_piecewise_testable(language: Any, **kwargs: Any) -> bool:
    """Return whether membership depends only on the scattered subwords up to some length.

    Simon's theorem: a language is piecewise testable iff its syntactic monoid
    is J-trivial :cite:`Simon1975`.

    Examples
    --------
    >>> is_piecewise_testable("a*b*", alphabet="ab"), is_piecewise_testable("(a|b)*a", alphabet="ab")
    (True, False)
    """
    return is_j_trivial(language, **kwargs)


def is_locally_testable(language: Any, **kwargs: Any) -> bool:
    """Return whether membership depends only on prefixes, suffixes, and factors of bounded length.

    Brzozowski-Simon and McNaughton: a language is locally testable iff its
    syntactic *semigroup* is locally idempotent and commutative, that is,
    ``eSe`` is an idempotent commutative monoid for every idempotent ``e``
    :cite:`BrzozowskiSimon1973` :cite:`McNaughton1974`.

    Examples
    --------
    >>> is_locally_testable("(ab)*", alphabet="ab"), is_locally_testable("a*ba*", alphabet="ab")
    (True, False)
    """
    sg = _semigroup(language, **kwargs)
    elements = list(range(len(sg)))
    for e in sg.idempotents():
        local = {sg.multiply(sg.multiply(e, s), e) for s in elements}
        for x in local:
            if sg.multiply(x, x) != x:
                return False
        for x, y in itertools.combinations(local, 2):
            if sg.multiply(x, y) != sg.multiply(y, x):
                return False
    return True


def is_definite(language: Any, **kwargs: Any) -> bool:
    """Return whether membership depends only on a bounded-length suffix.

    A language is definite iff its syntactic semigroup satisfies ``s e = e``
    for every element ``s`` and idempotent ``e`` :cite:`Pin1986`.

    Examples
    --------
    >>> is_definite("(a|b)*a", alphabet="ab"), is_definite("a(a|b)*", alphabet="ab")
    (True, False)
    """
    sg = _semigroup(language, **kwargs)
    return all(sg.multiply(s, e) == e for e in sg.idempotents() for s in range(len(sg)))


def is_reverse_definite(language: Any, **kwargs: Any) -> bool:
    """Return whether membership depends only on a bounded-length prefix.

    A language is reverse definite iff its syntactic semigroup satisfies
    ``e s = e`` for every element ``s`` and idempotent ``e`` :cite:`Pin1986`.
    """
    sg = _semigroup(language, **kwargs)
    return all(sg.multiply(e, s) == e for e in sg.idempotents() for s in range(len(sg)))


def is_group_language(language: Any, **kwargs: Any) -> bool:
    """Return whether the syntactic monoid is a group (the minimal DFA is a permutation automaton)."""
    return _monoid(language, **kwargs).is_group()


def is_commutative_language(language: Any, **kwargs: Any) -> bool:
    """Return whether membership is invariant under permuting letters (commutative syntactic monoid)."""
    gens = list(_monoid(language, **kwargs).generators.values())
    return all(a * b == b * a for a, b in itertools.combinations(gens, 2))
