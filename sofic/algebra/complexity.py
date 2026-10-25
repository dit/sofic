"""Krohn-Rhodes complexity bounds and prime decompositions.

By the Krohn-Rhodes theorem every finite transformation semigroup divides a
cascade of simple groups and flip-flops (the three-element monoid of two
resets and the identity), and the simple groups needed are exactly the simple
group divisors of ``S`` :cite:`KrohnRhodes1965`. The *group complexity*
``c(S)`` is the least number of group levels, separated by aperiodic levels,
in such a cascade. It is monotone under division, ``c(S) = 0`` exactly when
``S`` is aperiodic, and ``c(T_n) = n - 1`` for the full transformation monoid
:cite:`RhodesSteinberg2009`. Whether ``c(S)`` is computable in general is
open; only complexity one is known to be decidable
:cite:`HenckellRhodesSteinberg2010`, and that procedure is not implemented
here. This module returns bounds instead, exact when they meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from sofic.algebra._groups import composition_factor_orders
from sofic.exceptions import SemigroupTooLargeError

if TYPE_CHECKING:
    from sofic.algebra.holonomy import HolonomyDecomposition, Subset

__all__ = [
    "ComplexityBounds",
    "PrimeDecomposition",
    "PrimeFactor",
    "is_full_transformation_monoid",
    "krohn_rhodes_complexity",
    "prime_decomposition",
]


class ComplexityBounds(NamedTuple):
    """Lower and upper bounds on Krohn-Rhodes group complexity.

    ``exact`` is the complexity when ``lower == upper`` and ``None`` otherwise.
    """

    lower: int
    upper: int
    exact: int | None


def krohn_rhodes_complexity(machine: Any, *, decomposition: HolonomyDecomposition | None = None) -> ComplexityBounds:
    """Return bounds on the Krohn-Rhodes complexity of a semigroup or machine.

    * **Upper bound:** the number of holonomy levels with a nontrivial group.
      The holonomy cascade emulates ``S`` and each level is a permutation
      group with resets :cite:`EgriNagyNehaniv2015`.
    * **Lower bound:** ``0`` when ``S`` is aperiodic, which holds iff every
      holonomy group is trivial, and otherwise at least ``1``. If the
      elements mapping an image set ``A`` into itself restrict to all
      ``|A|**|A|`` maps of ``A``, then ``T_|A|`` divides ``S`` and so
      ``c(S) >= |A| - 1`` :cite:`RhodesSteinberg2009`. This second test
      enumerates ``S`` and is skipped if ``S`` is too large to enumerate,
      except that ``S = T_n`` itself is recognized from its generators
      (:func:`is_full_transformation_monoid`).

    Sharper lower bounds from the literature (Rhodes-Tilson chains of
    essential J-classes) are not implemented.

    Examples
    --------
    >>> from sofic.algebra import TransformationSemigroup
    >>> krohn_rhodes_complexity(TransformationSemigroup({"a": (1, 2, 0)}))
    ComplexityBounds(lower=1, upper=1, exact=1)
    """
    from sofic.algebra.holonomy import holonomy_decomposition

    hd = decomposition if decomposition is not None else holonomy_decomposition(machine)
    upper = sum(not level.is_trivial for level in hd.levels)
    lower = 0
    if upper:
        lower = max(1, _full_transformation_bound(hd))
    return ComplexityBounds(lower, upper, lower if lower == upper else None)


def is_full_transformation_monoid(semigroup: Any) -> bool:
    """Return whether ``S`` is the full transformation monoid ``T_n``, from its generators alone.

    A product involving a non-permutation is not a permutation, and rank
    never increases under composition. So ``S = T_n`` exactly when the
    permutation generators generate ``S_n`` and some generator has rank
    ``n - 1``; conversely ``S_n`` and any rank ``n - 1`` map generate ``T_n``
    :cite:`Howie1995`.
    """
    from math import factorial

    from sofic.algebra._groups import permutation_group

    n = semigroup.degree
    gens = list(semigroup.generators.values())
    if n == 1:
        return True
    if not any(g.rank() == n - 1 for g in gens):
        return False
    perms = [g.images for g in gens if g.is_permutation()]
    return bool(perms) and int(permutation_group(perms, n).order()) == factorial(n)


def _full_transformation_bound(hd: HolonomyDecomposition) -> int:
    """Largest ``k - 1`` such that ``T_k`` appears as a local action on an image set."""
    sg = hd.semigroup
    if is_full_transformation_monoid(sg):
        return sg.degree - 1
    try:
        elements = [g.images for g in sg]
    except SemigroupTooLargeError:
        return 0
    best = 0
    candidates: list[Subset] = sorted((s for s in hd.image_set if len(s) > 2), key=len, reverse=True)
    for subset in candidates:
        k = len(subset)
        if k - 1 <= best or k**k > len(elements):
            continue
        ordered = sorted(subset)
        position = {x: i for i, x in enumerate(ordered)}
        local = {tuple(position[e[x]] for x in ordered) for e in elements if all(e[x] in position for x in ordered)}
        if len(local) == k**k:
            best = k - 1
    return best


@dataclass(frozen=True)
class PrimeFactor:
    """A simple group occurring as a composition factor of a holonomy group.

    ``kind`` is ``"cyclic"`` (order prime) or ``"nonabelian simple"``. A
    nonabelian simple group is reported by order only, since order does not
    always determine it (``|A_8| = |PSL(3, 4)|``).
    """

    kind: str
    order: int
    depth: int
    representative: frozenset[int]


@dataclass(frozen=True)
class PrimeDecomposition:
    """The Krohn-Rhodes primes read off a holonomy decomposition.

    ``groups`` lists the composition factors of every holonomy group, with
    multiplicity and position. ``needs_flip_flop`` is true when ``S``
    contains a non-permutation, so resets (flip-flops) are required.
    """

    groups: tuple[PrimeFactor, ...]
    needs_flip_flop: bool

    def simple_group_orders(self) -> tuple[int, ...]:
        """Distinct orders of the simple group factors, ascending."""
        return tuple(sorted({f.order for f in self.groups}))


def prime_decomposition(machine: Any, *, decomposition: HolonomyDecomposition | None = None) -> PrimeDecomposition:
    """Return the simple-group and flip-flop components needed to build ``S``.

    The Jordan-Hölder factors of the holonomy groups are simple group
    divisors of ``S`` and, by the prime decomposition theorem, every simple
    group divisor of ``S`` divides one of them :cite:`KrohnRhodes1965`
    :cite:`EgriNagyNehaniv2015`.

    Examples
    --------
    >>> from sofic.algebra import TransformationSemigroup
    >>> pd = prime_decomposition(TransformationSemigroup({"t": (1, 0, 2), "c": (1, 2, 0), "e": (0, 0, 2)}))
    >>> pd.simple_group_orders(), pd.needs_flip_flop
    ((2, 3), True)
    """
    from sofic.algebra.holonomy import holonomy_decomposition

    hd = decomposition if decomposition is not None else holonomy_decomposition(machine)
    factors = []
    for level in hd.levels:
        if level.is_trivial:
            continue
        for rep, group in zip(level.representatives, level.groups, strict=True):
            for order in composition_factor_orders(group):
                kind = "cyclic" if _is_prime(order) else "nonabelian simple"
                factors.append(PrimeFactor(kind, order, level.depth, rep))
    return PrimeDecomposition(tuple(factors), not hd.semigroup.is_group())


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % p for p in range(2, int(n**0.5) + 1))
