.. complexity.rst
.. py:module:: sofic.algebra.complexity

***********************
Krohn-Rhodes Complexity
***********************

The *group complexity* of a finite semigroup is the least number of group
levels, separated by aperiodic levels, in a cascade that emulates it
:cite:`KrohnRhodes1965` :cite:`RhodesSteinberg2009`. It is zero exactly for
aperiodic semigroups, which for syntactic monoids means star-free languages.
It equals ``n - 1`` for the full transformation monoid ``T_n``. Whether it is
computable in general is open; only complexity one is known to be decidable
:cite:`HenckellRhodesSteinberg2010`.

:func:`krohn_rhodes_complexity` therefore returns
:class:`ComplexityBounds`:

* The upper bound counts the holonomy levels with a nontrivial group.
* The lower bound is ``0`` for an aperiodic semigroup and at least ``1``
  otherwise. It is raised to ``k - 1`` when the elements that map some
  ``k``-element image set into itself act on it as all of ``T_k``, because
  complexity cannot increase under division.

``exact`` is filled in when the two bounds meet. That happens for every
aperiodic semigroup, for every semigroup with a single nontrivial holonomy
level, and for ``T_n``. The sharper Rhodes-Tilson lower bounds are not
implemented.

.. ipython::

   In [1]: from sofic.algebra import TransformationSemigroup, krohn_rhodes_complexity, prime_decomposition

   In [2]: t4 = TransformationSemigroup({"t": (1, 0, 2, 3), "c": (1, 2, 3, 0), "e": (0, 0, 2, 3)})

   @doctest
   In [3]: krohn_rhodes_complexity(t4)
   Out[3]: ComplexityBounds(lower=3, upper=3, exact=3)

   @doctest
   In [4]: prime_decomposition(t4).simple_group_orders()
   Out[4]: (2, 3)

:func:`prime_decomposition` lists the Jordan-Hölder factors of every
holonomy group. These are the simple groups the Krohn-Rhodes theorem
requires, and every simple group divisor of ``S`` divides one of them. It
also reports whether flip-flops (resets) are needed. A nonabelian simple
factor is reported by its order only, because order alone does not identify
it.

.. autofunction:: krohn_rhodes_complexity
.. autoclass:: ComplexityBounds
.. autofunction:: prime_decomposition
.. autoclass:: PrimeDecomposition
   :members:
.. autoclass:: PrimeFactor

Holonomy levels and statistical complexity
==========================================

.. py:module:: sofic.algebra.information

Lifting each causal state to its holonomy coordinates turns the stationary
state distribution into a joint distribution over the level variables
``L_1, ..., L_h``. Distinct states get distinct coordinates, so
``H[L_1, ..., L_h] = C_mu``. The chain rule then splits the statistical
complexity into ``H[L_i | L_<i]``, the information resolved at each level,
from the coarsest approximation at the top to individual states at the
bottom. The split depends on the chosen lift; :func:`holonomy_information`
uses the canonical one.

.. note:: Original, uncited: the literature does not pair holonomy
   coordinates with a stationary measure.

.. ipython::

   In [1]: from sofic.algebra import holonomy_information

   In [2]: from sofic.examples import even_process

   In [3]: info = holonomy_information(even_process())

   @doctest
   In [4]: [round(h, 6) for h in info.level_entropies]
   Out[4]: [0.918296, 0.0]

For the even process the top level, which separates the two causal states,
carries all of ``C_mu``.

.. autofunction:: holonomy_information
.. autoclass:: HolonomyInformation

Belief-map semigroups
=====================

.. py:module:: sofic.algebra.belief

An observer of an HMM updates its belief by ``eta -> eta T^(x) / (eta T^(x) 1)``.
:func:`projective_semigroup` enumerates the word matrices ``T^(w)`` up to a
positive scalar, up to a cap, and returns the result as a transformation
monoid, so the rest of :mod:`sofic.algebra` applies. If the enumeration
closes, every initial belief has a finite orbit. Finiteness of matrix
semigroups is decidable :cite:`MandelSimon1977` :cite:`Jacob1977`, but this
is only a capped enumeration. It is also only a sufficient condition: the
mixed-state presentation from one particular belief can be finite even when
this semigroup is infinite. :func:`mixed_state_semigroup` takes the
transition semigroup of the mixed-state presentation directly.

.. ipython::

   In [1]: from sofic.algebra import projective_semigroup

   @doctest
   In [2]: len(projective_semigroup(even_process()).states)
   Out[2]: 7

For a unifilar machine like the even process this matches the transition
monoid. The simple nonunifilar source, whose mixed-state presentation is
infinite, never closes and raises
:class:`~sofic.exceptions.SemigroupTooLargeError`.

.. autofunction:: projective_semigroup
.. autofunction:: mixed_state_semigroup
