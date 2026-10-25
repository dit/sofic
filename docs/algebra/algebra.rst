.. algebra.rst
.. py:module:: sofic.algebra

*******
Algebra
*******

The :mod:`sofic.algebra` package brings algebraic automata theory to every
deterministic presentation in ``sofic``. A DFA, a unifilar automaton, a
unifilar HMM or ε-machine, or a right-resolving shift presentation has one
partial map of its states per symbol. Those maps generate a finite
*transformation semigroup*, and the Krohn-Rhodes theorem breaks that
semigroup into a cascade of simple groups and flip-flops
:cite:`KrohnRhodes1965` :cite:`RhodesNehanivHirsch2010`.

* :doc:`semigroups` covers transformation semigroups, Green's relations,
  syntactic monoids, and star-freeness.
* :doc:`holonomy` covers the holonomy decomposition as an explicit cascade
  product that can be checked against the original machine.
* :doc:`complexity` covers Krohn-Rhodes complexity bounds, prime factors,
  how statistical complexity splits across the holonomy levels, and the
  semigroup of an HMM's belief updates.
* :doc:`effective_theories` covers truncating the cascade into coarse-grained
  machines on overlapping superstates :cite:`DeDeo2011`.
* :doc:`noise` covers probabilistic automata: rewriting them as random choices
  of deterministic maps, the walk over semigroup elements, and how much
  information is erased versus dissipated.

The group-theoretic outputs need the optional ``sympy`` dependency, which the
``sofic[symbolic]`` extra installs.

.. toctree::
   :maxdepth: 1

   semigroups
   holonomy
   complexity
   effective_theories
   noise
