.. canonical_residual_hmm.rst
.. py:module:: sofic.generators.canonical_residual

**********************
Canonical Residual HMM
**********************

.. warning::

   Experimental. :func:`canonical_residual_hmm` raises :class:`RuntimeError`
   unless called with ``experimental=True``. The construction has no published
   reference. Its correctness is checked against brute-force word
   distributions in the test suite.

The canonical residual finite-state automaton (RFSA) of a regular language has
the *prime* residual languages as its states. A residual is prime when it is not
the union of other residuals :cite:`Denis2002`; see :cite:`MaarandTamm2022` for
its dual, the átomaton. Its stochastic analogue for a process with a finite
ε-machine replaces residual languages by future morphs
:math:`f_s(w) = P(w \mid s)` of the causal states, and unions by nonnegative
combinations. Every morph has :math:`f_s(\lambda) = 1`, so nonnegative
combinations of morphs are convex combinations.

Construction
============

1. Each causal state :math:`s` of positive stationary probability is
   represented by its probabilities :math:`P(w_j \mid s)` on a basis of test
   words. The test words are found breadth first: keep :math:`w` when
   :math:`T^{(w)} \mathbf{1}` is linearly independent of the vectors already
   kept. Linear relations among these coordinates are then exactly linear
   relations among the morphs.
2. A morph is *extreme* when it is not a nonnegative combination of the other
   morphs. This is decided by a linear program
   (:func:`scipy.optimize.linprog`). The extreme morphs form the state set
   :math:`R`.
3. Every morph is written as :math:`f_t = \sum_{r \in R} c_{t r} f_r` with
   :math:`c \ge 0`, again by a linear program.
4. Since :math:`f_r(x w) = \sum_t T^{(x)}_{r t} f_t(w)`, the generator on
   :math:`R` with joint transition matrices :math:`(T^{(x)} C)_{R,\cdot}` and
   initial law :math:`\pi C` emits the future :math:`f_r` from state
   :math:`r`. This follows by induction on word length. The generator
   therefore produces the same process.

The result is a :class:`~sofic.generators.mealy.MealyHMM`. It is in general not
unifilar, and it has at most as many states as the ε-machine. When the morphs
are affinely independent, every morph is extreme and the result is the
ε-machine itself. This holds for every two-state ε-machine, such as the golden
mean and even processes.

A strictly smaller example
==========================

Take two hidden states ``a`` and ``b``. Symbols ``2`` and ``3`` lead to ``a`` and
``b`` from either state. Symbol ``0`` is emitted with the same probability from
both states and keeps the state. Symbol ``1`` is emitted only by ``a`` and moves
to ``a`` or ``b`` with probabilities in the ratio ``split : 1 - split``. The
recurrent beliefs are ``a``, ``b`` and the mixture ``(split, 1 - split)``, so
the ε-machine has three causal states. The third morph is a convex combination
of the other two, and the canonical residual HMM has two states. With the
defaults used in the tests (``split = 1/2``) the ε-machine has statistical
complexity 1.55 bits and the canonical residual HMM has state entropy 0.99
bits.

Relation to generative complexity
=================================

Löhr and Ay :cite:`LohrAy2009` distinguish prescient models, whose minimal
member is the ε-machine, from generative HMMs, which can be much smaller. Their
Example 3.6 is a two-state HMM with infinitely many causal states. For that
example :func:`canonical_residual_hmm` raises
:class:`~sofic.exceptions.MixedStateExplosionError`, because it needs a finite
ε-machine. The canonical residual HMM is a generator, so its state count and
its state entropy are upper bounds on the minimal generator size and the
minimal generator state entropy. Nothing else is computed. In particular, no
minimality among generators is claimed.

API
===

.. autofunction:: canonical_residual_hmm
