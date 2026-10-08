.. relative_entropy_rate.rst
.. py:module:: sofic.generators.relative_entropy_rate

*********************
Relative Entropy Rate
*********************

The relative entropy rate between stationary processes :math:`P` and :math:`Q`,
in bits per symbol, is

.. math::

   D(P \| Q) = \lim_{n \to \infty} \tfrac{1}{n} D(P_{0:n} \| Q_{0:n})

:cite:`Gray1990,Cover2006`. Both processes are taken in their stationary laws.

Absolute continuity
===================

The rate is infinite as soon as some word with positive :math:`P` probability
has :math:`Q` probability zero. The stationary support of an HMM is a regular
language, so this is checked by building both support automata and asking
:meth:`~sofic.automata.base.LabeledAutomaton.includes`, which uses antichain
inclusion :cite:`DeWulf2006`. For example, the golden mean process is
absolutely continuous with respect to the fair coin, but not the other way
round.

Exact rate for unifilar references
==================================

When :math:`Q` is unifilar (an ε-machine or other unifilar HMM),

.. math::

   D(P \| Q) = -h_\mu(P) - \lim_{n \to \infty} \mathbb{E}_P[\log_2 Q(X_n \mid X_{0:n})].

For a :math:`k`-step Markov :math:`Q` this is Lemma 3.10 of Gray
:cite:`Gray1990` (§3.5). In general the past determines the set of :math:`Q`
states it is consistent with. The limit is then an average over the long-run
law of the finite chain of pairs (:math:`P` state, set of :math:`Q` states).
This requires :math:`P` to be ergodic and :math:`Q` to be exactly synchronized
by typical :math:`P` pasts.

.. ipython::

   In [1]: from sofic.examples import bernoulli, golden_mean

   In [2]: from sofic.generators.relative_entropy_rate import relative_entropy_rate

   @doctest float
   In [3]: relative_entropy_rate(golden_mean(0.5), bernoulli(0.5))
   Out[3]: 0.3333333333333335

   @doctest
   In [4]: relative_entropy_rate(bernoulli(0.5), golden_mean(0.5))
   Out[4]: inf

Bounds for general references
=============================

:func:`relative_entropy_rate_bounds` enumerates the words of length :math:`n`.
It brackets the cross-entropy rate between superadditive and subadditive block
quantities built from :math:`\max_s Q_s` and :math:`\min_s Q_s`. It brackets the
entropy rate of :math:`P` between the hidden-Markov bounds
:math:`H[X_{n-1} \mid X_{0:n-1}, S_0] \le h_\mu \le H[X_{n-1} \mid X_{0:n-1}]`
(:cite:`Cover2006`, Theorem 4.5.1).

API
===

.. autofunction:: relative_entropy_rate

.. autofunction:: relative_entropy_rate_bounds

.. autoclass:: RelativeEntropyRateBounds
   :members:
