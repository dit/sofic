.. support.rst
.. py:module:: sofic.generators.support

*******************
Process supports
*******************

The *stationary support* of an HMM is the set of finite words
:math:`w = x_0 \cdots x_{n-1}` with :math:`P(X_{0:n} = w) > 0` under the
stationary law. It is a factorial regular language, recognized by the NFA
:func:`support_nfa`: the states of positive stationary probability, all initial
and accepting, joined by the positive-probability transitions.

:meth:`~sofic.generators.base.HiddenMarkovModel.to_support_nfa` is a different
automaton: it starts from *every* state, transient ones included, so it also
accepts words that only a transient state can emit. Those words have stationary
probability zero and are rejected by :func:`support_nfa`.

Support decisions
=================

Comparisons of supports are inclusions of regular languages, decided by
antichain inclusion without determinizing :cite:`DeWulf2006`.

:func:`support_includes` ``(p, q)``
   Every word of positive :math:`P` probability has positive :math:`Q`
   probability.

:func:`is_absolutely_continuous` ``(p, q)``
   :math:`P_{0:n} \ll Q_{0:n}` for every block length :math:`n`, i.e.
   absolute continuity on the finite-dimensional cylinders. This is the same
   decision as :func:`support_includes`, and it is exactly the condition for the
   block divergences :math:`D(P_{0:n} \| Q_{0:n})` and the
   :doc:`relative entropy rate <relative_entropy_rate>` to be finite. It is not
   absolute continuity of the laws on infinite sequences: distinct ergodic
   stationary measures are mutually singular there.

:func:`support_equal` ``(p, q)``
   Inclusion both ways: the two processes allow the same finite words, whatever
   their probabilities.

The golden mean process forbids ``11``, so it is absolutely continuous with
respect to the fair coin but not the other way round.

.. ipython::

   In [1]: from sofic.examples import bernoulli, golden_mean

   In [2]: from sofic.generators.support import is_absolutely_continuous, support_equal, support_includes

   @doctest
   In [3]: support_includes(golden_mean(0.5), bernoulli())
   Out[3]: True

   @doctest
   In [4]: is_absolutely_continuous(bernoulli(), golden_mean(0.5))
   Out[4]: False

   @doctest
   In [5]: support_equal(golden_mean(0.3), golden_mean(0.8))
   Out[5]: True

The same decisions are available as
:meth:`~sofic.generators.base.HiddenMarkovModel.support_includes`,
:meth:`~sofic.generators.base.HiddenMarkovModel.is_absolutely_continuous`, and
:meth:`~sofic.generators.base.HiddenMarkovModel.support_equal`.

API
===

.. autofunction:: support_nfa

.. autofunction:: support_includes

.. autofunction:: is_absolutely_continuous

.. autofunction:: support_equal
