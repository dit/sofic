.. hidden_markov_model.rst
.. py:module:: pensive.generators.base

*******************
Hidden Markov Model
*******************

:class:`~pensive.generators.base.StochasticModel` is the base for row-stochastic
generators. :class:`~pensive.generators.base.HiddenMarkovModel` adds an
observation alphabet and emission semantics.

Mealy vs Moore
==============

* :class:`~pensive.generators.mealy.MealyHMM` — joint transition
  :math:`P(q', o \mid q)` on edges.
* :class:`~pensive.generators.moore.MooreHMM` — emission distribution
  :math:`P(o \mid q)` on states; convert with :func:`~pensive.generators.conversions.moore_to_mealy`.

.. ipython::

   In [1]: from pensive.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   @doctest float
   In [3]: eps.entropy_rate()
   Out[3]: 0.6666666666666665

API
===

.. autoclass:: StochasticModel
.. autoclass:: HiddenMarkovModel
   :members: joint_block_distribution
.. autoclass:: pensive.generators.mealy.MealyHMM
.. autoclass:: pensive.generators.moore.MooreHMM
