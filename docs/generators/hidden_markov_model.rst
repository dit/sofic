.. hidden_markov_model.rst
.. py:module:: pensive.generators.base

*******************
Hidden Markov Model
*******************

:class:`~pensive.generators.base.StochasticModel` is the base for row-stochastic
generators. :class:`~pensive.generators.base.HiddenMarkovModel` adds an
observation alphabet and emission semantics. The HMM conventions and standard
inference problems follow Baum and Petrie and Rabiner's tutorial
:cite:`BaumPetrie1966,Rabiner1989`.

Mealy vs Moore
==============

* :class:`~pensive.generators.mealy.MealyHMM` — joint transition
  :math:`P(q', o \mid q)` on edges.
* :class:`~pensive.generators.moore.MooreHMM` — emission distribution
  :math:`P(o \mid q)` on states; convert with :func:`~pensive.generators.conversions.moore_to_mealy`.
  The naming follows Mealy and Moore machine conventions :cite:`Mealy1955,Moore1956`.

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
   :members: joint_block_distribution, to_sofic_shift, to_support_nfa, to_support_dfa
.. autoclass:: pensive.generators.mealy.MealyHMM
   :members: add_transition, to_edge_machine
.. autoclass:: pensive.generators.moore.MooreHMM
   :members: add_transition, set_emission_distribution, to_mealy
