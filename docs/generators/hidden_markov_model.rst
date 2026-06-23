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

Finite observed-word probabilities can be queried directly:

.. ipython::

   @doctest float
   In [4]: eps.word_probability((0, 1))
   Out[4]: 0.3333333333333333

   In [5]: eps.word_probabilities(1)

Process equivalence compares finite HMM presentations by constructing
sufficient history and future word bases, rather than using a fixed brute-force
word cutoff:

.. ipython::

   In [6]: eps.is_equal_process(golden_mean(0.5))
   Out[6]: True

API
===

.. autoclass:: StochasticModel
.. autoclass:: HiddenMarkovModel
   :members: word_probability, log_word_probability, word_probabilities, conditional_word_probability, is_equal_process, joint_block_distribution, to_sofic_shift, to_support_nfa, to_support_dfa
.. autoclass:: pensive.generators.mealy.MealyHMM
   :members: add_transition, is_counifilar, is_irreducible, is_ergodic, is_stationary, is_detailed_balance, is_periodic, is_strictly_sofic, to_edge_machine
.. autoclass:: pensive.generators.moore.MooreHMM
   :members: add_transition, set_emission_distribution, to_mealy
