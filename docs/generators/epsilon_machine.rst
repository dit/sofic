.. epsilon_machine.rst
.. py:module:: pensive.generators.epsilon_machine

***********
ε-Machine
***********

An :class:`EpsilonMachine` is a unifilar Mealy HMM — the minimal causal
presentation of a stationary process :cite:`Crutchfield1994`.

.. math::

   h_\mu = \H{X_0 \mid X_{-\infty:0}}, \qquad
   C_\mu = \H{S^+}

Build from an HMM via partition refinement:

.. ipython::

   In [1]: from pensive.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   @doctest float
   In [3]: eps.entropy_rate()
   Out[3]: 0.6666666666666665

   @doctest float
   In [4]: eps.statistical_complexity()
   Out[4]: 0.9182958340544896

   @doctest
   In [5]: eps.markov_order()
   Out[5]: 1

See also :doc:`bidirectional_epsilon_machine`, :doc:`information_anatomy`, and
:doc:`epsilon_inference` (sample-based reconstruction).

API
===

.. autoclass:: EpsilonMachine
   :members: from_generator, from_sequence, from_time_reversed, statistical_complexity, excess_entropy, crypticity, markov_order, cryptic_order, is_exactly_synchronizable
