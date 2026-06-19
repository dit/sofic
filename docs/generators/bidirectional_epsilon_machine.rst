.. bidirectional_epsilon_machine.rst
.. py:module:: pensive.generators.bidirectional_epsilon_machine

***************************
Bidirectional ε-Machine
***************************

A :class:`BidirectionalEpsilonMachine` is a non-unifilar generator over joint
causal states :math:`(S^+, S^-)` :cite:`Ellison2009`.

.. math::

   C_\pm = \H{S^+, S^-}, \qquad E = \I{S^+ : S^-}, \qquad \chi = C_\pm - E

.. ipython::

   In [1]: from pensive.examples import golden_mean_bidirectional

   In [2]: bidir = golden_mean_bidirectional(0.5)

   @doctest float
   In [3]: bidir.entropy_rate()
   Out[3]: 0.6666666666666665

   @doctest float
   In [4]: bidir.excess_entropy()
   Out[4]: 0.25162916738782304

API
===

.. autoclass:: BidirectionalEpsilonMachine
   :members: from_epsilon_machine, from_epsilon_machines, joint_distribution, step_distribution, marginalize_forward, marginalize_reverse, entropy_rate, statistical_complexity, excess_entropy, crypticity, predicted_information, bound_information, ephemeral_information, information_anatomy
