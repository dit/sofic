.. bidirectional_epsilon_machine.rst
.. py:module:: pensive.generators.bidirectional_epsilon_machine

***************************
Bidirectional ε-Machine
***************************

A :class:`BidirectionalEpsilonMachine` is a non-unifilar generator over joint
causal states :math:`(S^+, S^-)` :cite:`Ellison2011`.

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

Minimal Generative Models
=========================

The bidirectional machine can be reduced to a non-unifilar minimal
generative model by optimizing the exact common information between the
forward and reverse causal states. This requires the optional
``pensive[measures]`` dependencies.

.. ipython::

   In [5]: mgm = bidir.minimal_generative_model()

   In [6]: mgm.state_entropy() > 0
   Out[6]: True

Wyner Generative Models
=======================

The same construction can use the Wyner-common-information auxiliary between
the forward and reverse causal states. The optimized value is
:math:`I[(S^+, S^-) : G]`, exposed as ``wyner_common_information``; the model's
state entropy is :math:`H[G]` and can be larger.

.. ipython::

   In [7]: wgm = bidir.wyner_generative_model()

   In [8]: wgm.wyner_common_information <= wgm.state_entropy()
   Out[8]: True

API
===

.. autoclass:: BidirectionalEpsilonMachine
   :members: from_forward, from_pair, joint_distribution, step_distribution, forward_epsilon_machine, reverse_epsilon_machine, entropy_rate, statistical_complexity, excess_entropy, crypticity, minimal_generative_model, wyner_generative_model, generative_complexity, predicted_information, bound_information, ephemeral_information, information_anatomy
