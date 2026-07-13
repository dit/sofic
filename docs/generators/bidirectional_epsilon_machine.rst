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
forward and reverse causal states (see :doc:`generative_models` for the full
family of constructions and their common-information ordering).

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

Functional Generative Models
============================

The functional variant uses the smallest *deterministic function*
:math:`G = f(S^+, S^-)` of the joint causal state that renders :math:`S^+` and
:math:`S^-` conditionally independent. Its entropy :math:`H[G]` is the
functional common information. Because :math:`G` is deterministic, the model's
state entropy equals that value exactly. The auxiliary is found by an exact
partition search rather than a stochastic optimizer, so this factory takes no
optimizer arguments.

.. ipython::

   In [9]: fgm = bidir.functional_generative_model()

   In [10]: fgm.generative_complexity() == fgm.functional_common_information
   Out[10]: True

Gács-Körner Generative Models
=============================

The Gács-Körner variant uses the *deterministic meet* :math:`S^+ \wedge S^-`
of the forward and reverse causal states — the largest random variable that is
simultaneously a function of both, obtained combinatorially as the connected
components of the joint support graph. Because the meet is deterministic there
is nothing to optimize, so this factory takes no optimizer arguments; the
model's state entropy equals the Gács-Körner common information
:math:`K[S^+ : S^-]` and keeps only the conserved "core" (phase /
ergodic-component structure), which is often trivial for mixing processes.

.. ipython::

   In [11]: ggm = bidir.gacs_korner_generative_model()

   In [12]: ggm.generative_complexity() == ggm.gk_common_information
   Out[12]: True

API
===

.. autoclass:: BidirectionalEpsilonMachine
   :members: from_forward, from_pair, joint_distribution, step_distribution, forward_epsilon_machine, reverse_epsilon_machine, entropy_rate, statistical_complexity, excess_entropy, crypticity, minimal_generative_model, wyner_generative_model, functional_generative_model, gacs_korner_generative_model, generative_complexity, predicted_information, bound_information, ephemeral_information, information_anatomy, caekl_causal_information
