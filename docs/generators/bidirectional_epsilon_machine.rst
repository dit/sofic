.. bidirectional_epsilon_machine.rst
.. py:module:: sofic.generators.bidirectional_epsilon_machine

***************************
Bidirectional ε-Machine
***************************

A :class:`BidirectionalEpsilonMachine` is a non-unifilar generator over joint
causal states :math:`(S^+, S^-)` :cite:`Ellison2011`.

.. math::

   C_\pm = \H{S^+, S^-}, \qquad E = \I{S^+ : S^-}, \qquad \chi = C_\pm - E

.. ipython::

   In [1]: from sofic.examples import golden_mean_bidirectional

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

Structural / Gauge Anatomy Refinement
=====================================

The information anatomy :cite:`James2013` splits the entropy rate into predicted
(:math:`\rho_\mu`), bound (:math:`b_\mu`), and ephemeral (:math:`r_\mu`) rates.
Because the next forward causal state :math:`S^+_1` is a deterministic function
of :math:`S^+_0` and :math:`X_0`, each generated rate splits further into a
*structural* part (randomness that selects a different next causal state, i.e.
an edge with structural consequence) and a *gauge* / parallel-edge part
(output relabeling on edges from the same state to the same state):

.. math::

   r_\mu &= \underbrace{\H{S^+_1 \mid S^+_0, S^-_1}}_{\text{structural}}
          + \underbrace{\H{X_0 \mid S^+_0, S^+_1, S^-_1}}_{\text{gauge}} \\
   b_\mu &= \underbrace{\I{S^+_1 : S^-_1 \mid S^+_0}}_{\text{structural}}
          + \underbrace{\I{X_0 : S^-_1 \mid S^+_0, S^+_1}}_{\text{gauge}}

The two structural atoms sum to :math:`\H{S^+_1 \mid S^+_0}` (the rate of
genuine next-state decisions) and the two gauge atoms to
:math:`\H{X_0 \mid S^+_0, S^+_1}` (the parallel-edge relabeling rate); together
they recover :math:`h_\mu`. This refinement has no separate canonical source; it
follows from the determinism of the forward transition function.

.. ipython::

   In [13]: from sofic.examples import butterfly_process

   In [14]: bidir = butterfly_process().to_bidirectional()

   @doctest float
   In [15]: bidir.structural_ephemeral_information()
   Out[15]: 2.25

   @doctest float
   In [16]: bidir.parallel_edge_information()
   Out[16]: 0.75

API
===

.. autoclass:: BidirectionalEpsilonMachine
   :members: from_forward, from_pair, joint_distribution, step_distribution, forward_epsilon_machine, reverse_epsilon_machine, entropy_rate, statistical_complexity, excess_entropy, crypticity, minimal_generative_model, wyner_generative_model, functional_generative_model, gacs_korner_generative_model, generative_complexity, predicted_information, bound_information, ephemeral_information, structural_ephemeral_information, parallel_edge_information, bound_structural_information, bound_parallel_edge_information, information_anatomy, caekl_causal_information
