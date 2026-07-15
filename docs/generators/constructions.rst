.. constructions.rst

*************
Constructions
*************

High-level builders for causal and bidirectional presentations. The ε-machine
builder combines mixed-state unifilarization with causal-state minimization,
and the bidirectional builder follows the forward/reverse construction
:cite:`Ellison2009,Loomis2019,Ellison2011`.

ε-Machine construction
======================

.. autofunction:: sofic.generators.epsilon_construction.build_epsilon_machine

Mixed-state presentation
========================

.. autofunction:: sofic.generators.mixed_state_construction.build_mixed_state_presentation
   :no-index:

Bidirectional ε-machine
=========================

.. autofunction:: sofic.generators.bidirectional_construction.build_bidirectional_epsilon_machine
.. autofunction:: sofic.generators.bidirectional_construction.infer_reverse_epsilon_machine
.. autofunction:: sofic.generators.bidirectional_construction.joint_distribution
.. autofunction:: sofic.generators.bidirectional_construction.forward_epsilon_machine
.. autofunction:: sofic.generators.bidirectional_construction.reverse_epsilon_machine

n-Machine construction
======================

.. autofunction:: sofic.generators.nmachine_construction.build_nmachine
   :no-index:
