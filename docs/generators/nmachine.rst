.. nmachine.rst
.. py:module:: pensive.generators.nmachine

*********
n-Machine
*********

An :class:`NMachine` is a quasiprobabilistic generator with signed transition
weights used in computational mechanics beyond Shannon measures. The signed
matrix view is related to quasi-realizations and observable-operator models
:cite:`Jaeger2000`.

API
===

.. autoclass:: NMachine
   :members: from_epsilon_machine, collision_entropy, process_negativity

.. autofunction:: pensive.generators.nmachine_construction.build_nmachine
.. autofunction:: pensive.generators.nmachine_construction.coarse_grained_distribution
