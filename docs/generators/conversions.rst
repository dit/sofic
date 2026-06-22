.. conversions.rst
.. py:module:: pensive.generators.conversions

***********
Conversions
***********

Convert between equivalent presentations of stochastic generators. The
conversions connect Mealy/Moore HMMs, edge presentations, ε-machines, and
quasi-realizations :cite:`Mealy1955,Moore1956,Rabiner1989,Jaeger2000`.

API
===

.. autofunction:: moore_to_mealy
.. autofunction:: pfa_to_mealy_hmm
.. autofunction:: edge_machine_from_hmm
.. autofunction:: epsilon_machine_to_unifilar_graph
.. autofunction:: quasi_realization_from_nmachine
.. autofunction:: nmachine_from_quasi_realization
