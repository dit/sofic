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
.. autofunction:: pfa_to_mealy
.. autofunction:: hmm_to_sofic_shift
.. autofunction:: hmm_to_support_nfa
.. autofunction:: hmm_to_support_dfa
.. autofunction:: hmm_to_edge_machine
.. autofunction:: epsilon_machine_to_unifilar_graph
.. autofunction:: quasi_realization_from_nmachine
.. autofunction:: nmachine_from_quasi_realization
