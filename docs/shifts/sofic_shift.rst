.. sofic_shift.rst
.. py:module:: sofic.shifts.sofic

***********
Sofic Shift
***********

A :class:`SoficShift` is a sofic subshift presented by a labeled graph
:cite:`Fischer1975,LindMarcus1995,Weiss1973`.

The measure of maximal entropy (Parry measure) and its information anatomy
(``h_top = b_top + r_top``) are documented in :doc:`topological_anatomy`.

API
===

.. autoclass:: SoficShift
   :members: topological_entropy, parry_measure, topological_anatomy, markov_order, cryptic_order, reset_threshold, synchronizing_word, is_exactly_synchronizable, is_asymptotically_synchronizable, is_definite

.. autofunction:: sofic.shifts.tmc_construction.topological_entropy
