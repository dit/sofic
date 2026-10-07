.. sofic_shift.rst
.. py:module:: sofic.shifts.sofic

***********
Sofic Shift
***********

A :class:`SoficShift` is a sofic subshift presented by a labeled graph
:cite:`Fischer1975,LindMarcus1995,Weiss1973`.

Topological entropy (bits per symbol) is the ``log2`` spectral radius of a
right-resolving presentation, built by subset construction when the given one is
not, so parallel edges with the same label count once.

The measure of maximal entropy (Parry measure) and its information anatomy
(``h_top = b_top + r_top``) are documented in :doc:`topological_anatomy`.

API
===

.. autoclass:: SoficShift
   :members: topological_entropy, parry_measure, topological_anatomy, markov_order, cryptic_order, reset_threshold, synchronizing_word, is_exactly_synchronizable, is_asymptotically_synchronizable, is_definite

.. autofunction:: sofic.shifts.algorithms.sofic_topological_entropy
