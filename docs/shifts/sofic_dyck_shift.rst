.. sofic_dyck_shift.rst
.. py:module:: sofic.shifts.sofic_dyck

****************
Sofic-Dyck Shift
****************

A :class:`SoficDyckShift` is a shift presented by a finite Dyck automaton:
edges are partitioned into call, return, and internal roles, and a matching
relation specifies which call edges may be paired with which return edges.
Finite factors are enumerated with visibly pushdown stack semantics
:cite:`BealBlockeletDima2015`.

The topological entropy API used for finite graph presentations is intentionally
not exposed for Dyck shifts, since stack constraints are not captured by the
ordinary adjacency matrix.

API
===

.. autoclass:: SoficDyckShift
   :members:

.. autofunction:: transition_ref

.. autofunction:: sofic.shifts.dyck_algorithms.is_admissible_word

.. autofunction:: sofic.shifts.dyck_algorithms.admissible_words
