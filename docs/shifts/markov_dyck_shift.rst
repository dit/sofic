.. markov_dyck_shift.rst
.. py:module:: sofic.shifts.markov_dyck

*****************
Markov-Dyck Shift
*****************

A :class:`MarkovDyckShift` is the Markov-Dyck specialization of
:class:`~sofic.shifts.sofic_dyck.SoficDyckShift`. It can be built from a
matrix or from a directed graph, with graph constructors for both edge-type and
vertex-type Markov-Dyck shifts :cite:`Matsumoto2014`.

The matrix constructor uses the convention that ``A[i, j]`` allows call symbol
``i`` after call symbol ``j``. Symbols are role-tagged as ``("call", label)``
and ``("return", label)``.

API
===

.. autoclass:: MarkovDyckShift
   :members:
