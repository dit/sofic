.. dyck_enumeration.rst
.. py:module:: sofic.shifts.dyck_enumeration

****************
Dyck Enumeration
****************

Small :class:`~sofic.shifts.sofic_dyck.SoficDyckShift` topologies can be
enumerated exhaustively via a canonical string encoding, analogous to the
topological ε-machine enumeration of :doc:`../generators/topological_epsilon_enumeration`
and finitary-process enumeration :cite:`Johnson2010,BealBlockeletDima2015`.

A :class:`DyckGraphString` is a canonical, hashable encoding of a small
matched-edge Dyck graph: its state count, call/return/internal symbols,
transitions, and matched call/return pairs. Every enumerable topology has a
unique canonical string, so the iterators emit each shift exactly once.

.. code-block:: python

   from sofic.shifts import iter_sofic_dyck_topologies, count_dyck_graph_strings

   n = count_dyck_graph_strings(n=1, call_symbols=("a",), return_symbols=("A",))

   for shift in iter_sofic_dyck_topologies(n=1, call_symbols=("a",), return_symbols=("A",)):
       shift.validate()

Round-trip a shift through its canonical encoding with
:func:`shift_to_dyck_graph_string` and :func:`dyck_graph_string_to_shift`.

API
===

.. autoclass:: DyckGraphString

.. autofunction:: shift_to_dyck_graph_string
.. autofunction:: dyck_graph_string_to_shift
.. autofunction:: iter_dyck_graph_strings
.. autofunction:: iter_sofic_dyck_topologies
.. autofunction:: count_dyck_graph_strings

.. autoexception:: DyckEnumerationError
