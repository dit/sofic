.. edge_machine.rst
.. py:module:: pensive.generators.edge_machine

*************
Edge Machine
*************

An **edge machine** (generator presentation) converts a non-unifilar HMM into a
Mealy generator whose states index labeled transitions :math:`(q, o)`.

.. ipython::

   In [1]: from pensive.generators.edge_machine import edge_machine_from_hmm
   In [2]: from pensive.examples import tent_map_misiurewicz_hmm

   In [3]: edge = edge_machine_from_hmm(tent_map_misiurewicz_hmm())

   @doctest
   In [4]: len(list(edge.states()))
   Out[4]: 7

API
===

.. autofunction:: edge_machine_from_hmm
.. autofunction:: edge_state_label
.. autofunction:: parse_edge_state_label
