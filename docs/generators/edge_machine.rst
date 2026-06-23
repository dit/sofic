.. edge_machine.rst
.. py:module:: pensive.generators.edge_machine

*************
Edge Machine
*************

An **edge machine** (generator presentation) converts a non-unifilar HMM into a
Mealy generator whose states index labeled transitions :math:`(q, o, q')`.
This is a presentation-level construction for hidden Markov generators and
computational mechanics :cite:`Rabiner1989,Crutchfield1994`.

.. ipython::

   In [1]: from pensive.examples import tent_map_misiurewicz_hmm

   In [2]: edge = tent_map_misiurewicz_hmm().to_edge_machine()

   @doctest
   In [3]: len(list(edge.states()))
   Out[3]: 7

API
===

.. autofunction:: hmm_to_edge_machine
.. autofunction:: edge_state_label
.. autofunction:: parse_edge_state_label
