.. unifilar_automaton.rst
.. py:module:: sofic.automata.unifilar

******************
Unifilar Automaton
******************

A :class:`UnifilarAutomaton` is right-resolving on input symbols — at most one
outgoing edge per symbol from each state. This is the automata-theoretic notion
of unifilarity; see :doc:`../generators/epsilon_machine` for the generator sense.
The terminology matches deterministic/right-resolving presentations used in
automata theory and symbolic dynamics :cite:`HopcroftUllman1979,LindMarcus1995`.

API
===

.. autoclass:: UnifilarAutomaton
   :members: markov_order, cryptic_order
