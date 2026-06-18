.. unifilar_automaton.rst
.. py:module:: pensive.automata.unifilar

******************
Unifilar Automaton
******************

A :class:`UnifilarAutomaton` is right-resolving on input symbols — at most one
outgoing edge per symbol from each state. This is the automata-theoretic notion
of unifilarity; see :doc:`../generators/epsilon_machine` for the generator sense.

API
===

.. autoclass:: UnifilarAutomaton
   :members: markov_order, cryptic_order
