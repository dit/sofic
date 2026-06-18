.. dfa.rst
.. py:module:: pensive.automata.dfa

***
DFA
***

A :class:`DFA` is a deterministic finite automaton with a single initial state
and no ε-transitions.

.. ipython::

   In [1]: from pensive.automata import DFA

   In [2]: dfa = DFA(
      ...:     input_alphabet=frozenset({0, 1}),
      ...:     initial_states=frozenset({"q0"}),
      ...:     accepting_states=frozenset({"q1"}),
      ...: )

   In [3]: dfa.graph.add_state("q0")

   In [4]: dfa.graph.add_state("q1")

   In [5]: dfa.add_transition("q0", "q1", symbol=0)

   In [6]: dfa.add_transition("q0", "q0", symbol=1)

   @doctest
   In [7]: dfa.recognizes((0,))
   Out[7]: True

   @doctest
   In [8]: dfa.recognizes(())
   Out[8]: False

API
===

.. autoclass:: DFA
   :members: add_transition, recognizes, minimize, from_nfa
