.. dfa.rst
.. py:module:: sofic.automata.dfa

***
DFA
***

A :class:`DFA` is a deterministic finite automaton with a single initial state
and no ε-transitions, following the standard finite-automata model
:cite:`RabinScott1959,HopcroftUllman1979`.

.. ipython::

   In [1]: from sofic.automata import DFA

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

   @doctest
   In [9]: dfa.accepted_word()
   Out[9]: (0,)

   @doctest
   In [10]: dfa.is_universal()
   Out[10]: False

The emptiness, shortest-witness, universality, and inclusion checks are shared
with :class:`~sofic.automata.nfa.NFA`; see that page for the antichain
algorithms :cite:`DeWulf2006`.

API
===

.. autoclass:: DFA
   :members: add_transition, recognizes, union, intersection, complement, difference, concat, kleene_star, minimize, from_nfa,
             is_empty, accepted_word, is_universal, includes
