.. nfa.rst
.. py:module:: sofic.automata.nfa

***
NFA
***

A :class:`NFA` supports multiple initial states and ε-transitions. The
deterministic and nondeterministic presentations recognize the same regular
languages :cite:`RabinScott1959,HopcroftUllman1979`.

.. ipython::

   In [1]: from sofic.automata import NFA, determinize

   In [2]: nfa = NFA(
      ...:     input_alphabet=frozenset({0, 1}),
      ...:     initial_states=frozenset({"q0"}),
      ...:     accepting_states=frozenset({"q1"}),
      ...: )

   In [3]: nfa.graph.add_state("q0")

   In [4]: nfa.graph.add_state("q1")

   In [5]: nfa.add_transition("q0", "q1", symbol=0)

   In [6]: dfa = determinize(nfa)

API
===

.. autoclass:: NFA
   :members: add_transition, recognizes, union, intersection, intersect, complement, difference, concat, concatenate, kleene_star, star, determinize, minimize
