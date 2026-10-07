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

Decision procedures
===================

``is_empty`` and ``accepted_word`` search the state graph breadth-first, so the
witness is a shortest accepted word (ε-moves cost nothing). ``is_universal``
and ``includes`` use the forward antichain algorithms of De Wulf et al.
:cite:`DeWulf2006`, which explore the subset construction on the fly and keep
only ``⊆``-minimal subsets instead of determinizing. Universality is relative
to ``input_alphabet`` by default, matching :meth:`~NFA.complement`.

.. ipython::

   @doctest
   In [7]: nfa.accepted_word()
   Out[7]: (0,)

   @doctest
   In [8]: nfa.is_universal()
   Out[8]: False

   @doctest
   In [9]: dfa.includes(nfa)
   Out[9]: True

API
===

.. autoclass:: NFA
   :members: add_transition, recognizes, union, intersection, complement, difference, concat, kleene_star, determinize, minimize,
             is_empty, accepted_word, is_universal, includes
