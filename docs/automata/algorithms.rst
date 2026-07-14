.. algorithms.rst
.. py:module:: pensive.automata.algorithms

**********
Algorithms
**********

Standard automata operations on :class:`~pensive.automata.base.LabeledAutomaton`
instances. The implementations cover subset construction, DFA equivalence and
minimization, Brzozowski double reversal, Moore refinement, Hopcroft refinement,
and state-elimination conversion to regular expressions
:cite:`RabinScott1959,Brzozowski1962,Moore1956,Hopcroft1971,Kleene1956`.

.. ipython::

   In [1]: from pensive.automata import DFA, trim, minimize, equivalent

   In [2]: dfa = DFA(
      ...:     input_alphabet=frozenset({0}),
      ...:     initial_states=frozenset({"q0"}),
      ...:     accepting_states=frozenset({"q0"}),
      ...: )

   In [3]: dfa.graph.add_state("q0")

   In [4]: dfa.add_transition("q0", "q0", symbol=0)

   In [5]: trimmed = trim(dfa)

   @doctest
   In [6]: equivalent(dfa, trimmed, frozenset({0}))
   Out[6]: True

API
===

.. autofunction:: trim
.. autofunction:: complete
.. autofunction:: reverse
.. autofunction:: determinize
.. autofunction:: minimize
.. autofunction:: minimize_hopcroft
.. autofunction:: minimize_moore
.. autofunction:: minimize_brzozowski
.. autofunction:: equivalent
.. autofunction:: pensive.automata.regex.automaton_to_regex

.. autodata:: MinimizationAlgorithm
