.. algorithms.rst
.. py:module:: pensive.automata.algorithms

**********
Algorithms
**********

Standard automata operations on :class:`~pensive.automata.base.LabeledAutomaton`
instances.

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
.. autofunction:: determinize
.. autofunction:: minimize
.. autofunction:: equivalent
.. autoclass:: MinimizationAlgorithm
