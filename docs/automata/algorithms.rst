.. algorithms.rst
.. py:module:: sofic.automata.algorithms

**********
Algorithms
**********

Standard automata operations on :class:`~sofic.automata.base.LabeledAutomaton`
instances. The implementations cover subset construction, DFA equivalence and
minimization, Brzozowski double reversal, Moore refinement, Hopcroft refinement,
and state-elimination conversion to regular expressions
:cite:`RabinScott1959,Brzozowski1962,Moore1956,Hopcroft1971,Kleene1956`.
All three minimizers return the same trimmed (partial) minimal DFA; use
:func:`complete` for the version with a sink state.

.. ipython::

   In [1]: from sofic.automata import DFA, trim, minimize, equivalent

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

Regular expressions
===================

:func:`~sofic.automata.regex.automaton_to_regex` converts an automaton to a
regular expression by state elimination :cite:`Kleene1956,HopcroftUllman1979`,
and :func:`~sofic.automata.regex.regex_to_nfa` (also
:meth:`NFA.from_regex <sofic.automata.nfa.NFA.from_regex>`) converts back with
Thompson's construction :cite:`Thompson1968`, producing an ε-NFA with one
initial and one accepting state. The two read and write the same dialect, which
is Python :mod:`re` syntax whenever every symbol is a single-character string:

* a single character is a symbol; ``\c`` escapes any character ``c``;
* ``'ab'`` quotes a multi-character symbol (``\'`` and ``\\`` escape inside the
  quotes), so ``'ab'`` is one symbol while ``ab`` is two;
* ``ε`` or ``(?:)`` is the empty word, ``∅`` or ``(?!)`` the empty language;
* ``|``, juxtaposition, and postfix ``*``, ``+``, ``?`` are alternation,
  concatenation, and repetition; ``(...)`` and ``(?:...)`` group;
* ``. [ ] { } ^ $`` are reserved and must be escaped.

Malformed input (unbalanced parentheses, dangling operators, empty
alternatives) raises :class:`~sofic.exceptions.RegexSyntaxError`, a
:class:`ValueError`. Passing ``alphabet`` resolves each parsed symbol to the
alphabet element with the same ``str``, so automata over non-string symbols
round-trip:

.. code-block:: python

   from sofic.automata import NFA, automaton_to_regex, equivalent, regex_to_nfa

   nfa = NFA.from_regex("'ab'(a|b)*")
   assert nfa.recognizes(("ab", "a", "b"))
   assert equivalent(nfa, regex_to_nfa(automaton_to_regex(nfa), alphabet=nfa.input_alphabet))

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
.. autofunction:: nerode_partition
.. autofunction:: equivalent
.. autofunction:: sofic.automata.regex.automaton_to_regex
.. autofunction:: sofic.automata.regex.parse_regex
.. autofunction:: sofic.automata.regex.regex_to_nfa
.. autoclass:: sofic.automata.regex.Regex
   :members: pattern
.. autoexception:: sofic.exceptions.RegexSyntaxError

.. autodata:: MinimizationAlgorithm
