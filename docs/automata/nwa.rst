.. nwa.rst
.. py:module:: pensive.automata.nwa

*********************
Nested Word Automata
*********************

:class:`NestedWordAutomaton` recognizes nested words: finite words whose input
also carries a call, return, or internal role for each position and an explicit
call-return matching relation. This differs from
:class:`~pensive.automata.vpa.VisiblyPushdownAutomaton`, where the alphabet
partition itself determines which positions are calls and returns. Nested words,
nested-word automata, and visibly pushdown languages are introduced by Alur and
Madhusudan :cite:`AlurMadhusudan2009`.

The same symbol may appear in several NWA role alphabets. The role belongs to
the :class:`NestedWord` input position, so overlapping call, return, and
internal alphabets are valid.

Constructor sketch
==================

.. code-block:: python

   from pensive.automata import NestedWord, NestedWordAutomaton

   nwa = NestedWordAutomaton(
       call_alphabet=frozenset({"("}),
       return_alphabet=frozenset({")"}),
       internal_alphabet=frozenset({"i"}),
       hier_alphabet=frozenset({"S"}),
       initial_state="q",
       accepting_states=frozenset({"q"}),
   )
   nwa.graph.add_state("q")
   nwa.add_call_transition("q", "q", "(", "S")
   nwa.add_return_transition("q", "q", ")", "S")
   nwa.add_internal_transition("q", "q", "i")

   word = NestedWord.from_visible_word(
       ("(", "i", ")"),
       call_alphabet={"("},
       return_alphabet={")"},
       internal_alphabet={"i"},
   )
   nwa.recognizes(word)

VPA interop
===========

Use :meth:`NestedWordAutomaton.from_vpa` to view a VPA as an NWA over the same
visible roles. Use :meth:`NestedWordAutomaton.to_vpa` to encode an NWA as a VPA;
by default symbols are tagged with their role so overlapping NWA alphabets still
become a disjoint visible alphabet.

API
===

.. autoclass:: NestedWord
   :members: from_visible_word, validate

.. autoclass:: NestedWordAutomaton
   :members: add_call_transition, add_return_transition, add_internal_transition, recognizes, recognizes_visible, from_vpa, to_vpa

.. autofunction:: pensive.automata.nwa_simulation.recognizes_nwa
