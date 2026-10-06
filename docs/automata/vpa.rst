.. vpa.rst
.. py:module:: sofic.automata.vpa

*************************
Visibly Pushdown Automata
*************************

:class:`VisiblyPushdownAutomaton` partitions the alphabet into call, return,
and internal symbols :cite:`AlurMadhusudan2009`. Call transitions push a stack
symbol, internal transitions leave the stack alone, and return transitions pop
it. A return guarded by a stack symbol fires only when that symbol is on top; a
wildcard return (no stack symbol) fires on every stack symbol, and also on the
empty stack when the VPA has a ``bottom_stack_symbol``. A return on the empty
stack is a *pending return*; it is possible only with a bottom symbol and leaves
the stack empty. Acceptance is by final state, so words may end with *pending
calls* still on the stack.

Operations and decisions
========================

Every closure operation returns a concrete automaton:

* ``union`` (disjoint sum) and ``intersection`` (synchronized product);
* ``determinize`` -- the summary construction of :cite:`AlurMadhusudan2009`,
  whose states pair a summary relation with the set of current states. The
  result is a complete :class:`DeterministicVisiblyPushdownAutomaton`, and
  :meth:`DeterministicVisiblyPushdownAutomaton.from_vpa` uses it whenever its
  input is nondeterministic;
* ``complement`` (determinize, then flip accepting states) and ``difference``;
* ``concat`` and ``kleene_star``. Each factor is read from an empty stack of its
  own: the finite control records whether the current factor's stack is empty
  and pushes that bit with every symbol, so a return that would pop a pending
  call of an earlier factor counts as a pending return of the current one.

Emptiness is decided by saturating the relation of well-matched summaries and
then searching states reachable with pending calls or pending returns;
``accepted_word`` returns a witness. ``is_universal``, ``includes``,
``equivalent``, and ``has_unmatched_word`` build on it.
:class:`~sofic.automata.nwa.NestedWordAutomaton` exposes the same operations by
delegating through :meth:`~sofic.automata.nwa.NestedWordAutomaton.to_vpa`.

.. code-block:: python

   balanced.union(other).equivalent(other.union(balanced))   # True
   balanced.complement().complement().equivalent(balanced)   # True
   balanced.concat(balanced).accepted_word()                 # e.g. ('(', ')')

Canonical and modular forms
===========================

General visibly pushdown languages have no unique minimal deterministic VPA, and
exact unrestricted minimization is NP-complete :cite:`Gauwin2020`. Canonical
forms exist for well-matched languages, or once calls are assigned to modules
:cite:`AlurKumarMadhusudanViswanathan2005`.

``CanonicalVisiblyPushdownAutomaton``
   The Myhill-Nerode canonical deterministic VPA of a well-matched language.
   Its states are classes of the finite algebra of well-matched summaries,
   refined jointly for top-level contexts and for contexts inside a pending
   call. With an empty call alphabet it is the minimal DFA. Languages with a
   pending call or return raise
   :exc:`~sofic.exceptions.NonWellMatchedLanguageError`.

``SingleEntryVisiblyPushdownAutomaton``
   A k-module SEVPA. Modules partition the states, calls are assigned to
   modules by ``call_partition``, every non-base module has one entry state, and
   every call pushes ``(caller_state, call_symbol)``.

``MultipleEntryVisiblyPushdownAutomaton``
   A k-module MEVPA. A module may have several entries, and the pushed symbol
   depends only on the caller state.

``CallDrivenAutomaton``
   The shared modular generalization: a call's target depends only on the call
   symbol.

:func:`~sofic.automata.vpa.operations.to_single_entry` and
:func:`~sofic.automata.vpa.operations.to_multiple_entry` convert any VPA of a
well-matched language into these forms (default: one module per call symbol).
On a call the state resets to the module's entry and the caller is pushed, so
the automaton forgets its caller; that is why pending calls -- and hence
non-well-matched languages, which raise
:exc:`~sofic.exceptions.NonWellMatchedLanguageError` -- are out of scope. The
modular ``minimize`` constructors call these conversions when no modules are
given, and otherwise quotient the supplied module structure.

.. code-block:: python

   SingleEntryVisiblyPushdownAutomaton.minimize(vpa)                     # convert, then minimize
   SingleEntryVisiblyPushdownAutomaton.minimize(
       sevpa, call_partition={"call": "module"}, modules=sevpa.modules,
   )
   CanonicalVisiblyPushdownAutomaton.from_vpa(vpa)

API
===

.. autoclass:: VisiblyPushdownAutomaton
   :members: union, intersection, complement, difference, concat, kleene_star, determinize, is_empty,
             accepted_word, is_universal, includes, equivalent, has_unmatched_word

.. autoclass:: DeterministicVisiblyPushdownAutomaton
   :members: from_vpa

.. autoclass:: SingleEntryVisiblyPushdownAutomaton
   :members: minimize

.. autoclass:: MultipleEntryVisiblyPushdownAutomaton
   :members: minimize

.. autoclass:: CallDrivenAutomaton
   :members: minimize

.. autoclass:: CanonicalVisiblyPushdownAutomaton
   :members: from_vpa

.. automodule:: sofic.automata.vpa.operations
   :members: normalize, determinize, complement, concat, kleene_star, well_matched_summaries, accepted_word,
             has_unmatched_word, to_single_entry, to_multiple_entry

.. autofunction:: sofic.automata.vpa.simulation.recognizes_vpa
