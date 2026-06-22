.. vpa.rst
.. py:module:: pensive.automata.vpa

*************************
Visibly Pushdown Automata
*************************

:class:`VisiblyPushdownAutomaton` partitions the alphabet into call, return,
and internal symbols. Call transitions push a stack symbol, return transitions
may either be guarded by a stack symbol or left unguarded as a wildcard over
ordinary stack entries, and internal transitions leave the stack untouched.
The model and its nested-word connection follow Alur and Madhusudan
:cite:`AlurMadhusudan2009`.

Canonical forms
===============

The VPA module includes four deterministic canonical forms:

``SingleEntryVisiblyPushdownAutomaton``
   A k-module SEVPA. Modules partition the states, calls are assigned to
   modules by ``call_partition``, every non-base module has one entry state in
   ``entry_states``, and every call pushes ``(caller_state, call_symbol)``.

``MultipleEntryVisiblyPushdownAutomaton``
   A k-module MEVPA. Calls are still assigned to modules, but a module may have
   several entries. The pushed call stack symbol must depend only on the source
   state.

``CallDrivenAutomaton``
   A CDA, used here as the shared modular generalization. The target of a call
   transition is determined by the call symbol, independent of the source state.

``CanonicalVisiblyPushdownAutomaton``
   The Myhill-Nerode canonical deterministic VPA. It is constructed from the
   finite algebra of well-matched summaries induced by a deterministic VPA and
   quotiented by finite right-context acceptance signatures. If the call
   alphabet is empty, this construction specializes to the usual minimal DFA
   right congruence.

The modular ``minimize`` constructors require deterministic input and fixed
module/call metadata. They intentionally do not attempt arbitrary VPA
minimization: visibly pushdown automata do not have unique minimum recognizers
in general, and exact unrestricted minimization is NP-complete.

Operations
==========

Finite automata expose the usual regular operations directly on ``DFA`` and
``NFA`` instances: ``union``, ``intersection``/``intersect``, ``complement``,
``difference``, ``concat``/``concatenate``, and ``kleene_star``/``star``.

VPAs expose the same operation names. These return
``CompositeVisiblyPushdownAutomaton`` instances, which are exact VPA language
expressions with a ``recognizes`` method. This keeps concatenation and Kleene
star correct even when an operand accepts with pending stack content; concrete
graph normalization for those composite VPAs is intentionally left separate
from the operation API.

Constructor sketch
==================

.. code-block:: python

   SingleEntryVisiblyPushdownAutomaton.minimize(
       vpa,
       call_partition={"call": "module"},
       modules={"main": {"q0"}, "module": {"entry", "body"}},
       entry_states={"module": "entry"},
   )

   MultipleEntryVisiblyPushdownAutomaton.minimize(
       vpa,
       modules={"main": {"q0"}, "module": {"entry0", "entry1"}},
       call_partition={"call0": "module", "call1": "module"},
   )

   CallDrivenAutomaton.minimize(
       vpa,
       modules={"main": {"q0"}, "module": {"entry"}},
       call_partition={"call": "module"},
   )

   CanonicalVisiblyPushdownAutomaton.from_vpa(vpa)

References
==========

The summary and SEVPA constructions follow Alur, Kumar, Madhusudan, and
Viswanathan :cite:`AlurKumarMadhusudanViswanathan2005`. The unrestricted
minimization limitation follows Gauwin, Muscholl, and Raskin
:cite:`Gauwin2020`.

API
===

.. autoclass:: VisiblyPushdownAutomaton
   :members: union, intersection, intersect, complement, difference, concat, concatenate, kleene_star, star

.. autoclass:: DeterministicVisiblyPushdownAutomaton

.. autoclass:: CompositeVisiblyPushdownAutomaton

.. autoclass:: SingleEntryVisiblyPushdownAutomaton

.. autoclass:: MultipleEntryVisiblyPushdownAutomaton

.. autoclass:: CallDrivenAutomaton

.. autoclass:: CanonicalVisiblyPushdownAutomaton

.. autofunction:: pensive.automata.vpa_simulation.recognizes_vpa
