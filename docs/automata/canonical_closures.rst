.. canonical_closures.rst
.. py:module:: sofic.automata.canonical.closures

**************************************************
Distromaton, xor automata, and universal automaton
**************************************************

A regular language has no unique state-minimal NFA, but it has several *canonical* ones. Myers, Adámek, Milius and
Urbat :cite:`Myers2014,Myers2015` obtain them all by one recipe: close the residuals :math:`u^{-1}L` under the
operations of an algebraic variety, which gives the minimal deterministic automaton of :math:`L` whose states form
an algebra of that variety (:func:`minimal_structured_dfa`), and take a minimal set of generators of that algebra
as the states of an NFA. Zetzsche, van Heerdt, Sammartino and Silva :cite:`Zetzsche2021` recast the second step as
generators for algebras over a monad and let the algebra of the deterministic automaton differ from the side effect
of the succinct one through distributive law homomorphisms, which brings in the átomaton and a new acceptor, the
minimal xor-CABA automaton.

==========================  ==========================  ======================  ===========================================
algebra of the DFA          closure of the residuals    states of the NFA       canonical automaton
==========================  ==========================  ======================  ===========================================
join-semilattice            unions                      join-irreducibles       canonical RFSA (*jiromaton*), :doc:`rfsa`
distributive lattice        unions, intersections       join-irreducibles       :class:`Distromaton`
Boolean algebra             Boolean operations          atoms                   átomaton, :doc:`atomaton`
GF(2) vector space          symmetric difference        a basis                 :func:`minimal_xor_automaton`
Boolean algebra as GF(2)    Boolean operations          a basis                 :func:`xor_caba_automaton`
==========================  ==========================  ======================  ===========================================

The universal automaton :cite:`LombardySakarovitch2008` (:class:`UniversalAutomaton`) keeps *every* intersection
of residuals; the distromaton is its subautomaton on the join-irreducible ones.

All constructions here are exact. Every language in these closures is a union of atoms of :math:`L`, and the atoms
are the states of the minimal complete DFA :math:`D` of :math:`L^R` :cite:`BrzozowskiTamm2014`, so a language is
stored as a set of states of :math:`D`: the residual :math:`u^{-1}L` is :math:`\{q : \delta(q, u^R) \in F\}`, the
derivative :math:`a^{-1}K` is the preimage :math:`\{q : \delta(q, a) \in K\}`, :math:`\varepsilon \in K` iff the
start state is in :math:`K`, and the lattice and vector-space operations are set operations.

Distromaton
===========

The distromaton's states are the intersections of residuals that are not unions of smaller intersections; a state
:math:`K` is initial iff :math:`K \subseteq L`, accepting iff :math:`\varepsilon \in K`, and
:math:`K \xrightarrow{a} K'` iff :math:`K' \subseteq a^{-1}K` :cite:`Myers2014`. There is one state per atom
(the intersection of the residuals containing it), so the distromaton and the átomaton have the same number of
states once the átomaton counts the negative atom (:cite:`Myers2014`, Corollary 3.21). Their transition structures
differ: the distromaton is the reverse of the minimal DFA of :math:`L^R` *ordered by inclusion*, with
:math:`z \xrightarrow{a} z'` whenever :math:`\delta(z', a)` includes :math:`z`, where the átomaton requires
equality.

Xor automata
============

A :class:`~sofic.automata.xor.XorAutomaton` is shaped like an NFA but accepts a word when it labels an odd number of
accepting paths :cite:`VuilleminGama2010`; it is a weighted automaton over GF(2), and
:meth:`~sofic.automata.xor.XorAutomaton.determinize` is the xor subset construction. The minimal xor automaton has
a basis of the span of the residuals as states. It is minimal among all xor automata and never larger than the
minimal DFA, but canonical only up to the basis; :func:`minimal_xor_automaton` picks the residuals greedily in
shortlex order of their access words. It is the GF(2) instance of the minimal linear representation that
:func:`~sofic.generators.minimal_quasi_realization.minimal_quasi_realization` computes over the reals.

The minimal xor-CABA automaton :cite:`Zetzsche2021` uses a basis of the whole Boolean closure, read as a GF(2)
space, so it has one state per atom. With ``basis="residuals"`` the basis extends that of the minimal xor
automaton, which is then its accessible part; with ``basis="atoms"`` it is the átomaton plus the negative atom,
because the átomaton is co-deterministic with one accepting state and so has at most one accepting path per word.

Minimality and state-minimal NFAs
=================================

Each canonical automaton is size-minimal in a class of acceptors (:cite:`Myers2014`, Section 4;
:cite:`Zetzsche2021`, Section 6):

========================  ===========================================================================
automaton                 has the fewest states among
========================  ===========================================================================
canonical RFSA            NFAs whose states accept unions of residuals
átomaton                  NFAs whose accepted languages (over all initial sets) are closed under complement
distromaton               NFAs whose accepted languages are closed under intersection
minimal xor automaton     all xor automata
minimal xor-CABA          xor automata whose accepted languages have equal GF(2) and Boolean closures
========================  ===========================================================================

When :func:`is_intersection_closed` holds -- every intersection of two residuals is a union of residuals -- the
canonical RFSA is a state-minimal NFA outright, and the distromaton and átomaton have at most one more state
(:cite:`Myers2014`, Theorem 4.4 and Corollary 4.6). For sofic shifts this certifies that the canonical RFSA
presentation is a smallest presentation
(:func:`~sofic.shifts.canonical_presentations.follower_sets_intersection_closed`).

.. code-block:: python

    from sofic.automata.canonical.closures import (
        Distromaton, minimal_xor_automaton, xor_caba_automaton, is_intersection_closed, minimal_structured_dfa,
    )

    Distromaton.from_language(nfa)
    xor = minimal_xor_automaton(nfa)          # an XorAutomaton
    xor.recognizes(word)                      # odd number of accepting paths
    xor.determinize()                         # DFA of the same language
    minimal_structured_dfa(nfa, "lattice")    # the distributive-lattice DFA the distromaton is read from
    is_intersection_closed(nfa)               # then the canonical RFSA is a state-minimal NFA

API
===

.. autoclass:: Distromaton
   :members: from_language
.. autofunction:: distromaton_from_language
.. autoclass:: UniversalAutomaton
   :members: from_language
.. autofunction:: universal_automaton_from_language
.. autofunction:: minimal_xor_automaton
.. autofunction:: xor_caba_automaton
.. autofunction:: minimal_structured_dfa
.. autofunction:: is_intersection_closed

.. autoclass:: sofic.automata.xor.XorAutomaton
   :members: from_language, recognizes, run, step, determinize, minimize, reverse
