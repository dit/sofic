.. atomaton.rst
.. py:module:: sofic.automata.atomaton

********
Átomaton
********

Atomic automata (:class:`AtomicAutomaton`, :class:`Atomaton`) and the
maximized prime átomaton (:class:`MaximizedPrimeAtomaton`) follow the regular
language atom and átomaton constructions :cite:`BrzozowskiTamm2011`
:cite:`BrzozowskiTamm2014`.

An *atom* is a non-empty intersection of complemented or uncomplemented left
quotients of the language.  Atoms partition the free monoid, every quotient is
a union of them, and the átomaton -- the NFA whose states are the atoms -- is
isomorphic to the reverse of the minimal DFA of the reverse language.  Atoms
therefore classify *futures* in the same way that the minimal DFA's states
classify *pasts*.

Atomicity
=========

An NFA is *atomic* when the right language of every state is a union of atoms,
which strictly generalizes the residual automata of :doc:`rfsa`.  Atomicity is
what makes the subset construction sharp: ``N.determinize()`` is minimal if and
only if ``N.reverse()`` is atomic :cite:`BrzozowskiTamm2014`, a theorem that
contains Brzozowski's double-reversal minimization :cite:`Brzozowski1962` as
the special case where the reverse is deterministic.

.. code-block:: python

    from sofic.automata.atomaton import Atomaton, atomic_states, is_atomic

    atomaton = Atomaton.from_language(dfa)
    is_atomic(atomaton)          # True
    atomic_states(nfa)           # states whose right language is a union of atoms
    is_atomic(nfa.reverse())     # iff nfa.determinize() is minimal

API
===

.. autofunction:: atomic_states
.. autofunction:: is_atomic

.. autoclass:: AtomicAutomaton
.. autoclass:: Atomaton
.. autoclass:: MaximizedPrimeAtomaton
