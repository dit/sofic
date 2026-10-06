.. rfsa.rst
.. py:module:: sofic.automata.canonical.rfsa

***
RFSA
***

Residual finite state automata (:class:`ResidualFiniteStateAutomaton`) are NFAs
whose every state accepts a residual (left quotient) of the language
:cite:`Denis2002`. :meth:`ResidualFiniteStateAutomaton.validate` checks this
exactly against the minimal DFA.

The canonical RFSA (:class:`CanonicalRFSA`) has one state per *prime*
residual -- a non-empty residual that is not the union of the residuals strictly
inside it -- with initial states the primes contained in the language, accepting
states the primes containing the empty word, and a transition
:math:`p \xrightarrow{a} p'` whenever :math:`L_{p'} \subseteq a^{-1} L_p`
:cite:`Denis2002`. It is never larger than the minimal DFA and can be
exponentially smaller: for :math:`\Sigma^* a \Sigma^n` the minimal DFA has
:math:`2^{n+1}` states and the canonical RFSA :math:`n + 2`.

Reversing a canonical RFSA gives the maximized prime átomaton of the reversed
language (:meth:`CanonicalRFSA.dual`; see :doc:`atomaton`). NL\* learns the
canonical RFSA from queries (:doc:`learning`).

.. code-block:: python

    from sofic.automata.canonical.rfsa import CanonicalRFSA

    rfsa = CanonicalRFSA.from_language(nfa)
    rfsa.validate()               # every state accepts a residual
    rfsa.dual()                   # maximized prime átomaton of the reverse

API
===

.. autoclass:: ResidualFiniteStateAutomaton
.. autoclass:: CanonicalRFSA
   :members: from_language, from_observation_table, dual

.. autofunction:: sofic.automata.canonical.residual.canonical_rfsa_from_language
.. autoclass:: sofic.automata.canonical.residual.ResidualTable
   :members: includes, is_covered, prime_states
