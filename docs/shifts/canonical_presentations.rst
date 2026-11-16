Canonical NFA presentations
===========================

.. currentmodule:: sofic.shifts.canonical_presentations

A sofic shift is determined by its factor language, which is factorial and extendable :cite:`LindMarcus1995`.
Any automaton for that language with every state accepting presents the shift once trimmed to the states on
bi-infinite paths, so the canonical NFAs of the factor language -- the átomaton :cite:`BrzozowskiTamm2014`, the
canonical residual finite-state automaton :cite:`Denis2002` and the maximized prime átomaton
:cite:`MaarandTamm2022` -- give canonical, generally non-resolving presentations beside the Fischer and Krieger
covers. The residual-side states are follower sets that are not unions of other follower sets.

.. code-block:: python

   shift = machine.to_sofic_shift()
   shift.canonical_rfsa_presentation()
   shift.maximized_prime_atomaton_presentation()
   shift.atomaton_presentation()

The stochastic analogues, which replace unions of residuals by mixtures of future morphs, are in
:mod:`sofic.generators.canonical_generators`.

.. autofunction:: canonical_rfsa_presentation
.. autofunction:: maximized_prime_atomaton_presentation
.. autofunction:: atomaton_presentation
.. autofunction:: factor_language_nfa
