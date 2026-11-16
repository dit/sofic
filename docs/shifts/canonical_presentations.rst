Canonical NFA presentations
===========================

.. currentmodule:: sofic.shifts.canonical_presentations

A sofic shift is determined by its factor language, which is factorial and extendable :cite:`LindMarcus1995`.
Any automaton for that language presents the shift once trimmed and restricted to the states on bi-infinite paths,
so the canonical NFAs of the factor language -- the átomaton :cite:`BrzozowskiTamm2014`, the canonical residual
finite-state automaton :cite:`Denis2002`, the maximized prime átomaton :cite:`MaarandTamm2022`, the distromaton
:cite:`Myers2014` and the universal automaton :cite:`LombardySakarovitch2008` -- give canonical, generally
non-resolving presentations beside the Fischer and Krieger covers. The residual-side states are follower sets that
are not unions of other follower sets; the distromaton's are intersections of follower sets that are not unions of
smaller intersections. The minimal xor automaton has no counterpart, since it accepts by parity of paths.

When every intersection of two follower sets is a union of follower sets
(:func:`follower_sets_intersection_closed`), the canonical RFSA of the factor language is a state-minimal NFA
(:cite:`Myers2014`, Theorem 4.4). A smallest presentation is essential, and an essential presentation is an NFA for
the factor language with every state initial and accepting, so the canonical RFSA presentation is then a
presentation with the fewest states. The golden mean shift is an example.

.. code-block:: python

   shift = machine.to_sofic_shift()
   shift.canonical_rfsa_presentation()
   shift.maximized_prime_atomaton_presentation()
   shift.atomaton_presentation()
   shift.distromaton_presentation()
   shift.universal_presentation()
   follower_sets_intersection_closed(shift)   # then canonical_rfsa_presentation is smallest

The stochastic analogues, which replace unions of residuals by mixtures of future morphs, are in
:mod:`sofic.generators.canonical_generators`.

.. autofunction:: canonical_rfsa_presentation
.. autofunction:: maximized_prime_atomaton_presentation
.. autofunction:: atomaton_presentation
.. autofunction:: distromaton_presentation
.. autofunction:: universal_presentation
.. autofunction:: follower_sets_intersection_closed
.. autofunction:: factor_language_nfa
