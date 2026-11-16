Canonical NFAs as generators
============================

.. currentmodule:: sofic.generators.canonical_generators

A regular language has several canonical automata :cite:`Myers2015`: the minimal DFA, the átomaton
:cite:`BrzozowskiTamm2014`, the canonical residual finite-state automaton (RFSA) :cite:`Denis2002`, and the
maximized prime átomaton :cite:`MaarandTamm2022`. For a stationary process with a finite bidirectional
ε-machine :cite:`Ellison2009`, each has a stochastic analogue that generates the process:

============================== ===================================================== ======================
canonical NFA                  generator                                             states
============================== ===================================================== ======================
minimal DFA                    forward ε-machine                                     :math:`S^+`
átomaton                       reverse ε-machine run forward (:func:`atomaton_hmm`)  :math:`S^-`
canonical RFSA                 :func:`canonical_rfsa_hmm`                            at most :math:`|S^+|`
maximized prime átomaton       :func:`maximized_prime_atomaton_hmm`                  at most :math:`|S^-|`
============================== ===================================================== ======================

A forward causal state's future morph is a mixture of the futures given the reverse causal states, which have
disjoint supports, with weights :math:`R_{s^+} = \Pr(S^- \mid S^+ = s^+)`. A morph that is not a mixture of the
other morphs -- the convex analogue of a prime residual, as in probabilistic residual automata
:cite:`Esposito2002` -- is therefore a vertex of :math:`\operatorname{conv}\{R_{s^+}\}`. The canonical-RFSA
analogue is the generator on those vertices: every causal state is a mixture of them, and the vertices'
transitions follow. The maximized-prime-átomaton analogue is the same construction applied to the reversed
process, on the vertices of the forward beliefs :math:`\operatorname{conv}\{Q_{s^-}\}`, time-reversed back. Both
are linear programs; both are in general not unifilar. The identification with the canonical NFAs is an
analogy and is not from the literature.

They are generators, so their state entropies are upper bounds on the generative complexity :cite:`LohrAy2009`,
and :func:`~sofic.generators.generative_search.generative_model` offers them as candidates and as starts of
its coupled search. For random-parameter ε-machines the RFSA analogue rarely differs from the ε-machine
(it needs the process rank to fall below :math:`|S^+|`); the prime-átomaton analogue often compresses a large
reverse machine to a few vertices, and on binary three-state ε-machines it is sometimes the smallest generator
known.

.. code-block:: python

   from sofic.generators import maximized_prime_atomaton_hmm

   prime = maximized_prime_atomaton_hmm(machine)   # a MealyHMM on the extreme reverse causal states
   prime.to_sofic_shift()                          # its support presents the same shift

The topological counterparts for sofic shifts are in :mod:`sofic.shifts.canonical_presentations`.

.. autofunction:: canonical_rfsa_hmm
.. autofunction:: maximized_prime_atomaton_hmm
.. autofunction:: atomaton_hmm
