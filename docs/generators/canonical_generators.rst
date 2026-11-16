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

Minimality within a class
-------------------------

The canonical RFSA is size-minimal among NFAs whose states accept unions of residuals (:cite:`Myers2014`,
Corollary 4.11; :cite:`Zetzsche2021`, Corollary 34). The analogue, with mixtures in place of unions, holds for
:func:`canonical_rfsa_hmm`: a generator of the process in which the future morph
:math:`h_g(w) = \Pr(w \mid G_0 = g)` of every state of positive stationary probability is a mixture of
causal-state morphs has at least as many states as :func:`canonical_rfsa_hmm`. By the Markov property

.. math::

   \Pr(\cdot \mid \text{past}) = \sum_g \Pr(G_0 = g \mid \text{past}) \, h_g,

so every causal-state morph lies in :math:`\operatorname{conv}\{h_g\}`. The hypothesis gives the reverse
inclusion, so the two polytopes coincide and each vertex is one of the :math:`h_g`; :func:`canonical_rfsa_hmm`
has one state per vertex and belongs to the class. Reversing time gives the same statement for
:func:`maximized_prime_atomaton_hmm`, with past morphs and reverse causal states. Generators outside the class can
be smaller. The argument is not from the literature.

The other canonical NFAs of :doc:`/automata/canonical_closures` have no analogue here: the distromaton and the
universal automaton are built from intersections of residuals, and mixtures of morphs have no meet. The minimal
xor automaton is a minimal linear representation over GF(2); its counterpart over the reals is
:func:`~sofic.generators.minimal_quasi_realization.minimal_quasi_realization`.

.. autofunction:: canonical_rfsa_hmm
.. autofunction:: maximized_prime_atomaton_hmm
.. autofunction:: atomaton_hmm
