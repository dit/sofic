.. symbolic_hmm.rst

*************************
Symbolic HMM probabilities
*************************

In addition to floating-point transition probabilities, :mod:`pensive` can carry
exact `sympy <https://www.sympy.org>`_ expressions on HMM edges. Stationary
distributions, mixed-state presentations, bidirectional constructions, and
information anatomy then return sympy expressions that can be substituted and
simplified exactly.

Install the optional extra::

   pip install pensive[symbolic]

Building a parametric machine
==============================

Pass a sympy symbol as the control parameter to the tent-map examples (or to
:meth:`~pensive.generators.mealy.MealyHMM.add_transition` directly):

.. ipython::

   In [1]: import sympy as sp

   In [2]: from pensive.examples import tent_map_misiurewicz_forward

   In [3]: a = sp.symbols("a", positive=True)

   In [4]: eps = tent_map_misiurewicz_forward(a)

   In [5]: any(hasattr(t.data["prob"], "free_symbols") for t in eps.transitions())
   Out[5]: True

Appendix pipeline (Chaos Forgets)
=================================

James, Burke & Crutchfield (2013) compute closed-form ephemeral / bound rates
for the tent map at a Misiurewicz parameter via the bidirectional ε-machine
:cite:`James2013`. With symbolic probabilities the same pipeline returns a
sympy expression that evaluates to the supplement's numerical value:

.. ipython::

   In [1]: import sympy as sp

   In [2]: from pensive.examples import (
   ...:     tent_map_misiurewicz_a,
   ...:     tent_map_misiurewicz_bidirectional,
   ...:     tent_map_misiurewicz_hmm,
   ...:     tent_map_misiurewicz_information_expected,
   ...: )
   ...: from pensive.generators.epsilon_machine import EpsilonMachine

   In [3]: a = sp.symbols("a", positive=True)

   # Published Fig. 6 (right) non-unifilar HMM → ε-machine (Fig. 7)
   In [4]: eps = EpsilonMachine.from_hmm(tent_map_misiurewicz_hmm())

   # Bidirectional machine (Fig. 8) with symbolic ``a``
   In [5]: bidir = tent_map_misiurewicz_bidirectional(a)

   In [6]: r = bidir.ephemeral_information()

   In [7]: a_num = tent_map_misiurewicz_a()

   @doctest float
   In [8]: float(r.subs(a, a_num))
   Out[8]: 0.648257836793515

   @doctest float
   In [9]: tent_map_misiurewicz_information_expected(a_num)["ephemeral_mu"]
   Out[9]: 0.648257836793515

The supplement's rational-in-``a`` formula
``r_μ = (1/4)(3 - 2/(a+1) - 4/(a+2) + 9/(2a+3))`` is recovered by
:func:`~pensive.examples.epsilon_machines.tent_map_misiurewicz_information_expected`
and agrees with the machine-derived rate at the Misiurewicz root of ``a``.

:func:`~pensive.examples.epsilon_machines.tent_map_misiurewicz_hmm` is the
published 4-state Fig.~6 (right) topology (non-unifilar at ``A`` and ``D``).
Symbolic machines also draw: Graphviz / TikZ labels use
:func:`~pensive.viz._format.format_prob_label` so expressions like ``a/(a+1)``
appear on edges instead of raising on ``float(...)``.

The machine carries the Misiurewicz minimal polynomial ``a**3 - 2*a - 2`` as
:class:`~pensive.generators.prob.SymbolConstraints`; belief de-duplication and
causal-state merging then compare symbolic probabilities *modulo* that relation
(in the residue field ``Q[a]/(a**3 - 2*a - 2)``), so
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.from_hmm` recovers the
4-state Fig.~7 machine.  A model without constraints keeps the two states that
coincide only at the Misiurewicz root distinct.

Probability helpers
===================

Low-level coercion and comparison live in :mod:`pensive.generators.prob`:

* :func:`~pensive.generators.prob.as_prob` — store floats or exact sympy Expr
* :func:`~pensive.generators.prob.is_symbolic` / :func:`~pensive.generators.prob.has_symbolic`
* :func:`~pensive.generators.prob.simplify_prob`, :func:`~pensive.generators.prob.probs_equal`
* :class:`~pensive.generators.prob.SymbolConstraints` — compare probabilities modulo algebraic side-relations

See also :doc:`information_anatomy` and :doc:`mixed_state_presentation`.
