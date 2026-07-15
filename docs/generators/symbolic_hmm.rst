.. symbolic_hmm.rst

**************************
Symbolic HMM probabilities
**************************

In addition to floating-point transition probabilities, :mod:`sofic` can carry
exact `sympy <https://www.sympy.org>`_ expressions on HMM edges. Stationary
distributions, mixed-state presentations, bidirectional constructions, and
information anatomy then return sympy expressions that can be substituted and
simplified exactly.

Install the optional extra::

   pip install sofic[symbolic]

Building a parametric machine
==============================

Pass a sympy symbol as the control parameter to the tent-map examples (or to
:meth:`~sofic.generators.mealy.MealyHMM.add_transition` directly):

.. ipython::

   In [1]: import sympy as sp

   In [2]: from sofic.examples import tent_map_misiurewicz_forward

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

The pipeline is: the published Fig. 6 (right) non-unifilar HMM is minimized to
its ε-machine (Fig. 7) via :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm`,
and the bidirectional machine (Fig. 8) is built with a symbolic control
parameter ``a``.

.. ipython::

   In [1]: import sympy as sp

   In [2]: from sofic.examples import tent_map_misiurewicz_a, tent_map_misiurewicz_bidirectional, tent_map_misiurewicz_hmm, tent_map_misiurewicz_information_expected

   In [3]: from sofic.generators.epsilon_machine import EpsilonMachine

   In [4]: a = sp.symbols("a", positive=True)

   In [5]: eps = EpsilonMachine.from_hmm(tent_map_misiurewicz_hmm())

   In [6]: bidir = tent_map_misiurewicz_bidirectional(a)

   In [7]: r = bidir.ephemeral_information()

   In [8]: a_num = tent_map_misiurewicz_a()

   @doctest float
   In [9]: float(r.subs(a, a_num))
   Out[9]: 0.648257836793515

   @doctest float
   In [10]: tent_map_misiurewicz_information_expected(a_num)["ephemeral_mu"]
   Out[10]: 0.648257836793515

The supplement's rational-in-``a`` formula
``r_μ = (1/4)(3 - 2/(a+1) - 4/(a+2) + 9/(2a+3))`` is recovered by
:func:`~sofic.examples.epsilon_machines.tent_map_misiurewicz_information_expected`
and agrees with the machine-derived rate at the Misiurewicz root of ``a``.

:func:`~sofic.examples.epsilon_machines.tent_map_misiurewicz_hmm` is the
published 4-state Fig.~6 (right) topology (non-unifilar at ``A`` and ``D``).
Symbolic machines also draw: Graphviz / TikZ labels use
:func:`~sofic.viz._format.format_prob_label` so expressions like ``a/(a+1)``
appear on edges instead of raising on ``float(...)``.

The machine carries the Misiurewicz minimal polynomial ``a**3 - 2*a - 2`` as
:class:`~sofic.generators.prob.SymbolConstraints`; belief de-duplication and
causal-state merging then compare symbolic probabilities *modulo* that relation
(in the residue field ``Q[a]/(a**3 - 2*a - 2)``), so
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm` recovers the
4-state Fig.~7 machine.  A model without constraints keeps the two states that
coincide only at the Misiurewicz root distinct.

Probability helpers
===================

Low-level coercion and comparison live in :mod:`sofic.generators.prob`:

* :func:`~sofic.generators.prob.as_prob` — store floats or exact sympy Expr
* :func:`~sofic.generators.prob.is_symbolic` / :func:`~sofic.generators.prob.has_symbolic`
* :func:`~sofic.generators.prob.simplify_prob`, :func:`~sofic.generators.prob.probs_equal`
* :class:`~sofic.generators.prob.SymbolConstraints` — compare probabilities modulo algebraic side-relations

See also :doc:`information_anatomy` and :doc:`mixed_state_presentation`.
