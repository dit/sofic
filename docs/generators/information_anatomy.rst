.. information_anatomy.rst

********************
Information Anatomy
********************

James, Burke & Crutchfield (2013) decompose the process entropy rate into
predicted, bound, and ephemeral components :cite:`James2013`:

.. math::

   h_\mu = b_\mu + r_\mu, \qquad
   \rho_\mu = \I{X_0 : S^+_0}, \qquad
   b_\mu = \I{X_0 : S^-_1 \mid S^+_0}, \qquad
   r_\mu = \H{X_0 \mid S^+_0, S^-_1}

.. ipython::

   In [1]: from pensive.examples import tent_map_misiurewicz_bidirectional

   In [2]: bidir = tent_map_misiurewicz_bidirectional()

   @doctest float
   In [3]: bidir.predicted_information()
   Out[3]: 0.12334603575116132

   @doctest float
   In [4]: bidir.bound_information()
   Out[4]: 0.174914618901008

   @doctest float
   In [5]: bidir.ephemeral_information()
   Out[5]: 0.648257836793596

   In [6]: anatomy = bidir.information_anatomy()

Symbolic probabilities
======================

Transition probabilities may be sympy expressions. The same anatomy methods then
return exact expressions; see :doc:`symbolic_hmm` for the Chaos Forgets
appendix walkthrough.

API
===

Use :meth:`~pensive.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.information_anatomy`
on bidirectional models. :class:`~pensive.generators.epsilon_machine.EpsilonMachine`
also exposes these quantities by building its bidirectional presentation.

When that construction is unavailable, use
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.approximate_information_anatomy`
or :meth:`~pensive.generators.epsilon_machine.EpsilonMachine.block_convergence_estimates`.
These finite-block estimates do not replace the exact bidirectional quantities;
they report the current block-length approximation to ``h_mu``, ``E``,
``rho_mu``, ``b_mu``, ``r_mu``, ``q_mu``, ``w_mu``, block coinformation, CAEKL,
and related convergence curves.  See :doc:`block_convergence`.

CAEKL past-present-future information
=====================================

The Chan-AlBashabsheh-Ebrahimi-Kaced-Liu multivariate mutual information
:cite:`chan2015multivariate` over the anatomy triple is a finite, closed-form
"shared information" among past, present, and future.  The forward causal state
:math:`S^+_0` and reverse causal state :math:`S^-_1` stand in for the
semi-infinite past and future, keeping the quantity finite:

.. math::

   \op{J}{S^+_0 : X_0 : S^-_1}

.. ipython::

   In [1]: from pensive.examples import tent_map_misiurewicz_bidirectional

   In [2]: bidir = tent_map_misiurewicz_bidirectional()

   @doctest float
   In [3]: bidir.caekl_causal_information()
   Out[3]: 0.22044436492357078

This single-step quantity is distinct from the block CAEKL rate ``j_μ`` below:
it is a bidirectional causal-state measure like ``ρ_μ``, whereas ``j_μ`` is the
asymptotic slope of the block curve ``J(ℓ)``.

CAEKL rate ``j_μ``
==================

The block CAEKL curve ``J(ℓ)`` and its asymptotic rate ``j_μ`` are documented in
:doc:`block_convergence`.  Per-block values ``J(ℓ)`` are exact from
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.caekl_block_information`;
``j_μ`` is **not** a single-step bidirectional quantity like ``ρ_μ``.  When the
affine tail of ``J(ℓ)`` stabilizes,
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.caekl_rate_converged`
returns ``True`` and :meth:`~pensive.generators.epsilon_machine.EpsilonMachine.caekl_rate`
is exact.  Multivariate ordering yields ``j_μ ≤ b_μ ≤ ρ_μ``; ``j_μ`` is not
determined by ``h_μ`` alone.

Causal irreversibility and stored information
=============================================

Time-asymmetric stored information is available via
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.causal_irreversibility`
and :meth:`~pensive.generators.epsilon_machine.EpsilonMachine.stored_information_decomposition`
:cite:`Crutchfield2009,Ellison2009`.

Finite-block convergence scalars — transient, oracular, gauge, and predictability-gain
information — are exposed as :meth:`~pensive.generators.epsilon_machine.EpsilonMachine.transient_information`,
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.oracular_information`,
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.gauge_information`, and
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.predictability_gain`.

Topological anatomy
===================

Evaluating this same anatomy at the measure of maximal entropy of a sofic shift
yields the *topological* split ``h_top = b_top + r_top`` — a conjugacy invariant
of the shift space rather than of a particular measure.  See
:doc:`../shifts/topological_anatomy`.
