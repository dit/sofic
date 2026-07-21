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

   In [1]: from sofic.examples import tent_map_misiurewicz_bidirectional

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

Structural / gauge refinement
=============================

Unifilarity of the forward ε-machine makes the next forward causal state a
deterministic function of the present, :math:`S^+_1 = \phi^+(S^+_0, X_0)`. This
splits the ephemeral and bound rates by whether the present randomness *changes
the next causal state* (structural) or merely *relabels the output on a fixed
transition* (gauge) :cite:`jurgens2026taxonomy,James2013`:

.. math::

   r_\mu = \underbrace{\H{S^+_1 \mid S^+_0, S^-_1}}_{r_\mu^{\text{struct}}}
         + \underbrace{\H{X_0 \mid S^+_0, S^+_1, S^-_1}}_{r_\mu^{\text{gauge}}},
   \qquad
   b_\mu = \underbrace{\I{S^+_1 : S^-_1 \mid S^+_0}}_{b_\mu^{\text{struct}}}
         + \underbrace{\I{X_0 : S^-_1 \mid S^+_0, S^+_1}}_{b_\mu^{\text{gauge}}}

The bound gauge term vanishes identically (Theorem A: once the source and
destination forward states are fixed, the present says nothing more about the
future), so :math:`b_\mu = b_\mu^{\text{struct}}`.

Five-variable anatomy
=====================

Adding the previous reverse causal state :math:`S^-_0 = \phi^-(S^-_1, X_0)` gives
the full joint :math:`\Pr(S^+_0, S^-_0, X_0, S^+_1, S^-_1)` and refines the
ephemeral rate into a symmetric four-atom partition, all non-negative
:cite:`jurgens2026taxonomy`:

.. math::

   r_\mu = \underbrace{\H{S^+_1 \mid S^+_0, S^-_1, S^-_0}}_{r_\mu^{\text{fwd}}}
         + \underbrace{\H{S^-_0 \mid S^+_0, S^-_1, S^+_1}}_{r_\mu^{\text{rev}}}
         + \underbrace{\I{S^+_1 : S^-_0 \mid S^+_0, S^-_1}}_{r_\mu^{\text{joint}}}
         + \underbrace{\H{X_0 \mid S^+_0, S^-_1, S^+_1, S^-_0}}_{r_\mu^{\text{gauge}}}

These regroup as :math:`r_\mu^{\text{struct}} = r_\mu^{\text{fwd}} +
r_\mu^{\text{joint}}` and :math:`r_\mu^{\text{par}} = r_\mu^{\text{rev}} +
r_\mu^{\text{gauge}}`, and the reverse structural ephemeral rate is
:math:`\bar r_\mu^{\text{struct}} = r_\mu^{\text{rev}} + r_\mu^{\text{joint}}`.
The reverse-time bound mirror obeys Theorem A′ (its gauge part
:math:`\I{X_0 : S^+_0 \mid S^-_1, S^-_0}` vanishes) and reproduces the
time-reversal invariance :math:`\bar b_\mu = b_\mu` :cite:`James2011`.

.. ipython::

   In [1]: from sofic.examples import nemo_process

   In [2]: anatomy = nemo_process().five_variable_anatomy()

   @doctest float
   In [3]: anatomy["ephemeral_forward"]
   Out[3]: 0.16666666666666666

   @doctest float
   In [4]: anatomy["ephemeral_reverse"]
   Out[4]: 0.16666666666666666

   @doctest float
   In [5]: anatomy["ephemeral_pure_gauge"]
   Out[5]: 0.08333333333333333

Internal Markov-chain entropy rate
==================================

The causal states of the ε-machine form a stationary Markov chain; its entropy
rate is :math:`h_\mu^{\text{imc}} = \H{S^+_1 \mid S^+_0}`.  By forward
unifilarity this equals :math:`\I{X_0 : S^+_1 \mid S^+_0}` — the present
randomness that *changes the next causal state* — so it is the process entropy
rate with the parallel-edge (pure output relabeling) randomness removed:

.. math::

   h_\mu^{\text{imc}} = \H{S^+_1 \mid S^+_0}
       = b_\mu + r_\mu^{\text{fwd}} + r_\mu^{\text{joint}}
       = h_\mu - r_\mu^{\text{rev}} - r_\mu^{\text{gauge}}

The reverse causal-state chain has its own rate
:math:`\bar h_\mu^{\text{imc}} = \H{S^-_0 \mid S^-_1} = b_\mu +
r_\mu^{\text{rev}} + r_\mu^{\text{joint}}`.  These are **not** equal in general:
the gap :math:`h_\mu^{\text{imc}} - \bar h_\mu^{\text{imc}} = r_\mu^{\text{fwd}}
- r_\mu^{\text{rev}}` is an arrow-of-time diagnostic, vanishing exactly when the
forward and reverse structural ephemeral branches match.

.. ipython::

   In [1]: from sofic.examples import nemo_process, NRPS

   In [2]: nemo = nemo_process().to_bidirectional()

   # Reversible branching (r_fwd == r_rev): the two chain rates agree.
   In [3]: round(nemo.internal_markov_entropy_rate(), 6), round(nemo.reverse_internal_markov_entropy_rate(), 6)
   Out[3]: (0.5, 0.5)

   In [4]: nrps = NRPS().to_bidirectional()

   # An arrow of time (r_fwd != r_rev): forward and reverse chain rates differ.
   In [5]: round(nrps.internal_markov_entropy_rate(), 6), round(nrps.reverse_internal_markov_entropy_rate(), 6)
   Out[5]: (0.333333, 0.5)

Information diagram (UpSet plot)
================================

The signed I-measure :cite:`yeung1991new` of the five random variables
:math:`(S^+_0, S^-_0, X_0, S^+_1, S^-_1)` has :math:`2^5 - 1 = 31` atoms; each is
the conditional co-information of the "inside" variables given the rest and
equals one region of the five-set information diagram.  Because the presentation
is unifilar in both time directions, the four ephemeral atoms collapse onto
*single* diagram atoms, so the anatomy is read directly off the diagram:

.. math::

   r_\mu = \underbrace{\H{X_0 \mid \dots}}_{a_{\{X_0\}}}
         + \underbrace{a_{\{X_0, S^+_1\}}}_{r_\mu^{\text{fwd}}}
         + \underbrace{a_{\{S^-_0, X_0\}}}_{r_\mu^{\text{rev}}}
         + \underbrace{a_{\{S^-_0, X_0, S^+_1\}}}_{r_\mu^{\text{joint}}},
   \qquad
   \sigma_\mu = \I{S^+_0 : S^-_1 \mid X_0}

Here :math:`b_\mu` is the sum of atoms with :math:`X_0` and :math:`S^-_1` inside
and :math:`S^+_0` outside, and the elusive information :math:`\sigma_\mu`
:cite:`ara2016elusive` is the past↔future information that bypasses the present.

:meth:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.information_diagram`
returns an
:class:`~sofic.generators.information_diagram.InformationDiagram`: the atoms in a
fixed *role* order (never sorted by value), each tagged with its anatomy role,
plus the named ``totals``.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: diagram = golden_mean().information_diagram()

   @doctest float
   In [3]: diagram.totals["r_mu"]
   Out[3]: 0.459147917027245

   @doctest float
   In [4]: diagram.totals["b_mu"]
   Out[4]: 0.20751874963942196

   In [5]: [(atom.symbol or atom.conditional_expression) for atom in diagram.atoms]

Each atom carries three names. ``atom.conditional_expression`` is the exact
I-measure (e.g. ``I[X₀:S⁺₁|S⁺₀,S⁻₀,S⁻₁]``). ``atom.symbol`` is a
``{zone} {branch}`` tag (``rμ`` / ``b⁺μ`` / ``b⁻μ`` / ``qμ`` / ``σμ`` /
``χ⁺`` / ``χ⁻`` × gauge/fwd/rev/joint, or transient/persistent for crypticity).
``atom.jurgens_label`` is the :cite:`jurgens2026taxonomy` Table II name when the
membership set is one of their fourteen atoms, or a ``†``-marked extra for the
seven cancelling partners omitted from Table II (Theorem A / A′ splits). Pass
``atoms="generic"`` to retain all 21 generically nonzero membership sets, or
``atoms="all"`` for every Yeung atom of the 31.

:func:`sofic.viz.plot_information_diagram` (also
:meth:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.plot_information_diagram`,
requires the optional ``sofic[viz]`` extra) renders the diagram as a colour-coded
UpSet plot :cite:`lex2014upset`: one signed bar per atom, a membership
dot-matrix below, Jurgens labels on the ticks, and aggregate colours —
red :math:`r_\mu`, dark green :math:`b^+_\mu`, light green :math:`b^-_\mu`,
purple :math:`q_\mu`, blue :math:`\sigma_\mu`, orange :math:`\chi^+`, amber
:math:`\chi^-`::

   from sofic.examples import nemo_process

   fig = nemo_process().plot_information_diagram(atoms="process")
   fig.savefig("nemo_anatomy.png", dpi=150, bbox_inches="tight")

Symbolic probabilities
======================

Transition probabilities may be sympy expressions. The same anatomy methods then
return exact expressions; see :doc:`symbolic_hmm` for the Chaos Forgets
appendix walkthrough.

API
===

Use :meth:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.information_anatomy`
on bidirectional models. :class:`~sofic.generators.epsilon_machine.EpsilonMachine`
also exposes these quantities by building its bidirectional presentation.

When that construction is unavailable, use
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.approximate_information_anatomy`
or :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.block_convergence_estimates`.
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

   In [1]: from sofic.examples import tent_map_misiurewicz_bidirectional

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
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.caekl_block_information`;
``j_μ`` is **not** a single-step bidirectional quantity like ``ρ_μ``.  When the
affine tail of ``J(ℓ)`` stabilizes,
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.caekl_rate_converged`
returns ``True`` and :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.caekl_rate`
is exact.  Multivariate ordering yields ``j_μ ≤ b_μ ≤ ρ_μ``; ``j_μ`` is not
determined by ``h_μ`` alone.

Causal irreversibility and stored information
=============================================

Time-asymmetric stored information is available via
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.causal_irreversibility`
and :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.stored_information_decomposition`
:cite:`Crutchfield2009,Ellison2009`.

Finite-block convergence scalars — transient, oracular, gauge, and predictability-gain
information — are exposed as :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.transient_information`,
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.oracular_information`,
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.gauge_information`, and
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.predictability_gain`.

Topological anatomy
===================

Evaluating this same anatomy at the measure of maximal entropy of a sofic shift
yields the *topological* split ``h_top = b_top + r_top`` — a conjugacy invariant
of the shift space rather than of a particular measure.  See
:doc:`../shifts/topological_anatomy`.
