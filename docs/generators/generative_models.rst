.. generative_models.rst
.. py:module:: sofic.generators.minimal_generative_model

*****************
Generative Models
*****************

The statistical complexity :math:`C_\mu = \H{S^+}` is the cost of *prediction*:
the memory a unifilar (ε-machine) presentation must carry. A process can often
be *generated* with less memory by a non-unifilar machine. ``sofic`` builds
these minimal generators from a
:class:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine`,
each realizing a different **common information** between the forward causal
state :math:`S^+` and the reverse causal state :math:`S^-`, restricted to
auxiliaries that are the states of a *lifted* generator (see
`Lifted generators`_).

The generative complexity :math:`C_g = \H{G}` of a presentation with generative
states :math:`G` obeys :math:`C_g \le C_\mu`. The three generator
constructions here sit in the order

.. math::

   \op{I}{S^+ : S^-} \;\le\; \text{(Wyner)} \;\le\; \text{(exact)} \;\le\;
   \text{(functional)} \;\le\; \min(C_\mu^+, C_\mu^-),

where the Wyner value is :math:`\op{I}{(S^+, S^-) : G}` and the others are
:math:`\H{G}` (the first two up to optimizer accuracy).

* :func:`minimal_generative_model` — the **exact common information**
  :cite:`Kumar2014`: the minimum-entropy lifted generator state :math:`G`
  rendering :math:`S^+` and :math:`S^-` conditionally independent, found by a
  stochastic optimizer. Its entropy bounds :math:`C_g` from above.
* :func:`wyner_generative_model` — the **Wyner common information**
  :cite:`Wyner1975`, minimizing :math:`\op{I}{(S^+, S^-) : G}` over the same
  lifted generator states.
* :func:`functional_generative_model` — the **functional common information**:
  the smallest *deterministic* function of :math:`(S^+, S^-)` that separates
  the two causal states and is a generator state, found by a partition search.
* :func:`gacs_korner_generative_model` — the **Gács–Körner common information**
  :cite:`GacsKorner1973`: the combinatorial meet :math:`S^+ \wedge S^-`, the
  largest variable that is simultaneously a deterministic function of both.
  This captures only the conserved "core" (phase / ergodic structure) and is
  often trivial for mixing processes.

Each returns a (generally non-unifilar) :class:`~sofic.generators.mealy.MealyHMM`
subclass whose ``generative_complexity()`` is its state entropy. The exact,
Wyner, and functional models reproduce the source process. The Gács–Körner model
does **not**: the meet need not render :math:`S^+` and :math:`S^-` conditionally
independent, so it is the common-information variable presented as an HMM
rather than a generator — ``is_equal_process(source)`` is ``False`` in general
(the golden mean collapses to an i.i.d. coin), and its ``entropy_rate()`` is its
own, not the source's.

.. code-block:: python

   from sofic.examples import golden_mean_bidirectional

   bidir = golden_mean_bidirectional(0.5)

   gk = bidir.gacs_korner_generative_model()   # deterministic, combinatorial
   gk.generative_complexity()                   # H[S+ ∧ S-]

   exact = bidir.minimal_generative_model()     # stochastic optimizer
   exact.generative_complexity()                # <= C_mu

The optimizers start from the corresponding ``dit`` common-information routines
and require ``dit`` (a core dependency). The convenience methods
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.minimal_generative_model`
and friends on an :class:`~sofic.generators.epsilon_machine.EpsilonMachine`
build the bidirectional presentation first.

Lifted generators
=================

The common informations of :math:`S^+` and :math:`S^-` constrain only
:math:`\op{I}{S^+ : S^- \mid G} = 0`. The HMM built from a channel applies
:math:`p(g \mid s^+, s^-)` at both ends of every bidirectional edge (the
*lift*), and that HMM generates the source process, with :math:`G` jointly
distributed with the past and future as the channel prescribes, only if the
lifted state sequence is Markov:

.. math::

   \op{I}{G_0 X_0 : S^-_1 \mid G_1} = 0, \qquad \op{I}{S^+_0 : X_0 G_1 \mid G_0} = 0.

The exact, Wyner, and functional constructors impose both conditions, so every
point they search is a generator state (Ellison et al. :cite:`Ellison2009` for
the bidirectional machine). The exact value, which
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.generative_complexity`
returns, bounds the generative complexity of Löhr & Ay :cite:`LohrAy2009` from
above. ``markov_residual`` reports the largest constraint information (bits) at
the returned channel.

Lifted generators are a proper subclass of generators: the lift draws
:math:`G_0` and :math:`G_1` independently given the bidirectional edge, while a
general generator may couple them. For
:math:`Z_t = Z_{t-2} \oplus \mathrm{Bernoulli}(0.1)` (two interleaved
first-order chains), the product of the two chains' minimal generators has
state :math:`G_t = (U_t, U_{t+1})`, which shares :math:`U_{t+1}` with
:math:`G_{t+1}`. It is a generator state of entropy 1.982, essentially
:math:`\op{G}{S^+ : S^-}`, but violates both lifting conditions (by 0.019 bits),
so the lifted search cannot reach it and returns a larger value. The exact and
Wyner values here are therefore upper bounds over lifted generators, not
minima over all generators. For a deterministic channel the coupling is
forced, so the functional search is exact over all deterministic generator
states.

For first-order Markov processes the constrained and unconstrained values
coincide. They differ already for the second-order Markov process in which
every 1 is followed by at least two 0s: its unconstrained exact (and functional)
optimum, of entropy 1.406, is a deterministic variable whose value
:math:`(C, R_{01})` ("the past ends in 10 and the future starts with 01") can
only be preceded by a value that does not know the past ended in 10, so no HMM
has it as its state; the lifted search finds 1.5. The stochastic optimizer is
local and seeded (``starts``, ``rng``); on harder processes more starts can
find lower values.

The unconstrained values remain properties of the process — they equal the
common informations of the past and future — and are available as measures:

.. code-block:: python

   bidir.wyner_common_information()   # C[S+ : S-] = C[past : future]
   bidir.exact_common_information()   # G[S+ : S-] = G[past : future]

Because every generator state separates the causal states, they bracket the
generative complexity: :math:`\op{G}{S^+ : S^-} \le C_g \le \H{G}` for the
state :math:`G` of :func:`minimal_generative_model`. :func:`generative_gap`
reports both ends and their difference, an upper bound on
:math:`C_g - \op{G}{S^+ : S^-}`, and the analogous Wyner comparison:

.. code-block:: python

   gap = bidir.generative_gap()            # measure="exact"
   gap.lower, gap.upper, gap.gap           # 1.406, 1.5, 0.094 for the process above
   bidir.generative_gap(measure="wyner")   # 1.129, 1.311, 0.182

Both ends are optimizer estimates, and the upper end is over lifted generators
only, so ``gap`` can be positive even when some non-lifted generator attains
:math:`\op{G}{S^+ : S^-}` (the interleaved process above). The gap vanishes for
first-order Markov processes.

Coupled generators
==================

:mod:`sofic.generators.coupled_generators` drops the lifting restriction. A
generator with :math:`k` states is described, jointly with the causal states,
by a state law :math:`\pi`, beliefs :math:`Q_w(s^+)` and :math:`R_w(s^-)`, and a
free kernel :math:`K(x, w' \mid w)`; the product
:math:`\pi_w Q_w(s^+_0) K(x_0, w_1 \mid w_0) R_{w_1}(s^-_1)` must reproduce the
bidirectional edge law :cite:`Ellison2009`, be stationary on the past side and
consistent on the future side. :func:`~sofic.generators.coupled_generators.coupled_search`
minimizes :math:`\H{\pi}` subject to these equations (an augmented Lagrangian
with analytic gradients) from the ε-machines, the bidirectional machine, extra
seeds and random starts; :func:`~sofic.generators.coupled_generators.coupled_generative_model`
returns the best candidate that reproduces the source process. For the
interleaved process above it reaches :math:`2 h(\tfrac49) = 1.98215`, below the
best lifted generator (1.99108); for the :math:`(2, \infty)` process it finds
the ε-machine (1.5).

Two narrower tools share the same equations:

* :func:`~sofic.generators.coupled_generators.memory_lift` builds the
  *memory-1 lift* of a channel :math:`\nu(v \mid s^+, s^-)`: draw
  :math:`V_t \sim \nu(\cdot \mid B_t)` independently at each time and take
  :math:`W_t = (V_t, V_{t+1})`, so consecutive states share randomness. The
  interleaved process's minimal generator is the memory-1 lift of a single
  flag, and :func:`~sofic.generators.coupled_generators.memory_lift_search`
  finds it.
* :func:`~sofic.generators.coupled_generators.generator_kernel` decides whether
  a given channel's variable is the state of *any* generator: for a fixed channel
  the equations are linear in :math:`K`, and a linear program returns the
  smallest violation and the kernel. The exact-common-information optimum of the
  :math:`(2, \infty)` process has violation 0.0625, so no generator has it as
  its state. :func:`~sofic.generators.coupled_generators.bilevel_search`
  (experimental) searches channels against this violation.

All three searches are local and return upper bounds on :math:`C_g` for the
given number of states.

Searching for a generator
=========================

:func:`~sofic.generators.generative_search.generative_model` is the primary entry
point: it builds a bidirectional presentation when needed, runs the full pipeline,
and returns the best verified generator with a bracket on :math:`C_g`.
:func:`~sofic.generators.generative_search.generative_model_from_bidirectional`
runs the same search when the bidirectional machine is already known.

* lower bounds: the excess entropy, support-type and flow bounds, and a coverage
  bound (``coverage="heuristic"`` or ``coverage="certified"``). The certified caps
  come from :func:`~sofic.generators.coverage_bound.mu_caps`: an exact ``mu_1`` by
  vertex enumeration (:func:`~sofic.generators.coverage_bound.mu1_vertex`), the
  singular-value tail cap ``mu_k <= 1 - sum_{i>k} sigma_i``
  (:func:`~sofic.generators.coverage_bound.mu_k_spectral_cap`, via
  Eckart–Young–Mirsky :cite:`mirsky1960symmetric`), ``k mu_1``, subadditivity, and, when
  the smaller side of the joint has three states, the exact ``mu_2`` of
  :func:`~sofic.generators.coverage_bound.mu_hyperplane_cap` (branch and bound over the plane
  containing the columns of a rank-2 matrix; exact since nonnegative rank two equals rank two
  :cite:`cohen1993nonnegative`);
* the persistent-crypticity criteria; when both vanish every separating variable
  is a generator state and :math:`C_g = \op{G}{S^+ : S^-}`;
* the exact-common-information optima (dit, several starts, since the optimum
  need not be unique and only some optima generate), each polished to exact
  separation with :func:`~sofic.generators.coupled_generators.polish_separating_channel`
  and tested with :func:`~sofic.generators.coupled_generators.generator_kernel`;
  when none generate, :func:`~sofic.generators.coupled_generators.bilevel_search`
  may propose feasible channels, then an entropy-ordered sweep over multistart
  separating decompositions (:func:`~sofic.generators.coupled_generators.separating_decomposition_channels`);
* upper bounds over nested classes, cheapest first: the ε-machines and
  bidirectional machine, memory-1 lifts, the exact minimal functional generator
  (:func:`~sofic.generators.functional_generators.functional_cover_search`), lifted channels (for
  small machines), and the coupled search seeded with everything found so far.

.. code-block:: python

   result = bidir.generative_model()
   result.lower, result.upper, result.certified   # bracket on C_g
   result.cg_exceeds_g                            # True when no G optimum generates
   result.separating_variables_generate           # some G optimum passed kernel LP
   result.model                                   # the generator, verified against the source

The searches are local; ``certified`` reports when the bracket closes (for
example the golden mean, where the support bound meets :math:`C_\mu`).

Wyner and functional generators
-------------------------------

Two variants return the same :class:`~sofic.generators.generative_search.GenerativeModelResult`,
with ``objective`` naming what the bracket bounds.

:func:`~sofic.generators.generative_search.minimal_wyner_model` minimizes
:math:`\I{W : S^+, S^-}` over generators, the Wyner :cite:`Wyner1975` analogue of :math:`C_g`.
It runs the candidate stages of :func:`~sofic.generators.generative_search.generative_model`,
adds the lifted Wyner model and the unconstrained Wyner optimum (when it passes the generator
kernel), and ranks the verified generators by
:math:`\I{W : S^+, S^-} = \H{S^+, S^-} - \sum_w \pi_w (\H{S^+ \mid w} + \H{S^- \mid w})`.
The state-entropy lower bounds do not apply, so ``lower`` is the excess entropy. When both
persistent crypticities vanish the minimum is the Wyner common information
:math:`\op{C}{S^+ : S^-}` (2/3 bit for the golden mean).

:func:`~sofic.generators.functional_generators.minimal_functional_generator` minimizes the state
entropy over generators whose state is a function of :math:`(S^+, S^-)`. Such a generator is a
coarse-graining of the bidirectional machine whose blocks are product rectangles of the joint
support, so the search
(:func:`~sofic.generators.functional_generators.functional_cover_search`) solves the
least-entropy exact cover by product rectangles, whose value is the functional common information
:math:`\op{F}{S^+ : S^-}` :cite:`james2017multivariate`, checks the lifting conditions, and
excludes failed covers until one generates. The result is exact, with a certified lower bound if
``max_covers`` runs out. On the :math:`(2, \infty)` process the first cover has entropy
:math:`2 - \tfrac38 \log_2 3` but does not generate; the minimal functional generator has
:math:`3/2` bits, equal to :math:`C_g`.

.. code-block:: python

   bidir.minimal_wyner_model().upper         # min I[W : S+, S-] over generators found
   bidir.minimal_functional_generator()      # exact, certified bracket

Lower bounds
============

:mod:`sofic.generators.generator_bounds` bounds :math:`\op{G}{S^+ : S^-}`, and
hence :math:`C_g` (Löhr & Ay :cite:`LohrAy2009`), from below using only the
support of the causal-state joint distribution. Each value of a separating
variable has a rectangular support; typing values by maximal rectangles gives a
polytope of type masses on which entropy is minimized at a vertex
(:func:`~sofic.generators.generator_bounds.support_type_bound`). The bound is
exact for the :math:`(2, \infty)` process (:math:`2 - \tfrac38 \log_2 3`) and
for the golden mean process (:math:`C_\mu`, proving :math:`C_g = C_\mu` there);
it is uninformative when the support is the full product.

Every bound above is at most :math:`\op{G}{S^+ : S^-}` except the flow bound, which
needs a sparse support. :mod:`sofic.generators.generator_caps` uses the generator
equations on full supports too: :func:`~sofic.generators.generator_caps.generator_weight_cap`
caps the total weight of a generator's ``k`` heaviest states by keeping them explicit,
aggregating the rest, writing the edge law, forward stationarity and reverse
consistency for the group moments, and branching over boxes of the explicit states'
beliefs with McCormick envelopes :cite:`mccormick1976computability`. Coverage with these
caps (:func:`~sofic.generators.generator_caps.generator_coverage_bound`, the
``generator_caps`` stage of the pipeline) exceeds :math:`\op{G}{S^+ : S^-}` on most
survey processes with positive persistent crypticity, and on the :math:`(2, \infty)`
process it gives caps :math:`(1/2, 3/4)` and :math:`C_g \ge 3/2` up to the LP tolerance,
which with the known generator gives :math:`C_g = 3/2`.

API
===

Constructors
------------

.. autofunction:: minimal_generative_model
.. autofunction:: wyner_generative_model
.. autofunction:: functional_generative_model
.. autofunction:: gacs_korner_generative_model

Coupled generators and bounds
-----------------------------

.. autofunction:: sofic.generators.generative_search.generative_model
.. autofunction:: sofic.generators.generative_search.generative_model_from_bidirectional
.. autofunction:: sofic.generators.generative_search.minimal_wyner_model
.. autofunction:: sofic.generators.functional_generators.minimal_functional_generator
.. autofunction:: sofic.generators.functional_generators.functional_cover_search
.. autofunction:: sofic.generators.functional_generators.product_rectangles
.. autofunction:: sofic.generators.generative_search.resolve_bidirectional
.. autoclass:: sofic.generators.generative_search.GenerativeModelResult
   :members: certified, gap, g_optima_generate
.. autofunction:: sofic.generators.coverage_bound.coverage_entropy_lower
.. autofunction:: sofic.generators.coverage_bound.mu1_optimize
.. autofunction:: sofic.generators.coverage_bound.mu1_vertex
.. autofunction:: sofic.generators.coverage_bound.mu_caps
.. autofunction:: sofic.generators.coverage_bound.mu_k_optimize
.. autofunction:: sofic.generators.coverage_bound.mu_k_spectral_cap
.. autofunction:: sofic.generators.coverage_bound.mu_hyperplane_cap
.. autofunction:: sofic.generators.generator_caps.generator_weight_cap
.. autofunction:: sofic.generators.generator_caps.generator_coverage_bound
.. autofunction:: sofic.generators.coverage_bound.lp_rank_k_weights
.. autofunction:: sofic.generators.coverage_bound.mu1_search
.. autofunction:: sofic.generators.coupled_generators.coupled_generative_model
.. autofunction:: sofic.generators.coupled_generators.polish_separating_channel
.. autofunction:: sofic.generators.coupled_generators.coupled_search
.. autofunction:: sofic.generators.coupled_generators.memory_lift
.. autofunction:: sofic.generators.coupled_generators.memory_lift_search
.. autofunction:: sofic.generators.coupled_generators.generator_kernel
.. autofunction:: sofic.generators.coupled_generators.bilevel_search
.. autofunction:: sofic.generators.coupled_generators.line_split_seeds
.. autofunction:: sofic.generators.coupled_generators.targeted_line_split_seeds
.. autofunction:: sofic.generators.generator_bounds.support_type_bound
.. autofunction:: sofic.generators.generator_bounds.flow_bound
.. autofunction:: sofic.generators.generator_bounds.flow_bound_result
.. autofunction:: sofic.generators.generator_bounds.rll_flow_bound_certified
.. autofunction:: sofic.generators.generator_bounds.generator_support_bound
.. autofunction:: sofic.generators.bidirectional_build.preflight_bidir
.. autofunction:: sofic.generators.bidirectional_build.try_build_bidirectional
.. autofunction:: sofic.generators.bidirectional_build.build_bidirectional_with_ladder
.. autofunction:: sofic.generators.universal_optimizer.universal_generative_optimizer
.. autoclass:: sofic.generators.universal_optimizer.UniversalGenerativeResult
   :members:
.. autofunction:: sofic.generators.universal_optimizer.compare_universal_to_pipeline
.. autoclass:: sofic.generators.generator_problem_spec.GeneratorProblemSpec
   :members:

Measures
--------

.. autofunction:: wyner_common_information
.. autofunction:: exact_common_information
.. autofunction:: generative_gap
.. autoclass:: GenerativeGap
   :members: gap

Models
------

.. autoclass:: MinimalGenerativeModel
   :members: generative_complexity, entropy_rate
.. autoclass:: WynerGenerativeModel
   :members: generative_complexity, entropy_rate
.. autoclass:: FunctionalGenerativeModel
   :members: generative_complexity, entropy_rate
.. autoclass:: GacsKornerGenerativeModel
   :members: generative_complexity, entropy_rate
