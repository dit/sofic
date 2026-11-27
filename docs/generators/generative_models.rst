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
* :func:`~sofic.generators.coupled_generators.generator_kernel` tests a given
  channel's variable against the generator equations: for a fixed channel the
  equations are linear in :math:`K`, and a linear program returns the smallest
  violation and the kernel. The equations are necessary, so a positive violation
  rules the variable out: the exact-common-information optimum of the
  :math:`(2, \infty)` process has violation 0.0625, so no generator has it as
  its state. They are not sufficient: future-side consistency only constrains
  the belief :math:`R_w` through its one-step average, and a kernel with
  violation 0 can still emit the wrong word distribution (on a survey process an
  11-state example matches words up to length 3 and is off by about 0.02 at
  length 8). Every candidate the pipeline reports is therefore checked against
  the source's word process: word probabilities up to a fixed length, and a
  difference-automaton test (Schützenberger, Tzeng) over words that span all word
  vectors, so a candidate that matches short words but drifts later is rejected. :func:`~sofic.generators.coupled_generators.bilevel_search`
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
* the persistent-crypticity criteria, with the cryptic orders and the block persistent crypticities; when both
  criteria vanish (cryptic order at most 1) every separating variable is a generator state,
  :math:`C_g = \op{G}{S^+ : S^-}`, and the exact path computes it with at most :math:`K(P)` states
  (:doc:`separating_generators`); at cryptic orders 2 and 3 the lookahead generator is a candidate;
* the exact-common-information optima (dit, several starts, since the optimum
  need not be unique and only some optima generate), each polished to exact
  separation with :func:`~sofic.generators.coupled_generators.polish_separating_channel`
  and tested with :func:`~sofic.generators.coupled_generators.generator_kernel`;
  when none generate, :func:`~sofic.generators.coupled_generators.bilevel_search`
  may propose feasible channels, then an entropy-ordered sweep over multistart
  separating decompositions (:func:`~sofic.generators.coupled_generators.separating_decomposition_channels`);
* upper bounds over nested classes, cheapest first: the ε-machines and
  bidirectional machine, exact one-sided generators from the simplex-chain search (below), memory-1 lifts, the
  exact minimal functional generator
  (:func:`~sofic.generators.functional_generators.functional_cover_search`), lifted channels (for
  small machines), and the coupled search seeded with everything found so far.

.. code-block:: python

   result = bidir.generative_model()
   result.lower, result.upper, result.certified   # bracket on C_g
   result.cg_exceeds_g                            # True when no G optimum generates
   result.separating_variables_generate           # some G optimum passed kernel LP
   result.model                                   # the generator, verified against the source
   result.reproduction_error                      # its word-process deviation from the source
   result.structure                               # states, weights, functional, lifted
   result.decomposition                           # pi, Q, R, K with labels (JSON-ready)

The searches are local; ``certified`` reports when the bracket closes and the generator reproduces the
source to the same tolerance (for example the golden mean, where the support bound meets :math:`C_\mu`).
``structure["lifted"]`` says whether the generator's channel satisfies the lifting conditions, so that
drawing its states independently at both ends of each bidirectional edge also generates the process;
coupled generators such as the interleaved product flag do not.

The coupled searches parameterize states by their causal-state beliefs :math:`(Q_w, R_w)`, which favours
generators whose states' pasts and futures are belief-mixtures of the causal states'. Some processes have
lower-entropy generators that are not. :func:`~sofic.generators.hmm_search.hmm_generator_search` optimizes the
symbol matrices of a :math:`k`-state HMM directly, minimizing :math:`H[\pi]` subject to exact word
probabilities up to length :math:`k + |S^+|`, which implies equality on all words :cite:`Tzeng1992`. It uses an
augmented Lagrangian with analytic gradients: first feasibility from a random start, then entropy descent. Its
``deadline`` is checked after every L-BFGS-B iteration, so a run stops within one iteration of it. Direct
parameter search for minimal generators of binary Markov processes is the approach of :cite:`Ruebeck2018`. The
pipeline runs it after the coupled search (``hmm_search=True``) for :math:`k = |S^+|, \ldots, |S^+| + 2` with
half the remaining budget, or for two values of :math:`k` and two starts without a budget, and offers the
results through :func:`~sofic.generators.hmm_search.coupled_seed_from_hmm`. On the binary 3-state dataset it
lowered seven open upper bounds, for example from :math:`C_\mu = 0.9037` to :math:`0.7956` with a 4-state
generator that the belief-parameterized searches never reached.

Both searches are local in the generator's parameters. :func:`~sofic.generators.simplex_chains.simplex_chain_search`
instead constructs *one-sided* generators exactly. A generator is reverse-sufficient when the future of each state
:math:`w` is the mixture of the reverse causal states' futures weighted by its belief :math:`R_w` on :math:`S^-`. It
is then a finite law on the simplex :math:`\Delta(S^-)` that dominates its image under the belief updates in convex
order :cite:`Strassen1965`. Writing every updated belief of a finite pool of atoms as a convex combination of the
pool gives a chain whose closed classes are exact generators. The state entropy is concave on the polytope of those
representations, so the search alternates linear programming over the representations with gradient steps on the
atoms; forward-sufficient generators are the mirror image on :math:`\Delta(S^+)`. Each side starts from its own
ε-machine, every iterate passes a word check over a Tzeng basis :cite:`Tzeng1992`, and at cryptic orders up to 4 the
atoms are kept on the faces of the synchronizing complex. The construction and the search are not from the
literature. The pipeline runs the search right after the exact path (``simplex_chains=True``) with a quarter of the
remaining budget, or 2 s per side without one, and stops once a generator meets the lower bound. On 306 archive
processes (binary 2- and 3-state and ternary 2-state ε-machines) it went below the best known upper bound on 42, by
more than :math:`10^{-6}` on 22 and by up to 0.295 bits; half the final values came within 0.57 s.

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
survey processes with positive persistent crypticity. The pipeline skips both coverage stages once the
cheaper bounds meet the entropy of an ε-machine, which then certifies. At cryptic order at most 1, where
:math:`C_g = \op{G}{S^+ : S^-}` and the exact path usually closes the bracket, the generator coverage bound
runs only if the exact path leaves it open; on the 510 dataset processes of that class it had taken about
17 s each and never supplied the lower bound.

The branch and bound drops a box only when HiGHS calls its LP infeasible with and without presolve and an
exact Farkas certificate verifies (:mod:`sofic.generators.certificates`). HiGHS's presolve calls feasible box
LPs infeasible: on ``((1,0),(2,2),(1,0))`` it did so 3,101 times at 1,000 and 2,000 boxes, dropping the box
that holds a 3-state generator's heaviest state at depth 81, and the coverage bound came out at
:math:`C_\mu = 1.0459`, above that generator's entropy :math:`1.0420`. Without presolve every one of those LPs
is feasible, and the bound meets the generator's entropy to :math:`3 \times 10^{-10}`. With float data a nonempty box is always feasible (the explicit
states can have weight 0 and the aggregate the whole generator), so the certificate only guards against
rationalized data.

Every coverage bound ends in :func:`~sofic.generators.coverage_bound.coverage_entropy_lower`, the least entropy
of a distribution whose :math:`j` heaviest weights sum to at most :math:`c_j`. The cap increments
:math:`(c_1, c_2 - c_1, \ldots, 1 - c_k)` majorize every such distribution, and entropy is Schur-concave
:cite:`marshall2011inequalities`, so their entropy is a lower bound, and the least entropy when they do not
increase. When they increase it is weaker, and not monotone in the caps. Given the :math:`k` heaviest weights, the
rest of the mass has least entropy as copies of the :math:`k`-th weight and a remainder, and for each number of
copies the entropy is concave in the heaviest weights. The minimum is therefore at a vertex of their polytope, or
where one of its edges leaves no remainder. With more than :math:`Q` copies every weight from the :math:`k`-th on is
at most :math:`r / (Q + 1)`, where :math:`r` is the mass outside the heaviest :math:`k - 1`, so the entropy is at
least :math:`\sum_{i<k} \phi(w_i) + \phi(r) + r \log_2 (Q + 1)` with :math:`\phi(w) = -w \log_2 w`. That bound is
concave in the heaviest :math:`k - 1` weights, so its least value is at a vertex of their polytope, and :math:`Q`
doubles until that value reaches the least entropy found. We know of no literature source for this reduction. On a binary 3-state process with caps :math:`0.4720` and :math:`0.7270`
(increments :math:`0.472, 0.255, 0.273`) it raises the bound from :math:`1.5252655` to :math:`1.5399102`, at
:math:`(0.4540, 0.2730, 0.2730)`.

Coverage only sees the top weights. :func:`~sofic.generators.exact_ci_bounds.heaviest_term_bound`
uses the whole decomposition: with heaviest term :math:`T = t\, q r^\top`, any exact decomposition has
entropy :math:`h(t) + (1 - t) H(\lambda')`, where :math:`\lambda'` decomposes the remainder
:math:`(P - T)/(1 - t)` with atoms at most :math:`t/(1 - t)`. Branching over boxes of :math:`(q, r, t)`
and bounding the remainder by certified caps of its entrywise upper envelope gives a lower bound on
:math:`\op{G}{S^+ : S^-}`; an optimum needs at most :math:`2^{\min(|S^+|, |S^-|)} - 1` terms
:cite:`Kumar2014`, so :math:`t` is at least the reciprocal. On joints whose optimum leaves two terms
after the heaviest, the bound is tight. The pipeline runs it (the ``heaviest-term bound`` stage) when
the certified lower bound is still below the exact-common-information estimate and
:math:`|S^+| + |S^-| \le 7`. On three survey processes with :math:`3 \times 3` joints it raises the
lower bound to the estimate, closing ``((0,1),(2,·),(0,1))`` at :math:`0.9335` bits.

A generator's states are an exact decomposition whose heaviest term is its heaviest state, and that
state satisfies the generator equations, so the same branching bounds :math:`C_g` when each box's
weight is also capped by the one-explicit-state generator-cap LP over the box of beliefs
(:func:`~sofic.generators.exact_ci_bounds.generator_heaviest_term_bound`). Where the LP excludes the
:math:`\op{G}{}`-optimal heavy term the bound exceeds :math:`\op{G}{S^+ : S^-}`: on
``((0,1),(0,2),(0,1))`` it certifies :math:`C_g \ge 0.928 > \op{G}{} = 0.924`, but only after about
220,000 boxes (half an hour), because the LP excludes that term only in boxes of radius about
:math:`10^{-4}`. Elsewhere it converges much faster. On the 106 open :math:`C_g` brackets of a
dataset of binary 2- and 3-state ε-machines (joints from :math:`2 \times 2` to :math:`3 \times 11`),
900 s per process raised the lower bound on 48 of 105 and closed 9, taking the median bracket width
from 0.066 to 0.039 bits. The closed ones include processes where some generator's heaviest state
outweighs the minimal generator's, which caps on the heaviest weights alone cannot certify. The
pipeline therefore runs it last, once the upper bound is verified, aiming at that bound with the
remaining budget (``generator_heaviest_term=True``, the default), on joints with up to 14 causal
states; without a budget it stays at 7 causal states and 2,000 boxes. Each generator-cap LP inside it
is limited to the time left, because a single degenerate LP otherwise ran past the deadline by more
than half an hour. This combination is not from the literature.

Lifted generators get their own bound.
:func:`~sofic.generators.exact_ci_bounds.lifted_heaviest_term_bound` uses that a lifted generator draws
consecutive states from one channel, so its heaviest state :math:`e` has channel
:math:`c_e(a, b) = t\, q_a r_b / p(a, b)` at both ends of every edge. Coarse-graining the other end to the
event that it is :math:`e` turns each lifting condition into a condition on :math:`(q, r)` alone: a
polynomial in the beliefs must be the same for every supported forward (or reverse) state. Its
coefficients are nonnegative, so it is checked on boxes by interval arithmetic and prunes the
heaviest-term branching. A coupled generator need not pass the test. On dataset process
``((0,1),(0,2),(0,2))`` (a :math:`3 \times 3` joint) every lifted generator has entropy at least
:math:`1.4815`, with a lifted generator at :math:`1.4817`, while :math:`C_g = \op{G}{S^+ : S^-} = 1.4225`:
the minimal generator is not lifted even though an exact-common-information optimum generates. This
bound is not from the literature.

:func:`~sofic.generators.generator_caps.certified_generator_weight_cap` removes the LP
tolerance: every branch-and-bound leaf gets an exact bound from rationalized HiGHS duals or an
exact Farkas certificate of infeasibility (:mod:`sofic.generators.certificates`, after
Neumaier & Shcherbina :cite:`neumaier2004safe`; the same module bounds SDPs through
positive-definite rounding of the dual matrices, as in Jansson et al. :cite:`jansson2007rigorous`),
and box tightening rounds outward so the leaves cover every generator.
:func:`~sofic.generators.generator_caps.verify_weight_cap_certificate` rechecks a certificate
without an LP solver. On the :math:`(2, \infty)` process the caps are exactly :math:`1/2` and
:math:`3/4`. The sorted weights of any generator are then majorized by
:math:`(1/2, 1/4, 1/4)`, so :math:`C_g \ge \operatorname{H}(1/2, 1/4, 1/4) = 3/2`. The
ε-machine attains this, so :math:`C_g = 3/2` (computer-assisted proof).

When the reverse ε-machine is infinite there is no bidirectional machine, but the generator
caps still apply with the reverse causal state replaced by the window of the next ``L``
symbols (:func:`~sofic.generators.generator_problem_spec.window_spec`): a generator state
separates the forward causal state from the window, the window at time 0 is a function of
:math:`X_0` and the window at time 1, and the later window is screened off by the next
generator state. Every generator therefore satisfies the window equations, and
:meth:`_GeneratorProblem.from_spec <sofic.generators.coupled_generators._GeneratorProblem.from_spec>`
feeds them to :func:`~sofic.generators.generator_caps.generator_coverage_bound` for lower bounds
on :math:`C_g` from the forward ε-machine alone. On the :math:`(2, \infty)` process a
two-symbol window already gives :math:`C_g \ge 3/2`.

Supports without 2 × 2 rectangles
---------------------------------

When no two forward causal states share two reverse causal states, every rectangle in the support
of the causal-state joint lies in one row or one column, and
:func:`~sofic.generators.generator_bounds.one_wide_exact_common_information` computes
:math:`\op{G}{S^+ : S^-}` exactly. Every distribution on such a block is a product, so with the
blocks fixed the decompositions form a product of simplices (each cell's mass split among the
blocks covering it), on which the concave entropy is minimized at a vertex: a partition of the
support. Merging two blocks of one row or one column lowers the entropy, so some optimal partition
sends each cell to its row's block or to its column's block; the function enumerates these
assignments on each connected component of the support and also reports the least entropy with at
most ``k`` blocks. Kumar, Li & El Gamal :cite:`Kumar2014` show that some optimal separating
variable has at most :math:`|S^+||S^-|` values (and at most :math:`2^{\min(|S^+|, |S^-|)} - 1`);
their argument restricted to the support gives at most the number of support cells, which caps the
number of terms :func:`~sofic.generators.coupled_generators.separating_optima` tries, and merges and
null moves bring any decomposition down to :math:`K(P) = \lfloor (|\mathrm{supp}\, P| + \min(|S^+|, |S^-|)) / 2
\rfloor` terms (:doc:`separating_generators`). The partition solver is not from the literature.

The skip-one chain shows that a minimal generator can need more than
:math:`\min(|S^+|, |S^-|) + 1` states. It is the first-order Markov chain on :math:`\mathbb{Z}_6`
that moves from :math:`i` to :math:`i + 1` or :math:`i + 2`, with the twelve cells
:math:`(i, i + 1), (i, i + 2)` of the joint of consecutive symbols, in cycle order, weighing heavy,
heavy, light (light cells :math:`10^{-3}`, heavy ones :math:`0.1245`). Its causal states are the
previous and the next symbol, so :math:`|S^+| = |S^-| = 6`. Both persistent crypticities vanish, so
:math:`C_g = \op{G}{S^+ : S^-}`, and the support, a 12-cycle, has no :math:`2 \times 2` rectangle.
The optimum pairs adjacent heavy cells and keeps the light ones apart: 8 blocks and
:math:`C_g = 2.037622`, while 7 blocks give at least :math:`2.269805`. This family has no published
source.

Restricted global bound
-----------------------

:func:`~sofic.generators.global_bound.generator_global_bound` (optional dependency PySCIPOpt,
``pip install sofic[global]``) minimizes :math:`\H{\pi}` over the coupled generator equations for
``k`` states, written in the flows :math:`F(w, x, v) = \pi_w K(x, v \mid w)` and solved by SCIP's
spatial branch and bound :cite:`Achterberg2009,Vigerske2018` through PySCIPOpt :cite:`Maher2016`.
With the bidirectional edge chain it adds two-step path flows and the two-step edge law. The
weights are sorted, so caps on the heaviest states enter as linear cuts, and an objective limit
(the entropy of a verified generator) lets SCIP prune. The result bounds the state entropy of every
generator with at most ``k`` states and says nothing about larger ones. It is therefore a lower
bound on :math:`C_g` only once the number of states of a minimal generator is bounded. No general
bound is known, and :math:`\min(|S^+|, |S^-|) + 1` is not one: the skip-one chain (under
`Supports without 2 × 2 rectangles`_) needs :math:`|S^+| + 2` states. When both persistent
crypticities vanish (cryptic order at most 1), ``k`` equal to :math:`K(P)` suffices
(:doc:`separating_generators`). It is also numerical, not a certificate: the equations with data on the right-hand side are relaxed by ``data_tol``, and SCIP
works to its own tolerances, and its presolve and optimization-based bound tightening can go wrong
(``presolve=False, obbt=False`` gives a second opinion; see :doc:`separating_generators`).
:func:`~sofic.generators.global_bound.separating_global_bound` is the static version, over separating
decompositions with ``k`` terms. SCIP's solution satisfies necessary conditions only;
:func:`~sofic.generators.hmm_search.hmm_generator_search` with ``initial`` polishes it into a
generator. On the binary process with symbol matrices
:math:`T^{(0)} = \begin{pmatrix} p & 0 \\ q & 0 \end{pmatrix}` and
:math:`T^{(1)} = \begin{pmatrix} 0 & 1 - p \\ 1 - q & 0 \end{pmatrix}` at
:math:`(p, q) = (0.836, 0.788)`, where :math:`C_\mu = 0.5866`, the bound at ``k = 3`` is
:math:`0.15563` (11 s), and the polished solution is a generator of entropy :math:`0.15571`: no
generator with at most 3 states does :math:`10^{-4}` bits better.

The pipeline runs it with ``global_bound=True``, after the belief-free search, for
``global_bound_states`` states (default :math:`K(P)`, lowered until the model has at most 2,000 SCIP
variables; a bound on the states of a minimal generator only at cryptic order at most 1).
It gets half the remaining
budget (120 s without a budget) and the best candidate's entropy as its objective limit, and it is
skipped above 2,000 SCIP variables. The value goes to ``restricted_lower`` and ``restricted_states``.
It enters ``lower`` (as a numerical bound) only at cryptic order at most 1 with :math:`K(P)` states;
otherwise ``certified`` is unchanged. The polished solution is offered as a candidate. The
two-step flows and this restricted use are not from the literature.

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
.. autofunction:: sofic.generators.hmm_search.hmm_generator_search
.. autofunction:: sofic.generators.hmm_search.coupled_seed_from_hmm
.. autofunction:: sofic.generators.simplex_chains.simplex_chain_search
.. autoclass:: sofic.generators.simplex_chains.SimplexChainResult
.. autofunction:: sofic.generators.global_bound.generator_global_bound
.. autoclass:: sofic.generators.global_bound.GlobalBoundResult
.. autofunction:: sofic.generators.global_bound.separating_global_bound
.. autoclass:: sofic.generators.global_bound.SeparatingBoundResult
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
.. autofunction:: sofic.generators.exact_ci_bounds.heaviest_term_bound
.. autofunction:: sofic.generators.exact_ci_bounds.generator_heaviest_term_bound
.. autofunction:: sofic.generators.exact_ci_bounds.lifted_heaviest_term_bound
.. autofunction:: sofic.generators.exact_ci_bounds.capped_coverage_entropy
.. autoclass:: sofic.generators.exact_ci_bounds.HeaviestTermBoundResult
.. autofunction:: sofic.generators.generator_caps.certified_generator_weight_cap
.. autofunction:: sofic.generators.generator_caps.verify_weight_cap_certificate
.. autoclass:: sofic.generators.generator_caps.WeightCapCertificate
.. autofunction:: sofic.generators.certificates.certified_lp_minimum
.. autofunction:: sofic.generators.certificates.dual_bound
.. autofunction:: sofic.generators.certificates.certified_psd
.. autofunction:: sofic.generators.generator_problem_spec.window_spec
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
.. autofunction:: sofic.generators.generator_bounds.one_wide_exact_common_information
.. autoclass:: sofic.generators.generator_bounds.OneWidePartitionResult
   :members:
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
