# Changelog

## Unreleased (0.4.0)

A second correctness review: every fix below was reproduced against a
brute-force reference or a closed-form value and has a regression test.

### Coverage bound: least entropy under the caps

- `coverage_entropy_lower(caps)`, the last step of `coverage_bound`, `generator_coverage_bound` and the pipeline's
  coverage stages, returns the least entropy of a distribution whose `j` heaviest weights sum to at most
  `caps[j - 1]` (was: the entropy of the increments `(c_1, c_2 - c_1, ..., 1 - c_k)`, the same value when the
  increments do not increase, and otherwise lower and not monotone in the caps). Given the `k` heaviest weights the
  rest is copies of the `k`-th and a remainder; for each number of copies the entropy is concave, so the minimum is
  at a vertex of the polytope of the heaviest weights or where an edge leaves no remainder. With more than `Q`
  copies every weight from the `k`-th on is at most `r / (Q + 1)`, where `r` is the mass outside the heaviest
  `k - 1`, which bounds the entropy of all larger counts from below; `Q` doubles until that bound passes the least
  entropy found. On 400 random vectors of up to 3 caps it matched vertex enumeration over distributions on up to
  32 atoms, 54 of them with optima on more than 8 atoms. On caps `(0.4720301, 0.7269834)` (generator caps of a
  binary 3-state process) the bound rises from 1.5252655 to 1.5399102.
- Caps above 1 no longer make it overstate: the increments were normalized by their sum, so caps
  `(2/3, 1.125)` gave 0.975 against the least entropy `H(2/3, 1/3) = 0.918`. Caps are clipped to `[0, 1]` and
  replaced by their suffix minimum (`S_j <= S_{j+1}`).

### Simplex-chain generator search

- New `sofic.generators.simplex_chains`: `simplex_chain_search(bidir)` constructs exact one-sided generators. A
  reverse-sufficient generator is a finite law on the simplex of beliefs on the reverse causal states that
  dominates its image under the belief updates in convex order (Strassen 1965). Writing each updated belief as a
  convex combination of a pool of atoms gives a chain whose closed classes are exact generators; the search
  alternates linear programs over those representations with gradient steps on the atoms, on both sides, starting
  from the ε-machines. Every iterate passes a Tzeng word check, and at cryptic orders up to 4 the atoms stay on the
  faces of the synchronizing complex. Results are `SimplexChainResult`s (`pi`, `K`, `atoms`, `word_error`).
- `generative_model` runs it right after the exact path (`simplex_chains=True`, stage `"chains"`) with a quarter of
  the remaining budget, or 2 s per side without one, and stops once a generator meets the lower bound. On 306
  archive processes it went below the best known upper bound on 42, by up to 0.295 bits (`n3-t138-d4`).

### Generator cardinality

- Minimal generators can need more than `min(|S+|, |S-|) + 1` states. The skip-one chain (the first-order Markov
  chain on `Z_6` moving `i -> i + 1` or `i + 2`, with light cells `1e-3`) has `|S+| = |S-| = 6`, vanishing
  persistent crypticities, and `C_g = G[S+ : S-] = 2.037622` with 8 states; 7 states give at least 2.269805. The
  remarks in `global_bound` and the docs that every certified case needs at most `|S+| + 1` states are replaced,
  and `global_bound_states` (default `min(|S+|, |S-|) + 1`) is documented as a convention, not a bound.
- New `generator_bounds.one_wide_exact_common_information(P)`: exact `G[X : Y]` when every rectangle in the support
  lies in one row or one column, by enumerating row-or-column assignments of the cells on each connected component;
  `by_terms[k]` gives the optimum with at most `k` values. The skip-one chain is its regression test.
- `separating_optima` tries at most `min(m, 8)` terms, `m` the number of support cells: Kumar, Li & El Gamal's
  argument restricted to the support bounds the terms of some optimum by `m` (was `|S+| |S-|`).
- Citations: `Feldman1998b` now points to Feldman & Crutchfield's "Discovering Noncritical Organization" (Entropy
  2022, SFI working paper 1998); `Travers2010` to J. Stat. Phys. 145 (2011). `structural_information` cites
  Feldman & Crutchfield (2003) and Travers & Crutchfield's asymptotic synchronization paper (2011);
  `predictability_gain` notes that `h_mu(L) - h_mu` is Bialek et al.'s learning curve and Crutchfield & Feldman's
  `r(L)`, whose "predictability gain" is the first difference `h_mu(L) - h_mu(L - 1)`.

### Restricted global bound (SCIP)

- New `sofic.generators.global_bound`: `generator_global_bound(bidir, k)` minimizes the state entropy over the
  coupled generator equations for `k` states with SCIP's spatial branch and bound (optional dependency,
  `pip install sofic[global]`), with two-step path flows and the two-step edge law when the bidirectional edge
  chain is known, sorted weights, optional caps and an objective limit. `lower` bounds the state entropy of
  every generator with at most `k` states, and only those: it is a lower bound on `C_g` only once the number
  of states of a minimal generator is bounded (open). It is numerical, not a certificate. Each entropy term is
  evaluated at `max(pi_w, pi_min)`, and the overestimate (`k pi_min log2(1 / pi_min)`) is subtracted, so
  states of tiny weight cannot slip under the bound.
- `hmm_generator_search(..., initial=[T])` polishes given `k`-state HMMs: their rounds keep the entropy term
  off and move them to a nearby generator (from a near-generator, the entropy descent stalls at an infeasible
  point). This turns a SCIP solution, which satisfies necessary conditions only, into a verified generator.
- `generative_model(..., global_bound=True, global_bound_states=k)` runs the bound after the belief-free
  search (default `k = min(|S+|, |S-|) + 1`, half the remaining budget, the best candidate's entropy as
  objective limit, models up to 2,000 variables), records `restricted_lower` and `restricted_states`, and
  offers the polished solution as a candidate. `lower` and `certified` are unchanged. At
  `(p, q) = (0.836, 0.788)` in the binary two-state family the bound at `k = 3` is 0.15563 (11 s), and the
  polished solution is a generator of entropy 0.15571, 0.43 bits below `C_mu`.

### Belief-free generator search

- New `sofic.generators.hmm_search`: `hmm_generator_search(machine, k)` optimizes the symbol matrices of a
  `k`-state HMM directly (softmax rows), minimizing the state entropy subject to exact word probabilities up
  to length `k + |S+|`, which implies equality on all words (Tzeng 1992). It uses an augmented Lagrangian with
  analytic gradients (forward recursion over the word tree, stationary law differentiated through the
  fundamental matrix): feasibility first, then entropy descent. `coupled_seed_from_hmm` recovers the coupled
  form `(pi, Q, R, K)` from the product chains with the causal states.
- `generative_model` runs it after the coupled search (`hmm_search=True`, `hmm_search_starts=8`) for
  `k = |S+| .. |S+| + 2` with half the remaining budget, or two values of `k` and two starts without a budget.
  The coupled searches parameterize states by their beliefs and miss generators whose states' pasts or futures
  are not belief-mixtures of the causal states'. On the 96 open brackets of the binary 3-state dataset this
  search lowered seven upper bounds, e.g. `n3-t122-d2` from `C_mu = 0.9037` to `0.7956` and `n3-t140-d3`
  from `1.5142` to `1.4010`.

### Generator heaviest-term bound by default

- `generative_model` now runs `generator_heaviest_term_bound` last, once the best generator is
  verified, aiming at its entropy with the remaining budget (`generator_heaviest_term=True` is the new
  default). With a time budget it runs on joints with up to 14 causal states; without one it stays at
  `|S+| + |S-| <= 7` and 2,000 boxes. On the 106 open `C_g` brackets of a dataset of binary 2- and
  3-state ε-machines, 900 s per process raised the lower bound on 48 of 105 and closed 9 (median
  bracket width 0.066 → 0.039 bits). Some of the closed ones cannot be certified by caps on the
  heaviest weights at all, since another generator has a heavier state.
- `generator_heaviest_term_bound` limits each generator-cap LP to the time left before its
  `deadline`. A single degenerate HiGHS solve previously ran more than half an hour past a 900 s
  deadline (`n3-t045-d3`); it now returns on time.

### Finite-state rounded HMM filters

- `quantized_filter` builds the reachable log-odds-grid observer of Feng & Meng
  (arXiv:2611.13760) for positive-transition Moore HMMs, with bit-valued event,
  uniform-word, and total-forward-KL budgets. It preserves the exact initial
  label, permits zero emissions, exposes label-capacity/error metadata, and
  separates recursively rounded generation from posterior-initialized private
  simulation. Bounds are theorem-derived in real arithmetic, explicitly not
  interval-certified (`bounds.rigorous=False`). Tests reproduce the paper's
  log-grid state counts and check finite-horizon losses against exact enumeration.

### ε-machine construction

- Processes whose hidden state no finite word reveals (e.g. a period-2 process whose phase stays
  uncertain) synchronize only asymptotically. Their mixed-state presentation either closed on hundreds of
  beliefs within round-off of the simplex boundary (rounding acts as a reflecting barrier for the
  posterior's random walk) or exceeded `max_states`, so `EpsilonMachine.from_hmm` returned a huge
  machine or raised `MixedStateExplosionError` although the ε-machine is finite. It now seeds the
  presentation from a limit belief (the heaviest near-boundary belief, or the belief after filtering a
  long sample) with sub-`1e-8` entries set to zero, and accepts the result only if it generates the
  source process exactly (difference-automaton test). On the 138 minimal binary 3-state topologies,
  9 reverse-machine "explosions" are now 3-state reverse ε-machines, and finite/infinite now agrees
  with `reverse_epsilon_machine_is_finite` on every topology.

### Canonical NFAs as generators and presentations

- `canonical_rfsa_hmm`, `maximized_prime_atomaton_hmm` and `atomaton_hmm` (new module
  `sofic.generators.canonical_generators`; also `EpsilonMachine` / `BidirectionalEpsilonMachine` methods) build the
  stochastic analogues of the canonical RFSA, the maximized prime átomaton and the átomaton as generators of a
  process with a finite bidirectional machine. The RFSA analogue's states are the forward causal states whose
  reverse beliefs are vertices of their convex hull (future morphs that are not mixtures of the others); the
  prime-átomaton analogue is the same construction in reverse time on the forward beliefs of the reverse causal
  states, time-reversed back. Both are linear programs and generally non-unifilar.
- `generative_model` offers both as candidates (`canonical RFSA generator`, `maximized prime átomaton
  generator`) and `coupled_search` starts from them. On the binary 3-state dataset the prime-átomaton analogue
  compresses the reverse machine on 112 of 620 processes and is the best generator known on 14, lowering the
  best upper bound on `C_g` on three (e.g. 1.2809 to 1.0994).
- `SoficShift.canonical_rfsa_presentation`, `maximized_prime_atomaton_presentation` and `atomaton_presentation`
  (new module `sofic.shifts.canonical_presentations`) present a sofic shift by the canonical NFAs of its factor
  language, trimmed to the essential states.

### Distromaton, xor automata, and the universal automaton

- New module `sofic.automata.canonical.closures` completes the canonical NFAs of Myers, Adámek, Milius & Urbat
  (CMCS 2014) and Zetzsche, van Heerdt, Sammartino & Silva (MFPS 2021): `Distromaton` (join-irreducible
  intersections of residuals), `minimal_xor_automaton`, `xor_caba_automaton` (the new acceptor of Zetzsche et al.,
  on a basis of the Boolean closure read as a GF(2) space), `UniversalAutomaton` (Lombardy & Sakarovitch),
  `minimal_structured_dfa` (the minimal DFA whose states form a semilattice, distributive lattice, Boolean algebra
  or GF(2) space), and `is_intersection_closed` (Myers et al.'s criterion for the canonical RFSA to be a
  state-minimal NFA). Every closure element is stored exactly as a set of atoms, i.e. of states of the minimal
  DFA of the reversed language, so derivatives are preimages and all constructions are exact.
- New `sofic.automata.xor.XorAutomaton`: NFA-shaped automata that accept by parity of accepting paths, with the xor
  subset construction (`determinize`), `reverse` and serialization. It is not a `LabeledAutomaton`, whose language
  operations assume NFA acceptance.
- Tests reproduce the papers' examples ((a+b)\*a and (a+b)\*b(a+b)^n), check the distromaton against Myers et al.'s
  second construction (the reverse minimal DFA ordered by inclusion), the minimal xor automaton's size against the
  GF(2) rank of a brute-force Hankel matrix, and Corollaries 3.21 and 4.6. Two corrections to the papers: the
  xor-CABA example of Zetzsche et al. (Fig. 10) uses four linearly dependent vectors where a basis has three, and
  Myers et al.'s remark that {aa} is not intersection-closed contradicts their Example 4.3 (singletons are).
- Shifts: `distromaton_presentation`, `universal_presentation` (also `SoficShift` methods), and
  `follower_sets_intersection_closed`, which certifies that `canonical_rfsa_presentation` has the fewest states of
  any presentation (brute-force checked against every smaller labeled graph). Canonical presentations now trim the
  automaton before building the graph, which the new automata need (their `Σ*` and `∅` states) and which leaves
  the existing presentations unchanged.
- Generators: `canonical_rfsa_hmm` and `maximized_prime_atomaton_hmm` are now documented as size-minimal among
  generators whose state morphs are mixtures of causal-state morphs, the stochastic counterpart of the canonical
  RFSA's minimality, with proof; the minimal xor automaton is cross-referenced as the GF(2) counterpart of
  `minimal_quasi_realization`.

### Generator verification

- `GenerativeModelResult` records the returned generator's word-process deviation from the source
  (`reproduction_error`, from the exact difference-automaton check) and its `structure`: state count,
  weights, whether the state is a function of `(S+, S-)` (`functional`), and the largest violation of the
  lifting conditions (`lifting_residual`; `lifted` when it vanishes). `certified` now also requires
  `reproduction_error <= 1e-6`, so a closed bracket is never reported on an approximate generator.
- `GenerativeModelResult.decomposition` returns the generator as JSON-ready lists (state names, forward and
  reverse causal-state labels, symbols, `pi`, `Q`, `R`, and the model's kernel `K[w][x][w']`), so minimal
  generators can be saved and studied later.

- The one-step generator equations (edge law, forward stationarity, summed future-side consistency) are
  necessary but not sufficient: a kernel with kernel-LP violation 0 can emit the wrong word process,
  because nothing forces a state's belief `R_w` to be the generated future's actual conditional. On survey
  process `((1,·),(2,0),(1,0))` an 11-state example matches words up to length 3 and is off by 0.024 at
  length 8. `generator_kernel`, the result flags (`n_g_optima_generating`, `separating_variables_generate`,
  `cg_exceeds_g`) and the docs now say so; positive violations still rule a variable out.
- Candidate verification (`_reproduction_error`, used by every search that returns a generator) adds a
  difference-automaton test (`_equivalence_error`): breadth-first search over words keeping those whose
  difference vectors are independent (at most `n1 + n2`, Schützenberger / Tzeng), with the largest
  word-probability deviation over the visited words. The fixed cutoff (at most length 8) alone could
  accept a candidate that matched short words and drifted later; two HMMs are only guaranteed equal once
  they agree up to length `n1 + n2 - 1`.

- `stationary_distribution_hmm` solves chains with more than 2,000 states sparsely (sparse transition
  matrix from the edges, one balance equation replaced by the normalization, `spsolve`). The dense
  path built the full matrix and ran `matrix_rank` and `eig` on it, $O(n^3)$, which never finished on
  a 50k-state reverse machine. Symbolic weights, several closed classes, or a failed residual check
  fall back to the dense path and its warnings.
- The retry ladder and `try_build_bidirectional` reuse the reverse and bidirectional machines built by
  the preflight instead of inferring and building them a second time (about half the build time).
- `build_bidirectional_with_ladder` and `try_build_bidirectional` take `max_reverse_states`: a ladder
  build whose truncated reverse machine is larger is classified as an explosion, so surveys fall back
  to window bounds. Off by default. A 4-state survey process built at the 50k cap with tens of
  thousands of reverse states, and the pipeline then spent its whole budget setting up the problem.

- `build_bidirectional_epsilon_machine` prunes joint states to a fixed point: dropping states
  outside the stationary support could leave others with incomplete outgoing rows, so builds from
  truncated reverse machines failed with "joint masses from ... sum to 0.86". Restriction to the
  stationary support and to states with complete rows now alternate until neither removes anything.
- The retry ladder takes a `time_budget` (`build_bidirectional_with_ladder`, `try_build_bidirectional`,
  `resolve_bidirectional`), and `generative_model` / `minimal_wyner_model` charge the build to their
  own `time_budget`. Mixed-state construction is not interruptible, so the ladder skips a cap when the
  previous cap's time times the squared cap ratio would overrun, and sets `budget_exhausted`. On a
  4-state survey process the 10k and 50k caps took 9 s and 86 s and the 200k cap did not finish in
  15 minutes, so the ladder ran past the survey's hard timeout before any bounds were reported.

### Exact common information: heaviest-term bound

- `heaviest_term_bound` (new module `sofic.generators.exact_ci_bounds`) gives a certified lower bound on
  `G[X : Y]` by branching on the heaviest term of the decomposition and bounding the remainder by the
  coverage of its caps, with every remaining atom at most `t/(1 - t)` (`capped_coverage_entropy`, which
  takes the least concave majorant of the caps). An optimum has at most `2^min - 1` terms (Kumar, Li &
  El Gamal), so the heaviest weighs at least the reciprocal. `generative_model` runs it after the exact
  stage when the certified lower is below the estimate and `|S+| + |S-| <= 7`. On the 3 × 3 survey
  joints `((0,1),(2,·),(0,1))`, `((0,1),(0,2),(0,1))` and `((0,1),(2,1),(0,·))` it certifies `G` to
  within 1e-5 of the estimate (0.9335, 0.9243, 0.4717; was 0.9118, 0.9056, 0.4436), which closes the
  first bracket.
- `generator_heaviest_term_bound` bounds `C_g` the same way with the heavy term restricted to generator
  states: `heaviest_term_bound(weight_cap=...)` caps each box's weight by the one-explicit-state
  generator-cap LP over the box of beliefs, and prunes boxes the LP rules out. On `((0,1),(0,2),(0,1))`
  it certifies `C_g >= 0.928 > G = 0.924`, but only after ~220k boxes, so the pipeline runs it on
  request (`generator_heaviest_term=True`).
- `lifted_heaviest_term_bound` bounds the state entropy of every *lifted* generator from below. A lifted
  generator's heaviest state has channel `t q_a r_b / p(a, b)` at both ends of each edge; coarse-graining
  the other end to "is that state" turns each lifting condition into a condition on `(q, r)` alone, checked
  on boxes by interval arithmetic inside `heaviest_term_bound`. On `((0,1),(0,2),(0,2))` it shows every
  lifted generator has entropy >= 1.4815 while `C_g = G = 1.4225`.

### Generator caps

- `generator_weight_cap`, and the coverage and heaviest-term bounds built on it, dropped every box LP that HiGHS
  called infeasible, and HiGHS's presolve calls feasible box LPs infeasible: on `((1,0),(2,2),(1,0))` it dropped the
  box holding a 3-state generator's heaviest state, and the coverage bound came out at 1.0459, above that
  generator's entropy 1.0420. An infeasible verdict is now solved again without presolve, and a box is dropped only
  with an exact Farkas certificate; the bound meets the generator's entropy to `3e-10`. `certified_lp_minimum`
  also retries without presolve when its Farkas test fails.

- With a deadline, each branch-and-bound child LP of `generator_weight_cap` gets the time left as a
  HiGHS `time_limit`; a cut-off LP keeps its parent's bound, so the cap stays valid. On a 4-state
  survey process (4 × 36 causal states) one degenerate node LP had run more than 8 minutes past a
  90 s stage deadline; the stage now ends on time (92.7 s).

- `generator_weight_cap` adds the forward analogue of the reverse joint-moment rows: per-state forward
  stationarity times the state's own reverse belief gives
  `T_h(a', b) = sum_{x, g, a : f(a, x) = a'} Gt[x, g, h, a, b]`, which every generator satisfies.
  Same cost; two-state caps drop (e.g. 0.918 -> 0.907 at 600 nodes) and the coverage bound on the
  survey processes `((1,1),(2,0),(1,0))` and `((1,·),(2,0),(1,0))` rises 1.261 -> 1.277 and
  1.175 -> 1.184.

### Anytime pipeline on large joints

- `generative_model` skips the dit exact- and Wyner-common-information fallbacks when the joint has
  more than 30 causal-state pairs (logged as a skipped `dit fallbacks` stage). dit's optimizers take no
  deadline: on finished 4-state survey cases with 30-70 pairs they used 190-760 s of a 900 s budget,
  and on a 95-pair case one call ran past the survey's hard timeout. Every 3-state survey joint has at
  most 21 pairs, so the benchmark is unaffected.
- `memory_lift_search` stops a start in progress at its `deadline` (keeping the last finished penalty
  stage) instead of only refusing to begin new starts. Its first start used to run all four
  finite-difference L-BFGS-B stages whatever the deadline, more than 14 minutes on a 95-pair joint.
- The line-split seeds' ALM polish (`_optimize`, `_optimize_coupled`) stops at the pipeline deadline
  and returns its last finished step; one polish had run more than 10 minutes past the budget on a
  144-pair joint.
- `hmm_generator_search` checks its `deadline` after every L-BFGS-B iteration, not only between starts and
  rounds, so a run stops within one iteration of it (an end point cut short is kept if it is feasible). One
  uninterrupted start at `k = 6` on a 3-state binary process ran 19 s past a 0.5 s deadline. The stop is a
  `StopIteration` raised in the optimizer callback, which needs SciPy 1.11, now the minimum.
- Sparse word distributions (`word_probabilities`, the default) extend only prefixes of positive probability,
  so their cost follows the number of allowed words instead of `|X|^L`, and `generative_model`'s reproduction
  check uses them. On the butterfly process (8 symbols, 1088 allowed words of length 8) the check ran about
  1900 s of a 60 s budget; `generative_model(butterfly_process(), time_budget=60)` now certifies
  `C_g = log2 5` in about 4 s.
- `generative_model` skips both coverage stages once the cheaper lower bounds meet the entropy of an
  ε-machine, which then certifies, and at cryptic order at most 1 runs the generator coverage bound only when
  the exact path leaves the bracket open. On the 510 dataset processes of cryptic order at most 1 the
  generator coverage bound took about 17 s each, 96% of the pipeline's time, and never supplied the lower
  bound; with these changes the 510 runs take 2.6 minutes instead of 2.5 hours, with the same brackets.

### Support-type bound speed

- Exact vertex enumeration of the support-type polytope (`support_type_bound`, the certified flow
  bounds) first drops rows that the others imply, each proved redundant by an exact LP dual bound,
  then screens candidate active sets in batched floating point and verifies each candidate in exact
  arithmetic. A 4-state survey support with 6 maximal rectangles went from hours (all
  `C(132, 5)` row subsets in `Fraction` arithmetic) to 0.3 s; 7 rectangles take about 7 s.
  
### Mixed-state presentation speed

- `build_mixed_state_presentation` indexes numeric beliefs by a projection onto fixed positive weights
  instead of their first coordinate. Beliefs equal to within the matching tolerance still land in the
  same window, but many beliefs sharing a first coordinate (often 0) no longer make every lookup scan
  all states. Reverse inference on a 4-state process reaches a 5,000-state cap in 2.5 s instead of
  135 s; reverse machines of the 100 finite 3-state survey processes are unchanged.

### Anytime generator search

- The remaining search loops take a ``deadline`` too: `coupled_search` (the ε-machine seed always
  runs), `memory_lift_search`, `targeted_line_split_seeds` (no new start or hop, and only the best
  decomposition is polished, once it passes) and `universal_generative_optimizer` (no larger state
  count once a generator is verified). The pipeline gives them 50%, 20%, 80% and 60% of the remaining
  budget, so with ``time_budget`` set every stage either finishes or stops between iterations.

- `generative_model`, `generative_model_from_bidirectional` and `minimal_wyner_model` take
  ``time_budget`` (seconds). Once it is spent, the remaining stages are skipped (each logged as
  ``skipped: time budget exhausted``) and the best verified bracket so far is returned, with
  `GenerativeModelResult.budget_exhausted` set. Every ``stages`` entry now records ``elapsed``
  seconds, so slow stages are visible.
  The expensive inner searches take a ``deadline`` too (`generator_weight_cap`,
  `generator_coverage_bound`, `separating_optima`, `bilevel_search`), each stopping with a valid
  result; the pipeline gives them 30%, 25% and 15% of the remaining budget.

### State labels

- `sofic.states.sequential_labels` no longer stops at 26 states: after ``A``–``Z`` it continues
  ``AA``–``ZZ``, then ``AAA``–``ZZZ`` and so on (bijective base 26, like spreadsheet columns), and
  `letter_label_index` / `next_sequential_label_index` read multi-letter labels. Bidirectional
  builds whose reverse machine has more than 26 states used to fail with "sequential_labels
  supports at most 26 states A–Z"; the cover, residual-automaton and ε-machine constructions
  that fell back to integer labels above 26 states now use letters too.

### Generator search reliability

- `certificates` module: rigorous LP/SDP lower bounds from approximate duals in exact rational
  arithmetic (`dual_bound`, `certified_lp_minimum` with Farkas certificates from an elastic LP,
  `certified_psd` for positive-definite rounding of SDP duals).
  `generator_caps.certified_generator_weight_cap` certifies every branch-and-bound leaf this way
  (box tightening now rounds outward), optionally against rationalized problem data, and
  `verify_weight_cap_certificate` rechecks a certificate without an LP solver. On the
  `(2, inf)` process the caps are exactly 1/2 and 3/4, so `C_g = 3/2` (computer-assisted proof).

- `generator_problem_spec.window_spec` and `_GeneratorProblem.from_spec`: generator equations from
  edge tensors alone, with the reverse causal state replaced by the next ``L`` symbols. Every
  generator satisfies the window equations, so generator caps and coverage on a window spec are
  lower bounds on `C_g` that need only the forward ε-machine, for processes whose reverse
  ε-machine is infinite. On `(2, inf)` a two-symbol window gives `C_g >= 3/2`.

- `targeted_line_split_seeds`: iterated local search after COBYLA (``hop_chains`` best decompositions,
  ``hops`` perturbations of the broad terms' logits each, keeping improvements), and only the
  ``polish_best`` lowest-entropy decompositions get the coupled-ALM polish. On the widest survey
  bracket it reaches the known 0.970-bit generator in 8 of 8 seeds (COBYLA alone: 0.978 to 1.10).
  The kernel LP is now assembled with `einsum` (same LP, 1.5-3.5x faster to build).

- `targeted_line_split_seeds`: searches the line-split family directly (broad terms at their
  largest fitting weight plus point-mass row/column states, an exact decomposition fixed by the
  broad terms' beliefs). From each exact-CI optimum's heaviest terms, Nelder-Mead reaches kernel-LP
  feasibility and COBYLA then minimizes the state entropy subject to staying feasible. Pipeline
  stage ``targeted`` (default on, skipped for ``minimal_wyner_model``) runs only when the bracket is
  still wider than 0.01 bits. On the widest survey bracket, ``[0.906, 1.371]``, it finds a
  0.98-bit generator (a 0.970-bit one is known).

- `line_split_seeds`: coupled seeds that keep the one or two heaviest terms of each exact
  common-information optimum and split the remainder into point-mass states (one per row or
  column of the joint), with kernels from the kernel LP. The pipeline polishes each from
  several logit floors (saturated logits freeze the softmax gradients); `minimal_wyner_model`
  skips them. On the survey's 20 open
  brackets this lowered the upper bound in 3 cases, e.g. ``[1.234, 1.397] -> [1.234, 1.237]``.

- `generator_caps` module: `generator_weight_cap` caps the total weight of a generator's ``k``
  heaviest states (``k`` explicit states plus one aggregate, generator equations on group
  moments, McCormick envelopes over branched belief boxes); `generator_coverage_bound` feeds the
  caps into the coverage bound. These are the first full-support lower bounds above
  ``G[S+ : S-]``. New pipeline stage ``generator_caps`` (default 2 explicit states, with
  ``coverage="certified"``). On the 43-case survey (v5 brackets): certified 11 -> 18, median
  width 0.0164 -> 0.0007 bits, lower bound above the ``G`` estimate in 24 cases. On the
  ``(2, inf)`` process the caps are ``(1/2, 3/4)``, giving ``C_g >= 3/2`` to LP tolerance.

- **Fixed:** `generative_model` skipped the coverage bound on full-support joints. It ran only
  with more than 7 maximal rectangles, but a full support has one, and there the support-type
  bound is 0, so the lower bound fell back to the excess entropy. On the 43-case survey of
  3-state processes with positive persistent crypticity, running it raises the lower bound in
  26 cases and cuts the median bracket width from 0.285 to 0.044 bits (mean 0.449 to 0.110).
- `coverage` now defaults to ``"certified"`` (cheap, and a valid lower bound; ``"heuristic"``
  is not).
- `mu_hyperplane_cap`: certified cap on ``mu_{m-1}`` (``m`` the smaller side) by branch and bound
  over the hyperplane containing the columns of a rank-``(m-1)`` matrix; exact for ``m = 3``.
  `mu_caps` uses it for ``mu_2`` on 3-state sides, lowering the median survey bracket width
  from 0.044 to 0.0073 bits on the same uppers.
- **Fixed:** `exact_common_information_estimate` (and the exact-CI optima tested for
  generation) could come from dit's channel even when polishing left it short of exact
  separation, reporting an entropy below the true ``G[S+ : S-]`` and below the certified lower
  bound. Such channels are now discarded (``I[S+ : S- | W] > 1e-9``).
- The functional stage of `generative_model` uses the exact `functional_cover_search` at any
  joint size instead of the old partition search (exhaustive up to 9 pairs, greedy beyond).

### Added (generator objectives)

- `minimal_wyner_model`: generator search ranked by ``I[W : S+, S-]`` (Wyner analogue of
  `generative_model`), with the lifted Wyner model and unconstrained Wyner optimum as extra
  candidates; ``lower`` is the excess entropy.
- `minimal_functional_generator`: exact least-state-entropy generator whose state is a function of
  ``(S+, S-)``, by least-entropy exact cover with product rectangles plus no-good cuts
  (`functional_cover_search`, `product_rectangles`); certified at any size unless ``max_covers``
  runs out.
- `GenerativeModelResult.objective` (``"entropy"``, ``"wyner"`` or ``"functional"``); both new
  searches are also methods on `EpsilonMachine` and `BidirectionalEpsilonMachine`.

### Added (Phases A & B)

- ``coverage_bound`` with ``k_used = min(k_max, na, nb)`` caps; ``coverage_k_max`` and
  ``coverage_certified_only`` on ``generative_model`` / ``generative_model_from_bidirectional``.
- ``support_exact_cover_bound`` (A3) when each joint cell lies in a unique maximal rectangle.
- ``generator_state_cap`` (B4) and ``systematic_coupled_seeds`` (B2) wired into the pipeline.
- Survey driver uses ``generative_model_from_bidirectional`` after bidirectional build.

- `generative_model` / `GenerativeModelResult`: bracket search for generative
  complexity with support, flow, and heuristic coverage lower bounds;
  enumeration of exact-common-information optima with `generator_kernel`;
  optional `bilevel_search`; coupled and memory-lift uppers; gap flags
  (`cg_exceeds_g`, `n_g_optima_generating`). `BidirectionalEpsilonMachine.generative_model()`.
- `coverage_bound` helpers for partial-sum (coverage) lower bounds on full supports.
- Entropy-ordered separator sweep (`separating_decomposition_channels`,
  `first_generating_separator`) in `generative_model` when no `G` optimum generates.
- Certified coverage caps on any support: exact `mu_1` by vertex enumeration of
  ``{p_a + q_b <= log M_ab}`` (`mu1_vertex`, used by `mu1_optimize`); singular-value
  tail caps ``mu_k <= 1 - sum_{i>k} sigma_i`` (`mu_k_spectral_cap`); `mu_caps` takes the
  minimum of these, ``k mu_1``, and ``mu_i + mu_{k-i}``. `generative_model(..., coverage="certified")`.
  On the interleaved product ``D (x) D`` at ``eps = 9/20`` the certified coverage bound is
  0.787 bits (caps ``mu_2 <= 0.9725``, ``mu_3 <= 0.9975``).
- RLL flow-bound certification (`flow_bound_result`, `rll_flow_bound_certified`); RLL
  `generative_model` sets `lower_certified` when the flow stage wins.
- Bidirectional build preflight and retry ladder (`preflight_bidir`,
  `try_build_bidirectional`, `build_bidirectional_with_ladder`) for survey
  topologies with truncated or exploding reverse MSP; optional
  `reverse_max_states` on `BidirectionalEpsilonMachine.from_forward` /
  `infer_reverse_epsilon_machine`.
- `GeneratorProblemSpec` / `spec_from_generator_problem` and a stub
  `_GeneratorProblem.from_edge_tensors` for future edge-law-only search with
  explicit truncation metadata.
- `universal_generative_optimizer`: direct ``min H[pi]`` over coupled generator
  equations (edge + forward + reverse) with memory-lift and ``G`` seeds;
  `compare_universal_to_pipeline` for benchmarking against the full pipeline.
- **`generative_model`**: primary entry — forward ε-machine or bidirectional in,
  bidir build (preflight / ladder), full bracket search, universal coupled stage,
  and a final **reliability summary** in ``stages``.
- `generative_model_from_bidirectional`: same pipeline when the bidirectional
  machine is already built.
- `ultimate_generative_model`: deprecated alias for ``generative_model``.
- Coupled search: joint block ALM by default; optional **dual strategy** (retry
  with alternating ``K`` LP if joint ALM stays infeasible); ``alternating`` and
  ``dual_strategy`` parameters on ``coupled_search``.
- `scripts/benchmark_generative_robustness.py`: timing and bracket metrics on
  golden mean, RLL, and interleaved benchmarks.
- Coupled search: **masked / block-weighted** generator residuals and **block ALM**
  (plus alternating ``K`` LP when requested).

### Breaking changes

- ``generative_model(forward_or_bidir, ...)`` is now the full pipeline (formerly
  ``ultimate_generative_model``). The bidirectional-only pipeline is
  ``generative_model_from_bidirectional(bidir, ...)`` (formerly ``generative_model(bidir, ...)``).
- `minimal_generative_model`, `wyner_generative_model`,
  `functional_generative_model` and `generative_complexity()` optimize only
  over lifted generator states: besides `I[S+ : S- | G] = 0` they impose
  `I[G_0 X_0 : S-_1 | G_1] = 0` and `I[S+_0 : X_0 G_1 | G_0] = 0`, the
  conditions under which the HMM built from the channel (applied at both ends
  of every bidirectional edge) generates the process. Lifted generators are a
  proper subclass, so the exact and Wyner values are upper bounds over all
  generators; the functional search is exact over deterministic ones.
  The reported value is the returned model's own (no functional fallback), the
  exact value upper-bounds the generative complexity, and models carry
  `markov_residual`. The functional model, previously returned even when it
  did not reproduce the process, now always does. The unconstrained values are
  `BidirectionalEpsilonMachine.wyner_common_information()` and
  `exact_common_information()`, and `generative_gap()` reports how much the
  lifted-generator requirement raises each (`G[S+ : S-] <= C_g <= H[G]`).
- `correction="bonferroni"` is the default for every CSSR learner (process,
  subtree, transducer, stack), and process CSSR counts its length-`(L+1)`
  resolution tests; pass `correction=None` for the old behavior.
- `viterbi` returns the `n + 1` states `X_0, ..., X_n`, aligned with `smooth`.
- `information_anatomy()["crypticity"]` is `C_mu - E` everywhere;
  `C± - E` is `"bidirectional_crypticity"` (also in
  `stored_information_decomposition`), and
  `BidirectionalEpsilonMachine.crypticity()` is now `bidirectional_crypticity()`.
- The Gács–Körner model is the common-information variable, not a generator of
  the process; its `entropy_rate()` is the model's own rate.
- `atoms()` includes the negative atom (atoms partition Σ*), and
  `left_quotients` / `residuals` include the empty residual.
- Brzozowski minimization returns a trimmed DFA, like Hopcroft and Moore.
- `TopologicalMarkovChain.parry_measure()` emits `(symbol, k)` labels when
  parallel edges share a symbol, so its entropy rate equals `h_top`.
- `butterfly_process()` is the five-state process of Mahoney, Ellison &
  Crutchfield (2009, Fig. 3); the previous construction was i.i.d.
- **Examples emit string symbols** (`"0"`, `"1"`, ...) everywhere; integer
  words now have probability 0 under the examples.
- **One golden mean**, in the Lind & Marcus form (forbids `11`, Example 1.2.3).
  The whole golden-mean family (`restricted_golden_mean`, `random_golden_mean`,
  `stretched_gm`, `rk_gm`, `rn_gm`, `nonunifilar_golden_mean`,
  `golden_mean_ghmm`) and `gm_to_even`'s input now use that convention;
  `nonunifilar_golden_mean(b)` equals `golden_mean(1 - b)`.
  `coupled_gmps` / `uncoupled_gmps` still pair the golden mean with its mirror.
- Removed duplicate examples:

  | Removed | Use |
  |---|---|
  | `golden_mean_forward(p)`, `golden_mean_reverse(p)`, `golden_mean_markov(p)`, `golden_mean_forbid_00(p)` | `golden_mean`, with `0` and `1` exchanged (and `1 - p` for `golden_mean_markov` / `golden_mean_forbid_00`) |
  | `golden_mean_shift_parry()` | `golden_mean(1 / phi)` (the Parry measure) |
  | `butterfly_two_branch()` | `butterfly_process()` |
  | `fair_coin()` | `bernoulli()` |
  | `ellison_fig9_forward()` | `irreversible_two_state()` |
  | `restricted_gm(k)` | `restricted_golden_mean(k)` |
  | `period7()` | `period8()` (they used the same word) |
  | `tetris_tgm()` | `tetris_history()` |

### Fixes

- The exact/Wyner generative-model channel was read off dit's optimum with
  the reverse-state axis permuted whenever the joint support listed a
  higher-indexed reverse state first (dit numbers symbols by first
  appearance). The channel then failed to screen `S+` from `S-`.

- **Shifts:** `SoficShift.topological_entropy` counts words on a
  right-resolving presentation (parallel same-label edges no longer inflate it);
  new `ShiftOfFiniteType.topological_entropy`; `trim_transient` actually prunes
  and `factor_language` drops words that cannot extend both ways; Dyck shift
  `reverse()` returns the mirror-image shift; `from_adjacency` assigns symbols in
  sorted order and honors subclasses; the synchronization methods on
  `SoficShift` (`markov_order`, `reset_threshold`, ...) no longer raise
  `AttributeError`; the Fischer cover raises on reducible shifts instead of
  returning a single terminal component; the Parry measure drops transient
  states.
- **Generators:** `cryptic_order` is 0 for zero-crypticity processes (even,
  periodic) via a new algorithm checked against brute force;
  `log_word_probability` no longer underflows; channel statistical complexity
  uses the driven process's occupation law (it was hash-seed dependent);
  mixed-state enumeration of infinite belief sets fails in under a second;
  numeric stationary distributions warn when not unique;
  `EpsilonMachine.from_hmm` handles more than 26 causal states;
  `word_probability` no longer returns 0 below 1e-15 (and
  `conditional_word_probability` no longer raises `ZeroDivisionError` there);
  `backward(normalize=True)` normalizes the final row; `reverse_is_finite` is
  documented as deciding finiteness of the reverse mixed states, a sufficient
  but not necessary condition for a finite reverse ε-machine (alternating
  biased coins has infinitely many reverse beliefs but a two-state reverse
  machine); `from_time_reversed` and `to_bidirectional` warn when numeric belief
  merging returns a finite truncation of an infinite reverse machine; the bidirectional machine's choice
  between joint classes that tie on the anatomy identity no longer depends on
  floating-point round-off (it chose a wrong class on x86-64, giving E = 1.0
  instead of 1.5 for a 3-state machine).
- **Inference:** Bayesian posterior machines start where the data start (they
  started in its final state, giving −∞ likelihoods and infinite BIC); WAIC
  scores from the stationary distribution; `learn_stack_hmm_mle` and the stack
  CSSR fitter use the exact maximum-likelihood weights over legal moves
  (Hunter 2004); spectral learning has a sampling noise floor for rank
  selection (`noise_scale`), noise-aware state merging (`belief_tolerance`), fast
  failure at `max_states`, and no negligible-mass states on exact input;
  `learn_pfa_alergia(..., censored=True)` for windowed samples;
  cross-validation never joins separate sequences.
- **Automata:** Büchi lasso acceptance sees accepting states visited mid-loop;
  Wheeler tie handling, initial states in `determinize_wheeler` and
  `wheeler_index`, word counting, empty marks, `minimize_wheeler` on dead
  states, and `colex_width`; modular VPA `minimize` on valid well-matched VPAs;
  intersection, difference and complement with empty-language operands;
  `complete()` no longer reuses an accepting trap left by a previous
  complement (double complement could return the wrong language);
  `right_quotient` by the empty word; `sofic.automata.languages.residuals`
  callable again; IDFA enumeration includes missing transitions before the first
  flag (`count_accessible_idfa(2, 2)` is 45), which changes some canonical
  topological ε-machine strings; transducer product and composition handle ε
  moves without double counting; faster VPA determinization, complement,
  emptiness and equivalence.
- **Examples:** `golden_mean_ghmm` uses its stationary initial vector (it did
  not reproduce the golden mean); power-automaton constructions iterate the
  alphabet in sorted order, so `synchronizing_word()` no longer depends on
  `PYTHONHASHSEED`.
- **Serialization and viz:** YAML stores observation, output and stack
  alphabets and supports sympy and `Fraction` values; `copy()` deep-copies
  attributes; Graphviz/TikZ node names are unique; TikZ escapes `^`, `~`, `\`
  for math mode and handles newlines in labels; YAML uses libyaml's C loader
  and dumper when available.
- `entropy_rate_hmm` and `state_distribution` no longer crash on presentations
  whose states are `MixedState` objects.
- The `riechers2017spectral2` citation pointed at an unrelated Phys. Rev. E
  paper; it is Chaos 28, 033116 (2018).
- `automaton_to_regex` was ambiguous for multi-character symbols (`ab` vs `a`,
  `b`); such symbols are now quoted (`'ab'`), and `ε`, `∅` and `'` are escaped.
- `HiddenMarkovModel.to_support_dfa` accepted only terminal recurrent subsets,
  so it rejected words of positive probability (e.g. `1` for the even process);
  every non-empty subset now accepts.
- `entropy_rate` and `state_entropy` no longer raise `TypeError` when state
  labels mix types (e.g. `0` and `(1, 0)` after a split).
- `is_equal_process` works on sympy-valued HMMs (object-dtype matrices broke
  `matrix_rank`).
- `DFA.add_transition` adds a missing source state, as `NFA.add_transition`
  does.
- `EpsilonMachine.from_hmm` split causal states that differed only by
  floating-point round-off from belief updates (a fair-coin HMM gave 33 states
  and C_mu ≈ 2.46); numeric transition probabilities are now compared within the
  mixed-state tolerance.

### New

- `FiniteStateChannel`: a stochastic `MealyMachine` (or ε-transducer) as a
  channel for `dit.coding` channel codes. `MealyMachine.__dit_channel__` returns
  it, so `hamming(3).probability_of_error(trapdoor())` works directly. The
  likelihood comes from the forward algorithm, the soft decoders get BCJR symbol
  likelihoods, and `capacity_gap` returns a `(lower, upper)` bracket from the
  no-feedback capacity bounds. Needs a dit with `dit.coding.as_channel`.
- `FiniteStateChannel.feedback_step` / `feedback_initial` step the channel one
  use at a time (the belief is the normalized forward vector) and
  `feedback_capacity` brackets the capacity with feedback, so dit's feedback
  codes (`erasure_arq`, `stop_and_wait_arq`, `horstein`) run over a channel
  with memory. Horstein's scheme on `trapdoor()` reaches rates above the
  no-feedback upper bound (`0.584`). Needs a dit with `dit.coding.horstein`.
- `sofic.generators.trapdoor_code.trapdoor_code(k)`: the zero-error feedback
  code of Permuter et al. for the trapdoor channel (actions relative to the
  known state, enumerative coding into words with no `11`, backward decoding
  from output differences), with rate approaching `log2(phi)`.
- `sofic.algebra`: algebraic automata theory for deterministic presentations.
  `TransformationSemigroup` (lazy closure, shortest words, sink completion of
  partial maps), Green's relations and maximal subgroups, `syntactic_monoid`
  and `is_star_free`, and the Krohn-Rhodes holonomy decomposition
  (`holonomy_decomposition`) with an explicit `CascadeProduct` whose
  `emulates()` checks the covering. `krohn_rhodes_complexity` returns
  `ComplexityBounds(lower, upper, exact)`, and `prime_decomposition` lists the
  simple-group factors. `holonomy_information` splits `C_mu` across the
  holonomy levels. DFAs, unifilar HMMs / ε-machines and sofic shifts gain
  `transition_semigroup()` and `krohn_rhodes_complexity()`, and the shift
  invariants gain `factor_syntactic_monoid` and
  `factor_krohn_rhodes_complexity`. Wheeler-presentation failures now report
  whether the language is star-free.
- `sofic.algebra`, following DeDeo's "Effective Theories for Circuits and
  Automata" (2011): `HolonomyDecomposition.effective_theory(k)` and
  `CascadeProduct.truncate(k)` give coarse-grained machines on overlapping
  superstates; `independent_random_maps` / `coupled_random_maps` /
  `random_maps` rewrite probabilistic automata as random choices of
  deterministic maps; `element_walk` and `classify_elements` follow the walk
  over semigroup elements; and `walk_information` splits lost information into
  irreversibility and dissipation. Also new: language varieties
  (`is_piecewise_testable`, `is_locally_testable`, `is_definite`, ...),
  `minimal_ideal` / `minimal_rank` / `is_synchronizing` /
  `schutzenberger_group`, belief-map semigroups (`projective_semigroup`,
  `mixed_state_semigroup`), and `is_full_transformation_monoid`, which makes
  the complexity of `T_n` exact without enumerating it.
- `sofic.generators.capacity`: capacity bounds for finite ε-transducers
  (unifilar finite-state channels). `EpsilonTransducer.feedback_capacity()`
  brackets the feedback capacity with Q-graph bounds (convex upper bound,
  certified graph-based-encoder lower bound), the Q-graph duality bound, or
  belief-state value iteration; `EpsilonTransducer.capacity()` brackets the
  capacity without feedback with an optimized Markov input and the Q-graph
  duality dynamic program. Results are `CapacityBounds(lower, upper, ...)`.
  The optional `capacity` extra installs cvxpy for the convex step.
- `trapdoor()` / `trapdoor_et()` and `ising_channel()` / `ising_channel_et()`
  example channels, each with a uniform initial state.
- `sofic.inference.Trials`: `learn_epsilon_machine_cssr`,
  `learn_epsilon_transducer_cssr`, and `suggest_max_history` learn from
  independent trials, pooling counts without any history spanning two trials
  (`SuffixCounts.from_sequences`; `JointSuffixCounts.from_sequences` accepts
  paired `Trials`). A `dit.inference.Trials` is accepted too.
- Hypothesis strategies in `sofic.testing` for NFAs, Büchi automata and lassos,
  Wheeler NFAs, Markov chains, Mealy HMMs, sofic shifts, SFTs, VPAs, NWAs, and
  Mealy transducers; `ci` and `nightly` Hypothesis profiles (`HYPOTHESIS_PROFILE`).
- Experimental `EpsilonMachine.reverse_epsilon_machine_is_finite()` (and
  `sofic.generators.reversal.reverse_epsilon_machine_is_finite`): decides whether
  the reverse ε-machine has finitely many *recurrent* causal states, which
  `reverse_is_finite` only bounds from one side.
- A property and metamorphic test suite (`tests/test_properties_*.py`) checking
  shifts, generators, inference, automata, serialization, viz and examples
  against brute-force oracles and invariants.
- `parse_regex`, `regex_to_nfa` and `NFA.from_regex` build an NFA from a regular
  expression by Thompson's construction (Thompson 1968); malformed input raises
  `sofic.exceptions.RegexSyntaxError`.
- `learn_dfa_k_testable`, the García–Vidal learner for strictly k-testable
  languages: the smallest such language containing a set of positive samples.
- `LabeledAutomaton.is_empty`, `accepted_word` (a shortest witness),
  `is_universal` and `includes` for NFAs and DFAs, by the antichain algorithms
  of De Wulf et al. (2006) without full determinization. `BuchiAutomaton`
  overrides them with ω-semantics: `is_empty` and the new `accepted_lasso`
  decide Büchi emptiness; universality and inclusion raise
  `NotImplementedError`.
- `sofic.generators.relative_entropy_rate`: `relative_entropy_rate(p, q)` in
  bits (exact for unifilar `q`, `inf` when P is not absolutely continuous with
  respect to Q on finite words), `relative_entropy_rate_bounds(p, q, n)` for
  general HMMs, and `HiddenMarkovModel.relative_entropy_rate`.
- `periodic_points(n)` and `zeta_function()` for topological Markov chains,
  shifts of finite type and sofic shifts (Manning's signed-subset formula for
  sofic shifts; Lind & Marcus §6.4); counts are exact integers, and
  `zeta_function` needs the `symbolic` extra.
- In- and out-state splitting and amalgamation of topological Markov chains with
  division and edge matrices (`A = DE`, `A' = ED`; Lind & Marcus §2.4);
  `bowen_franks_group` (with the sign of `det(I − A)`) and
  `jordan_form_away_from_zero` as conjugacy invariants (§7.4).
- `sofic.generators.correlations`: closed-form `autocorrelation`,
  `power_spectrum` (continuous part; delta peaks at unit-circle eigenvalues
  excluded) and `mutual_information_function` for any finite HMM (Riechers &
  Crutchfield 2018).
- `predictive_rate_distortion` and `PredictiveRateDistortionCurve`: the causal
  information bottleneck / predictive rate-distortion curve of a finite
  ε-machine, by an annealed, seeded Blahut–Arimoto iteration over exact future
  morphs (Still et al. 2010; Marzen & Crutchfield 2016; Tishby et al. 2000).
- `HiddenMarkovModel.entropy_rate(method="auto"|"exact"|"bounds"|"blackwell")`
  handles non-unifilar presentations: exact when the mixed states close,
  otherwise the converged Cover & Thomas (Thm 4.5.1) bounds, with a warning if
  they have not met `tol`; `auto` never returns a stochastic estimate. New
  `entropy_rate_bounds`, `entropy_rate_blackwell` (batch-means standard error)
  and `mixed_state_walk`.
- `statistical_complexity_dimension` and `ifs_lyapunov_dimension` (Jurgens &
  Crutchfield 2021): the Lyapunov dimension of the Blackwell measure from the
  mixed-state random walk; reproduces the paper's Cantor (log 2 / log 3) and
  Sierpinski (log₂ 3) examples.
- `EpsilonMachine.from_hmm(hmm, max_states=...)` caps the mixed-state
  enumeration (it silently ignored keyword arguments).
- `minimal_quasi_realization` and `process_rank` (plus
  `HiddenMarkovModel.process_rank`): Schützenberger/Fliess minimization of an
  HMM's linear representation to a minimal-dimension `QuasiRealization` whose
  dimension is the process rank; exact for sympy input.
- `bisimulation_partition` and `coarsest_lumping`: the coarsest strongly
  lumpable partition (Larsen–Skou probabilistic bisimulation) by splitter-based
  refinement; `lump(model)` and the `lump` methods accept `partition=None` to
  mean the coarsest partition.
- `SlidingBlockCode.apply_to_process` and `factor_codes.image_process`: the
  image process of a sliding block code; `HiddenMarkovModel.higher_block(k)` for
  k-block presentations. Excess entropy and statistical complexity are not
  conjugacy invariants (`E(β_k X) = E(X) + (k − 1) h_μ`); only `h_μ` is.
- `sofic.generators.state_splitting`: process-preserving out- and in-splitting
  (`split_state`) and `amalgamate`.
- `sofic.generators.support`: `support_nfa`, `support_includes`,
  `is_absolutely_continuous`, `support_equal` (stationary support;
  finite-cylinder absolute continuity), with matching `HiddenMarkovModel`
  methods.
- `condition_on_language`: conditions an HMM on a prefix-closed regular
  constraint by the Doob h-transform; the uniform i.i.d. process conditioned on
  an irreducible SFT is its Parry measure.
- `omega_probability` for deterministic Büchi properties of HMM output (bottom
  strongly connected components of the product chain; Baier & Katoen 2008) and
  `regular_language_probability` for exact `P(X_{0:n} ∈ L)`.
- `sofic.inference.learn_epsilon_machine_active` and `ProcessOracle`: an
  L*-style active learner for ε-machines from probability and equivalence
  queries; exact for finite, exactly synchronizable ε-machines, with `history`
  seeding for infinite transients.
- `renyi_entropy_rate`, `pressure` and `rate_function`
  (`sofic.generators.renyi`): Rényi entropy rates for α ∈ [0, ∞] (topological,
  Shannon and min-entropy rates as special cases) and the large-deviation rate
  function of −(1/n) log₂ P(X_{0:n}) by Legendre transform of the pressure.
- Experimental `canonical_residual_hmm` (`sofic.generators.canonical_residual`,
  requires `experimental=True`): a generator whose states are the extreme future
  morphs of a finite ε-machine, the process analogue of the canonical RFSA; it
  can be strictly smaller than the ε-machine.

## 0.3.0

A correctness review and breaking refactor. Old names are **not** kept as
aliases; use the tables below to migrate.

### Behavior changes

- **All information quantities are in bits**, including log-likelihoods
  (`log_likelihood`, the Baum-Welch trace, cross-validated log-likelihood,
  HDP-HMM traces), topological entropy (was nats), and collision entropy (was
  nats). AIC, AICc, BIC, and WAIC keep their standard deviance scale (computed
  from the natural log-likelihood); the MDL code length is in bits. `score` and
  `observed_information` remain derivatives of the natural log-likelihood, so
  standard errors are unchanged.
- **Covers follow Lind & Marcus**: *right* means right-resolving.
  `RightFischerCover` is now the exact minimal right-resolving presentation
  (it was built from follower sets truncated at length 8 and labeled *left*);
  reducible shifts raise `SoficValidationError`. `LeftFischerCover` is its
  mirror image.
- `MaximizedPrimeAtomaton` subclasses `NFA`, not `AtomicAutomaton`: it is the
  reverse of the canonical RFSA of the reversed language and need not be atomic.
- `CanonicalVisiblyPushdownAutomaton.from_vpa` raises
  `NonWellMatchedLanguageError` for languages with pending calls or returns.
- `DeterministicVisiblyPushdownAutomaton.from_vpa` determinizes
  nondeterministic input instead of raising.
- VPA boolean and structural operations return concrete automata.
- Wildcard VPA returns fire on every stack symbol and, with a bottom symbol, on
  the empty stack (the simulator's behavior; the docstring said otherwise).
- `ShiftOfFiniteType.from_forbidden_words` raises instead of silently
  truncating at `max_states`; an empty forbidden set builds the full shift.
- `equivalent()` compares automata over the union of both alphabets.
- `MarkovChain.words_of_length(0)` and `sample_path` respect the initial
  distribution; sampling from an initial law with no mass raises `ValueError`.
- `block_entropy_estimates(use_exact=True)` reports crypticity as `C_mu - E`.
- `SlidingBlockCode.apply` keeps constraints longer than the window and raises
  when the block map misses an allowed block.
- Baum-Welch raises when every sequence is impossible and warns when some are.
- `BuchiAutomaton.accepts_lasso` rejects an empty loop.
- Stack CSSR no longer drops histories during determinization (which produced
  zero-mass states and validation errors).
- TikZ labels: fixed uncompilable `\midcall` / `\uparrowA` and escaped
  `\times` / sympy LaTeX.
- cmpy example factories validate parameters like their curated counterparts
  (e.g. a coin bias of 1 raises). Processes previously built by `Even`, `Nemo`,
  and `ABC` now come from the curated functions and emit integer symbols `0`/`1`;
  `alternating_biased_coins(p, p)` keeps its two-state presentation.
- `joint_block_distribution(block_length=n)` counts blocks of `n` symbols
  (`history_length=h` meant `h + 1` symbols); the default `2` is unchanged.

### New

- Exact canonical RFSA (`CanonicalRFSA.from_language`), maximized prime
  átomaton, `CanonicalRFSA.dual` / `MaximizedPrimeAtomaton.dual`, exact
  `prime_residuals`, `atoms`, `prime_atoms`, and `ResidualTable`.
- NL\*: `learn_rfsa_nlstar`, `learn_prime_atomaton_nlstar`,
  `learn_rfsa_from_language`, and `AutomatonEquivalenceOracle`.
- VPA: `determinize`, `is_empty`, `accepted_word`, `is_universal`, `includes`,
  `equivalent`, `has_unmatched_word`; `sofic.automata.vpa.to_single_entry` /
  `to_multiple_entry`; modular `minimize` converts automatically when no
  modules are given. `NestedWordAutomaton` gains the same operations.
- Exact left/right Krieger covers.
- `sofic.generators.matrices` (joint matrices and start-vector policies) and
  `sofic.generators.sampling`.

### Module moves

| Old module | New module |
|---|---|
| `sofic.generators.hmm_inference` | `sofic.inference.hmm` (`filtering`, `em`, `information`); `sample` → `sofic.generators.sampling` |
| `sofic.generators.epsilon_inference` | `sofic.inference.cssr` (`process`, `subtree`, `counts`, `significance`); `spectral` → `sofic.inference.spectral` |
| `sofic.generators.epsilon_transducer_inference` | `sofic.inference.cssr.transducer` |
| `sofic.generators.stack_inference` | `sofic.inference.cssr.stack` |
| `sofic.automata.{active,rpni,edsm,dfasat,alergia,papni,observation}` | `sofic.automata.learning.*` |
| `sofic.automata.learning` (NL\*) | `sofic.automata.learning.nlstar` |
| `sofic.automata.vpa`, `vpa_simulation` | `sofic.automata.vpa` package (`base`, `operations`, `deterministic`, `modular`, `canonical`, `simulation`) |
| `sofic.automata.{icdfa,idfa,enumeration}` | `sofic.automata.enumeration.{icdfa,idfa,words}` |
| `sofic.automata.{canonical_extraction,rfsa,atomaton,canonical_dual}` | `sofic.automata.canonical.{residual,rfsa,atomaton,dual}` |
| `sofic.shifts.sofic_relation` | `sofic.shifts.product_alphabet_shift` |

### Renames and removals

| Old | New |
|---|---|
| `cssr` | `learn_epsilon_machine_cssr` |
| `subtree_merge` | `learn_epsilon_machine_subtree` |
| `spectral` (wrapper) | `learn_epsilon_machine_spectral` |
| `transcssr` | `learn_epsilon_transducer_cssr` |
| `stack_cssr` / `stack_subtree_merge` / `fit_stack_hmm_mle` | `learn_stack_hmm_cssr` / `learn_stack_hmm_subtree` / `learn_stack_hmm_mle` |
| `suggest_lmax` | `suggest_max_history` |
| `Lmax=`, `L=` (CSSR family and subtree learners) | `max_history=` |
| `reconstruction_sweep(lmaxes=)` | `reconstruction_sweep(max_histories=)` |
| `GoodnessOfFit.L`, `goodness_of_fit(L=)` | `block_length` |
| `forward(scaled=)`, `backward(scaled=)` | `normalize=` |
| `joint_block_distribution(history_length=h)` | `joint_block_distribution(block_length=h + 1)` |
| `QuasiStochasticModel.transition_matrices`, `quasi_inference.transition_matrices` | `symbol_matrices` |
| `sofic.generators.words.hmm_*`, `pfa_*`, `quasi_*`, `markov_*` functions | private; use the model methods |
| `cartesian_product_gg` / `cartesian_product_tt` | `generator_product` / `transducer_product` |
| `compose_tt` / `compose_tg` | `compose_transducers` / `compose_transducer_generator` |
| `wnfa_to_wdfa` / `minimum_wdfa` | `determinize_wheeler` / `minimize_wheeler` |
| `papni_encode` / `papni_encode_samples` | `encode_dyck_word` / `encode_dyck_samples` |
| ICDFA helpers `next_flags`, `string_from_flags`, `flags_from_string`, `count_flag_sequences` | `icdfa_next_flags`, `icdfa_string_from_flags`, `icdfa_flags_from_string`, `icdfa_count_flag_sequences` |
| IDFA helpers `string_from_flags`, `extended_flags`, `transition_count` | `idfa_string_from_flags`, `idfa_extended_flags`, `idfa_transition_count` |
| `CallDrivenAutomaton` | `ModularVisiblyPushdownAutomaton` |
| `CompositeVisiblyPushdownAutomaton`, `union_vpa`, `intersection_vpa`, `complement_vpa`, `difference_vpa`, `concat_vpa`, `kleene_star_vpa` | removed; use the VPA methods |
| `LabeledAutomaton.intersect` / `concatenate` / `star` | `intersection` / `concat` / `kleene_star` |
| `learn_maximized_prime_atomaton` | `learn_prime_atomaton_nlstar` (or `learn_rfsa_nlstar`) |
| `SoficRelation`, `to_sofic_relation` | `ProductAlphabetShift`, `to_product_alphabet_shift` |
| cover `from_sofic` | `from_presentation` |
| `{left,right}_{fischer,krieger}_from_sofic` | `{left,right}_{fischer,krieger}_cover` |
| `sofic.from_yaml`, `serialization.from_yaml` | `model_from_yaml` |
| `examples.processes`: `BiasedCoin`, `FairCoin`, `Even`, `Nemo`, `NRPS`, `ABC` | removed: `bernoulli`, `fair_coin`, `even_process`, `nemo_process`, `noisy_random_phase_slip`, `alternating_biased_coins(1 - p, 1 - q)` |
| `GoldenMean` / `Butterfly` / `PSB` | `golden_mean_forbid_00` / `butterfly_two_branch` / `phase_slip_backtrack_cmpy` |
| other PascalCase `examples.processes` factories (e.g. `BitFlip`, `Delay`, `RestrictedGM`, `IrreversibleTwoState`) | snake_case (`bit_flip`, `delay`, `restricted_gm`, `irreversible_two_state`) |
