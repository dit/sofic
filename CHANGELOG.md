# Changelog

## Unreleased (0.4.0)

A second correctness review: every fix below was reproduced against a
brute-force reference or a closed-form value and has a regression test.

### Breaking changes

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

### New

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
