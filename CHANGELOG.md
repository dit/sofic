# Changelog

## 0.3.0 (unreleased)

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
