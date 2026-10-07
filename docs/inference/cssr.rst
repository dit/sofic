.. cssr.rst
.. py:module:: sofic.inference.cssr

**************************
ε-Machine inference
**************************

Sample-based reconstruction of ε-machines from observed symbol sequences.
This complements the **oracle** path :func:`~sofic.generators.epsilon_construction.build_epsilon_machine`,
which merges probabilistically equivalent states in a *given* generator.

Three algorithms are implemented:

* **CSSR** (Causal-State Splitting Reconstruction) :cite:`Shalizi2002,Shalizi2004`
* **Subtree merging** (Crutchfield--Young batch reconstruction) :cite:`CrutchfieldYoung1989,Crutchfield1994`
* **Spectral** (Hankel SVD, then mixed-state extraction) :cite:`Balle2014,Hsu2012,Ellison2009`

Quick start
===========

.. code-block:: python

   import numpy as np
   from sofic.examples import even_process
   from sofic.generators.epsilon_machine import EpsilonMachine

   rng = np.random.default_rng(0)
   oracle = even_process(0.5)
   observations, _ = oracle.sample(5000, rng)

   inferred = EpsilonMachine.from_sequence(
       observations, method="cssr", max_history=4, alpha=0.001
   )
   len(list(inferred.states()))  # 2 for the even process

CSSR
====

CSSR :cite:`Shalizi2002` starts from an IID model and grows causal states in three phases:

1. **Initialize** — one state for the empty history.
2. **Homogenize** — extend each suffix one symbol into the past, up to ``max_history``; a
   child suffix whose next-symbol distribution differs significantly from its
   state's (G-test, :math:`\chi^2`, Monte Carlo exact G-test, or total-variation
   threshold) moves to the best
   matching state, or starts a new one. States keep suffixes of every length.
3. **Determinize** — drop transient states, then split states until each state and
   symbol lead to a single successor, then keep the most-visited recurrent class.

A length-``max_history`` suffix has no one-symbol extension in the suffix tree, so its
successor drops the oldest symbol. For a non-Markovian process that can forget the
phase: in the even process with ``max_history = 3``, the successor of ``011`` on ``1`` would
be the ambiguous ``111``. So the length-``max_history + 1`` suffix (here ``0111``) is tested
against the truncated suffix's state, and is sent to the best matching state when
the two differ.

Choose ``max_history`` at least the synchronization length of the source (its order, for
a Markov source). Much larger values run many more significance tests, and some
split states by chance; lowering ``alpha`` counters this. A process that is not
exactly synchronizable has no finite-``max_history`` reconstruction, and CSSR returns
extra states.

Choosing ``max_history`` and calibrating the tests
--------------------------------------------------

``max_history="auto"`` sets ``max_history`` to :func:`suggest_max_history`, the Markov order estimated
by :func:`dit.inference.select_markov_order`. Its default method tests order
:math:`n` against :math:`n + 1` with surrogates that preserve the observed
:math:`(n + 1)`-gram counts exactly, so the test holds its nominal size at any
sample length, where the asymptotic chi-squared test is badly anti-conservative
:cite:`Pethel2014`. For a Markov source this is its order, which is the
synchronization length CSSR needs. A strictly sofic source such as the even
process has infinite Markov order, so the suggestion grows with the sample: read
it as the longest history the data supports, not as a synchronization length.

The morph tests also rely on the chi-squared limit, which fails for the sparse
counts of long suffixes. ``test="exact"`` compares the G statistic with tables
drawn uniformly given the observed margins whenever an expected count is below 5
(seeded from the table, so reconstruction stays deterministic). The G-test
(``test="g"``) applies Yates' continuity correction when the table has one degree
of freedom (two observed symbols), matching
``scipy.stats.chi2_contingency(table, lambda_="log-likelihood")``; transCSSR and
stack CSSR share the same implementation.
``correction="bonferroni"`` (the default) divides ``alpha`` by the number of
suffixes eligible for testing, including the length-``max_history + 1`` suffixes
tested when resolving transitions, bounding the chance of any spurious split;
``correction=None`` tests each suffix at level ``alpha``. Because CSSR chooses each
test in light of earlier outcomes, false-discovery-rate step-up procedures do not
apply directly.

.. code-block:: python

   inferred = EpsilonMachine.from_sequence(
       observations, method="cssr", max_history="auto", test="exact", correction="bonferroni"
   )

.. autofunction:: learn_epsilon_machine_cssr

.. autofunction:: suggest_max_history

After reconstruction, check the result with
:func:`~sofic.inference.diagnostics.goodness_of_fit` and
:func:`~sofic.inference.diagnostics.structure_stability`
(see :doc:`diagnostics`).

.. autofunction:: morphs_differ

Subtree merging
===============

Subtree merging :cite:`CrutchfieldYoung1989` clusters histories with statistically
equivalent next-symbol distributions (metric tolerance ``delta``), then determinizes
to a unifilar presentation.  With ``delta=0``, two morphs are equivalent unless a
G-test at significance 0.01 tells them apart, a tolerance that scales with the
sample. Transitions follow the same successor rule as CSSR.

``learn_epsilon_machine_subtree`` accepts ``alpha``, ``test`` (including ``"exact"``) and
``correction`` (default ``"bonferroni"``, which divides ``alpha`` over the history
pairs compared; ``None`` disables it), as well as ``max_history="auto"``.

.. autofunction:: learn_epsilon_machine_subtree

Spectral reconstruction
=======================

Spectral reconstruction learns a weighted finite automaton from the Hankel matrix
of block probabilities :cite:`Balle2014,Hsu2012`, then extracts causal states as
mixed states of the learned operators :cite:`Ellison2009`. When the operators
are non-negative in the learned basis this is a Mealy projection followed by
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm`; signed
operators use mixed-state enumeration of the observable operators (not a
clustering heuristic). The same extraction is
:func:`~sofic.inference.spectral.project_to_epsilon_machine`.

Pass ``rank`` when the model order is known (two for the golden mean and even
process). Otherwise the Hankel singular-value gap selects the rank.

.. code-block:: python

   inferred = EpsilonMachine.from_sequence(
       observations, method="spectral", prefix_length=3, rank=2
   )

.. autofunction:: sofic.inference.spectral.learn_epsilon_machine_spectral

Unified entry point
===================

Use :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_sequence` to
dispatch to CSSR, subtree merging, or spectral reconstruction
(see :doc:`../generators/epsilon_machine`).

Related inference methods
=========================

**transCSSR** — input/output ε-transducers; see :doc:`transcssr`.

**Bayesian structural inference** — conjugate Dirichlet–multinomial evidence over
candidate unifilar topologies :cite:`Strelioff2014`; see :doc:`epsilon`.
This is not a Gibbs clustering heuristic over history labels.

Subtree merging is the literature reconstruction by morph clustering; a separate
agglomerative "causal-state merging" procedure is not provided. k-means on
history-morph embeddings (sometimes labelled "neural state discovery" despite
involving no neural network) is also not provided — mixed-state extraction is
the causal-state construction used after spectral learning.

**VLMC / context algorithm** — sparse Markov trees; not causally minimal in general.

**Context-tree weighting** — universal prediction; no causal-state semantics.

**Cross-validated HMM / EM** — fixed-architecture baseline :cite:`Shalizi2004`.

**REMAPF / decisional states** — utility-conditioned coarse-graining (Brodu).

**RKHS ε-machines** — continuous-time extension (arXiv:2011.14821).

See also :doc:`../generators/epsilon_machine`, :doc:`../generators/constructions`, :doc:`hmm`,
and :doc:`spectral`.
