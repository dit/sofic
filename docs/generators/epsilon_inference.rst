.. epsilon_inference.rst
.. py:module:: sofic.generators.epsilon_inference

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
       observations, method="cssr", Lmax=4, alpha=0.001
   )
   len(list(inferred.states()))  # 2 for the even process

CSSR
====

CSSR :cite:`Shalizi2002` starts from an IID model and grows causal states in three phases:

1. **Initialize** — one state for the empty history.
2. **Homogenize** — extend suffixes; split or assign child histories when next-symbol
   distributions differ (G-test, :math:`\chi^2`, or total-variation threshold).
3. **Determinize** — split homogeneous states until transitions are unifilar; drop
   transient bottom-SCC states.

.. autofunction:: cssr

Subtree merging
===============

Subtree merging :cite:`CrutchfieldYoung1989` clusters histories with statistically
equivalent next-symbol distributions (metric tolerance ``delta``), then determinizes
to a unifilar presentation.  With ``delta=0``, morphs are compared up to a small
numerical tolerance for finite-sample estimates.

.. autofunction:: subtree_merge

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

.. autofunction:: spectral

Unified entry point
===================

Use :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_sequence` to
dispatch to CSSR, subtree merging, or spectral reconstruction
(see :doc:`epsilon_machine`).

Related inference methods
=========================

**transCSSR** — input/output ε-transducers; see :doc:`epsilon_transducer_inference`.

**Bayesian structural inference** — conjugate Dirichlet–multinomial evidence over
candidate unifilar topologies :cite:`Strelioff2014`; see :doc:`../inference/epsilon`.
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

See also :doc:`epsilon_machine`, :doc:`constructions`, :doc:`hmm_inference`,
and :doc:`../inference/spectral`.
