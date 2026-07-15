.. epsilon_inference.rst
.. py:module:: sofic.generators.epsilon_inference

**************************
ε-Machine inference
**************************

Sample-based reconstruction of ε-machines from observed symbol sequences.
This complements the **oracle** path :func:`~sofic.generators.epsilon_construction.build_epsilon_machine`,
which merges probabilistically equivalent states in a *given* generator.

Two algorithms are implemented:

* **CSSR** (Causal-State Splitting Reconstruction) :cite:`Shalizi2002,Shalizi2004`
* **Subtree merging** (Crutchfield--Young batch reconstruction) :cite:`CrutchfieldYoung1989,Crutchfield1994`

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

Unified entry point
===================

Use :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_sequence` to
dispatch to CSSR or subtree merging (see :doc:`epsilon_machine`).

Related inference methods (not yet implemented)
===============================================

**transCSSR** — input/output ε-transducers (Darmon, 2014).

**VLMC / context algorithm** — sparse Markov trees; not causally minimal in general.

**Context-tree weighting** — universal prediction; no causal-state semantics.

**Cross-validated HMM / EM** — fixed-architecture baseline :cite:`Shalizi2004`.

**REMAPF / decisional states** — utility-conditioned coarse-graining (Brodu).

**RKHS ε-machines** — continuous-time extension (arXiv:2011.14821).

See also :doc:`epsilon_machine`, :doc:`constructions`, and :doc:`hmm_inference`.
