.. hdp_hmm.rst
.. py:module:: sofic.inference.bayesian.hdp_hmm

*******************************
Bayesian nonparametric HMM
*******************************

The hierarchical Dirichlet process HMM (HDP-HMM) :cite:`Teh2006` and its
**sticky** variant :cite:`Fox2011` place a nonparametric prior over the number of
hidden states, so the state count is *inferred* from the data rather than fixed
in advance -- the canonical Bayesian answer to "how many states?". This module
implements the **weak-limit blocked Gibbs sampler** :cite:`Fox2011`: the
countably-infinite HDP prior is truncated to ``max_states`` components, and each
sweep

1. resamples the entire hidden-state sequence by forward-filter/backward-sample
   (FFBS) given the current parameters,
2. draws conjugate Dirichlet transition, initial, and **categorical
   (Dirichlet-multinomial)** emission rows given that state sequence, and
3. updates the shared top-level weights ``beta`` from Antoniak table counts, with
   the sticky self-transition override of :cite:`Fox2011`.

Emissions are discrete throughout; the Gaussian-emission variant is deliberately
excluded. Each retained draw is returned as a
:class:`~sofic.generators.moore.MooreHMM` restricted to the states occupied in
that sweep, together with a posterior over the number of occupied states.

.. code-block:: python

   from sofic.inference.bayesian import infer_hdp_hmm

   # e.g. sequences from a period-3 process 0,1,2,0,1,2,...
   posterior = infer_hdp_hmm(sequences, max_states=10, iterations=200, burn_in=100)

   posterior.state_count_posterior()   # {3: 0.62, 4: 0.30, ...}
   posterior.map_state_count()         # 3
   model = posterior.best_sample()     # highest-likelihood MooreHMM draw

Stickiness and over-segmentation
================================

The plain HDP-HMM (``kappa=0``) tends to *over-segment* processes with
overlapping, stochastic state emissions, because two states with nearly
identical emission laws are not penalized. The sticky mass ``kappa`` biases
toward self-transitions and suppresses this rapid state-switching
:cite:`Fox2011`; increase it for persistent regimes:

.. code-block:: python

   posterior = infer_hdp_hmm(sequences, kappa=30.0, max_states=12)

The sampler warm-starts the state labels from the observed symbols, which breaks
the emission-label symmetry that otherwise makes a randomly-initialized chain mix
poorly.

Relationship to the other estimators
====================================

The HDP-HMM infers the state count nonparametrically, whereas
:doc:`model_selection` scores a *fixed* set of candidate orders with information
criteria and :mod:`sofic.inference.bayesian` compares fixed unifilar topologies
by exact Dirichlet-multinomial evidence. For point-estimate reconstruction see
CSSR in :doc:`../generators/epsilon_inference`.

API
===

.. autofunction:: infer_hdp_hmm

.. autoclass:: HDPHMMPosterior
   :members: state_count_posterior, map_state_count, best_sample
