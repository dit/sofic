.. inference.rst

*********
Inference
*********

The :mod:`sofic.inference` package infers stochastic generators from data.
Its Bayesian core (:mod:`sofic.inference.bayesian`) implements exact
conjugate **Dirichlet–multinomial** structural inference for Markov chains,
ε-machines, and stack HMMs, following the Bayesian structural inference
programme of Strelioff & Crutchfield :cite:`Strelioff2014`. Alongside it,
:mod:`sofic.inference.spectral` provides consistent, local-optimum-free
**spectral (method-of-moments)** learning of observable-operator models from the
Hankel matrix of block statistics :cite:`Balle2014,Hsu2012`, and
:func:`sofic.inference.bayesian.infer_hdp_hmm` infers the *number* of hidden
states nonparametrically with a weak-limit (sticky) **HDP-HMM** sampler
:cite:`Teh2006,Fox2011`.

For a fixed topology, the posterior over transition probabilities is a product
of Dirichlet distributions, one per transition row, so the marginal likelihood
(model evidence) is available in closed form. Model comparison then ranks a set
of candidate topologies — Markov orders, unifilar ε-machines, or stack
topologies — by their posterior probabilities, with no sampling required.
Optional `PyMC <https://www.pymc.io/>`_ backends (``pip install sofic[bayes]``)
expose the same models for full posterior sampling.

The historical names ``InferMC`` and ``InferEM`` are retained as aliases for
:class:`~sofic.inference.bayesian.markov.MarkovChainPosterior` and
:class:`~sofic.inference.bayesian.epsilon.EpsilonMachinePosterior`.

.. note::

   For *non-Bayesian* reconstruction — Causal-State Splitting Reconstruction
   (CSSR), subtree merging, and spectral mixed-state extraction — see the
   point-estimate routines in
   :doc:`../generators/epsilon_inference`,
   :doc:`../generators/hmm_inference`, and
   :doc:`../generators/stack_inference`.

.. toctree::
   :maxdepth: 1

   markov
   epsilon
   spectral
   model_selection
   hdp_hmm
   stack_hmm
   pymc
