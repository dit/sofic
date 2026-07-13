.. inference.rst

*********
Inference
*********

The :mod:`pensive.inference` package infers stochastic generators from data.
Its Bayesian core (:mod:`pensive.inference.bayesian`) implements exact
conjugate **Dirichlet–multinomial** structural inference for Markov chains,
ε-machines, and stack HMMs, following the Bayesian structural inference
programme of Strelioff & Crutchfield :cite:`Strelioff2014`.

For a fixed topology, the posterior over transition probabilities is a product
of Dirichlet distributions, one per transition row, so the marginal likelihood
(model evidence) is available in closed form. Model comparison then ranks a set
of candidate topologies — Markov orders, unifilar ε-machines, or stack
topologies — by their posterior probabilities, with no sampling required.
Optional `PyMC <https://www.pymc.io/>`_ backends (``pip install pensive[bayes]``)
expose the same models for full posterior sampling.

The historical names ``InferMC`` and ``InferEM`` are retained as aliases for
:class:`~pensive.inference.bayesian.markov.MarkovChainPosterior` and
:class:`~pensive.inference.bayesian.epsilon.EpsilonMachinePosterior`.

.. note::

   For *non-Bayesian* reconstruction — Causal-State Splitting Reconstruction
   (CSSR) and subtree merging — see the point-estimate routines in
   :doc:`../generators/epsilon_inference`,
   :doc:`../generators/hmm_inference`, and
   :doc:`../generators/stack_inference`.

.. toctree::
   :maxdepth: 1

   markov
   epsilon
   stack_hmm
   pymc
