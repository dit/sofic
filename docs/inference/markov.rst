.. markov.rst
.. py:module:: sofic.inference.bayesian.markov

**********************
Markov-Chain Inference
**********************

Bayesian inference of finite-order Markov chains from a sequence
:cite:`Strelioff2014`. A :class:`MarkovChainPosterior` (aliased ``InferMC``)
places a :class:`DirichletPriorMC` over the transition rows of an order-``k``
chain, accumulates counts from the data, and exposes the conjugate posterior in
closed form. It yields point estimates (posterior-mean or maximum-likelihood
transition probabilities), the model evidence, and a
:class:`~sofic.generators.mealy.MealyHMM` realization of the fitted chain.

.. code-block:: python

   from sofic.inference.bayesian import MarkovChainPosterior

   post = MarkovChainPosterior(alphabet=("0", "1"), data=data, order=2)
   post.log_evidence()                 # marginal likelihood of the order-2 model
   hmm = post.generate_mealy_hmm(method="PME")   # posterior-mean chain as a MealyHMM

Model comparison
================

:class:`ModelComparisonMC` ranks a contiguous range of Markov orders by
posterior probability; :class:`ModelComparisonMC2` compares an explicit list of
orders.

.. code-block:: python

   from sofic.inference.bayesian import ModelComparisonMC

   cmp = ModelComparisonMC(alphabet=("0", "1"), data=data, min_order=0, max_order=4)
   cmp.model_probabilities()   # {order: P(order | data)}
   best = cmp.most_probable_model()

API
===

.. autoclass:: MarkovChainPosterior
   :members: add_counts_from, contexts, transition_probability_mle, transition_probability_pme, posterior_alpha_matrix, posterior_mean_matrix, log_evidence, generate_mealy_hmm, sample_mealy_hmms, as_pymc_model

.. autoclass:: DirichletPriorMC
   :members: create_random_prior, set_alpha, get_alpha

.. autoclass:: sofic.inference.bayesian.comparison.ModelComparisonMC
   :members: log_evidence, model_probabilities, most_probable_model

.. autoclass:: sofic.inference.bayesian.comparison.ModelComparisonMC2

.. autoclass:: sofic.inference.bayesian.counts.WordCountsMC
   :members: add_counts_from, get_word_count, set_word_count, clear_word_counts
