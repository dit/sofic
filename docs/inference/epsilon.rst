.. epsilon.rst
.. py:module:: pensive.inference.bayesian.epsilon

********************
ε-Machine Inference
********************

Bayesian inference over a *fixed unifilar topology* and its unknown start state
:cite:`Strelioff2014`. A :class:`DirichletDistributionEM` holds a
product-of-Dirichlets prior/posterior over the edge probabilities of a candidate
:class:`~pensive.generators.mealy.MealyHMM`; an
:class:`EpsilonMachinePosterior` (aliased ``InferEM``) marginalizes over the
unknown start state as well, giving the model evidence and posterior-mean
machine.

.. code-block:: python

   from pensive.examples import golden_mean
   from pensive.inference.bayesian import EpsilonMachinePosterior

   topology = golden_mean(0.5)                       # a unifilar candidate
   post = EpsilonMachinePosterior(topology, data=data)
   post.log_evidence()
   mean_machine = post.posterior_mean_machine()

Model comparison and diversity
==============================

:class:`ModelComparisonEM` ranks a set of candidate unifilar machines by
posterior probability and can sample the posterior predictive process. The
posterior **process diversity** quantifies how spread out the posterior is over
distinct processes (as opposed to over parameterizations); see
:class:`~pensive.inference.bayesian.diversity.PosteriorDiversityResult`.

.. code-block:: python

   from pensive.inference.bayesian import ModelComparisonEM

   cmp = ModelComparisonEM(machines=[m0, m1, m2], data=data)
   cmp.model_probabilities()
   cmp.process_diversity(method="posterior_mean")

API
===

Posterior over a fixed topology
-------------------------------

.. autoclass:: EpsilonMachinePosterior
   :members: log_evidence, start_node_probabilities, probability_start_node, sample_start_node, generate_sample, posterior_mean_machine, as_pymc_model

.. autoclass:: DirichletDistributionEM
   :members: set_edge_alpha, get_edge_alpha, get_node_alpha, log_evidence_start_node, mean_edge_probability, posterior_mean_machine, generate_sample

.. autoclass:: pensive.inference.bayesian.counts.PathCountEM
   :members: get_edges, get_nodes, get_edge_count, get_node_count, get_possible_start_nodes

Model comparison
----------------

.. autoclass:: pensive.inference.bayesian.comparison.ModelComparisonEM
   :members: log_evidence, model_probabilities, generate_sample, machine_diversity, process_diversity

Process diversity
-----------------

.. autoclass:: pensive.inference.bayesian.diversity.PosteriorDiversityResult

.. autofunction:: pensive.inference.bayesian.diversity.posterior_process_diversity
.. autofunction:: pensive.inference.bayesian.diversity.machine_diversity
.. autofunction:: pensive.inference.bayesian.diversity.process_identification_word_length
.. autofunction:: pensive.inference.bayesian.diversity.posterior_mean_word_distribution
.. autofunction:: pensive.inference.bayesian.diversity.word_distribution_to_pmf
