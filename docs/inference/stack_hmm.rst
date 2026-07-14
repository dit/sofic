.. stack_hmm.rst
.. py:module:: pensive.inference.bayesian.stack_hmm

********************
Stack-HMM Inference
********************

Bayesian inference for :class:`~pensive.generators.stack_hmm.HiddenMarkovStackModel`
topologies. As with the finite-state case, the posterior over the transition
probabilities *enabled by each stack configuration* factorizes into Dirichlet
rows, so evidence and posterior-mean models are closed-form
:cite:`Strelioff2014,BealBlockeletDima2015`. Because the configuration space is
generally infinite, counting uses a bounded ``max_stack_depth``.

.. code-block:: python

   from pensive.inference.bayesian import StackHMMPosterior

   post = StackHMMPosterior(topology, data=data, max_stack_depth=8)
   post.log_evidence()
   mean_model = post.posterior_mean_model()

Model comparison
================

:class:`ModelComparisonStackHMM` ranks enumerated stack topologies (for example
those produced by :func:`~pensive.shifts.dyck_enumeration.iter_sofic_dyck_topologies`)
by conjugate marginal likelihood.

API
===

.. autoclass:: StackHMMPosterior
   :members: log_evidence, posterior_mean_model

.. autoclass:: DirichletDistributionStackHMM
   :members: log_evidence, posterior_mean_probabilities, posterior_mean_model

.. autoclass:: PathCountStackHMM
   :members: get_count, get_config_count

.. autoclass:: ModelComparisonStackHMM
   :members: log_evidences, model_probabilities, most_probable_model
