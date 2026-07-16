.. model_selection.rst
.. py:module:: sofic.inference.model_selection

***************
Model Selection
***************

Classical model-selection criteria for choosing the order (state count) of a
fitted generator when a fully-Bayesian evidence is unavailable or undesirable.
The module scores any fitted :class:`~sofic.generators.base.HiddenMarkovModel`
using the natural-log likelihood from
:func:`sofic.generators.hmm_inference.log_likelihood` and a free-parameter count
read off the transition graph:

* **AIC** :cite:`Akaike1974` and the small-sample-corrected **AICc**
  :cite:`HurvichTsai1989`,
* **BIC** :cite:`Schwarz1978`,
* the two-part **minimum description length** :cite:`Rissanen1978`,
* **cross-validated** held-out log-likelihood, and
* **WAIC** :cite:`Watanabe2010` from posterior parameter samples.

Information criteria follow the convention *lower is better*.

.. code-block:: python

   from sofic.examples import golden_mean
   from sofic.inference.model_selection import score_model, information_criterion

   process = golden_mean(0.3)
   data, _states = process.sample(3000)

   score_model(process, data)                          # ModelScores(aic=..., bic=..., mdl=...)
   information_criterion(process, data, criterion="bic")

Cross-validation and WAIC
=========================

.. code-block:: python

   from sofic.inference.model_selection import cross_validated_log_likelihood, waic_epsilon_machine
   from sofic.inference.bayesian import EpsilonMachinePosterior

   def fit(train):
       model, _trace = golden_mean(0.6).baum_welch(train)
       return model

   cross_validated_log_likelihood(fit, data, folds=5)   # held-out log score (higher is better)

   posterior = EpsilonMachinePosterior(golden_mean(0.3), data)
   waic_epsilon_machine(posterior, [data], n_samples=200)

Ranking candidate topologies
============================

:func:`rank_topological_epsilon_machines` enumerates canonical topological
ε-machines (:func:`~sofic.generators.topological_epsilon_enumeration.iter_topological_epsilon_machines`),
fits each to data, and ranks them by an information criterion -- the frequentist
counterpart of the Bayesian
:class:`~sofic.inference.bayesian.comparison.ModelComparisonEM`, which also
exposes :meth:`~sofic.inference.bayesian.comparison.ModelComparisonEM.information_criteria`
and :meth:`~sofic.inference.bayesian.comparison.ModelComparisonEM.best_by_information_criterion`.

.. code-block:: python

   from sofic.inference.model_selection import rank_topological_epsilon_machines

   ranked = rank_topological_epsilon_machines(data, alphabet=[0, 1], num_states=[1, 2, 3], criterion="bic")
   best = ranked[0].machine

API
===

.. autoclass:: ModelScores
   :members: value

.. autoclass:: WAICResult

.. autofunction:: count_free_parameters

.. autofunction:: score_model

.. autofunction:: information_criterion

.. autofunction:: compare_information_criteria

.. autofunction:: cross_validated_log_likelihood

.. autofunction:: waic

.. autofunction:: posterior_pointwise_log_likelihoods

.. autofunction:: waic_epsilon_machine

.. autofunction:: rank_topological_epsilon_machines
