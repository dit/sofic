.. spectral.rst
.. py:module:: sofic.inference.spectral

*****************
Spectral Learning
*****************

Spectral (method-of-moments) learning fits a weighted finite automaton (WFA) /
observable-operator model to a stationary, discrete-time, discrete-alphabet
process from sampled sequences. The estimator builds the empirical **Hankel
matrix** of block probabilities, takes its singular value decomposition, and
reads the observable operators off the truncated factorization
:cite:`Balle2014`. It is the automata-theoretic twin of the spectral hidden
Markov model algorithm of Hsu, Kakade & Zhang :cite:`Hsu2012`. Unlike
Baum-Welch (:func:`sofic.generators.hmm_inference.baum_welch`), spectral
learning is a consistent, one-shot estimator with no local optima, and the model
order is read from the singular-value spectrum instead of being fixed in
advance.

The learned model is returned as a
:class:`~sofic.generators.quasi_realization.QuasiRealization` -- sofic's native
matrix observable-operator model :cite:`Jaeger2000` -- whose ``word_probability``
implements the WFA recursion directly.

.. code-block:: python

   from sofic.examples import golden_mean
   from sofic.inference.spectral import learn_spectral_wfa, spectral_singular_values

   process = golden_mean(0.4)
   data, _states = process.sample(20000)

   spectral_singular_values(data, prefix_length=3)   # gap reveals the model order
   model = learn_spectral_wfa(data, prefix_length=3, rank=2)
   model.word_probability((0, 1, 0))                 # WFA recursion pi @ A_0 A_1 A_0 @ tau

Small alphabets
===============

The single-symbol spectral HMM of :cite:`Hsu2012` needs at least as many
symbols as hidden states (a full-rank observation matrix). The Hankel
formulation avoids this by indexing moments with multi-symbol *prefixes* and
*suffixes*: increasing ``prefix_length`` / ``suffix_length`` is the discrete
analogue of observation stacking and recovers processes -- golden-mean, even
process -- whose alphabet is smaller than the state count.

Projection to a generator
==========================

The observable-operator representation is *signed*, so it is always well defined
even when no non-negative (hidden Markov) realization of the same rank exists.
:func:`project_to_nmachine` renormalizes the operators into an explicit
:class:`~sofic.generators.nmachine.NMachine` (which may carry signed weights);
:func:`project_to_mealy` returns a stochastic
:class:`~sofic.generators.mealy.MealyHMM` when a non-negative realization exists
in the learned basis and raises :class:`SpectralInferenceError` otherwise.

.. code-block:: python

   from sofic.inference.spectral import project_to_nmachine

   machine = project_to_nmachine(model)   # observable-operator generator with a graph

API
===

.. autofunction:: learn_spectral_wfa

.. autofunction:: spectral_singular_values

.. autofunction:: hankel_matrices

.. autofunction:: project_to_nmachine

.. autofunction:: project_to_mealy

.. autoexception:: SpectralInferenceError
