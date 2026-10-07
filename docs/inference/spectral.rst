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
Baum-Welch (:func:`sofic.inference.hmm.baum_welch`), spectral
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
   model.word_probability(("0", "1", "0"))           # WFA recursion pi @ A_0 A_1 A_0 @ tau

Without an explicit ``rank``, the order is the number of Hankel singular values
above ``singular_value_threshold`` times the largest and, for sampled data,
above the sampling noise floor ``noise_scale * sqrt((prefix_length + 1)
(suffix_length + 1) / n)`` for ``n`` observed symbols. Each Hankel entry is an
empirical block frequency with variance about ``f(uv) / n``, and the ``f(uv)``
of one prefix and suffix length sum to one, so this is the typical size of the
noise matrix; a relative cutoff alone keeps noise singular values (about
``0.003`` for the golden mean at ``n = 50000``). Exact ``word_probability``
input has no noise floor.

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

Projection to an ε-machine
==========================

:func:`project_to_epsilon_machine` extracts the causal presentation: a
non-negative Mealy projection when one exists in the learned basis, otherwise
mixed-state enumeration of the observable operators
:cite:`Ellison2009`. The same path is
:func:`~sofic.inference.spectral.learn_epsilon_machine_spectral` /
``EpsilonMachine.from_sequence(..., method="spectral")``.

Learned from exact statistics, mixed states are rounded to six decimals and
emissions of probability below ``1e-6`` are dropped as rounding noise. Learned
from samples, the operators are only accurate to the sampling error, so mixed
states are identified when their predicted probabilities of all words up to
``suffix_length`` agree within ``5 / sqrt(n)`` and emissions below ``3 / sqrt(n)``
are dropped (both scaled by ``noise_scale``, and the merge tolerance can be set
with ``belief_tolerance``). Enumeration stops with
:class:`SpectralInferenceError` after ``max_states`` mixed states.

.. code-block:: python

   from sofic.inference.spectral import learn_epsilon_machine_spectral
   from sofic.examples import golden_mean

   process = golden_mean(0.5)
   eps = learn_epsilon_machine_spectral(word_probability=process.word_probability, alphabet=("0", "1"), prefix_length=3, rank=2)
   len(list(eps.states()))  # 2

API
===

.. autofunction:: learn_spectral_wfa

.. autofunction:: spectral_singular_values

.. autofunction:: hankel_matrices

.. autofunction:: project_to_nmachine

.. autofunction:: project_to_mealy

.. autofunction:: project_to_epsilon_machine

.. autoexception:: SpectralInferenceError
