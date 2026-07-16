.. hmm_inference.rst
.. py:module:: sofic.generators.hmm_inference

*************
HMM Inference
*************

Forward-backward, Viterbi decoding, and sampling for hidden Markov models
:cite:`BaumPetrie1966,Rabiner1989,Viterbi1967`, together with the
Cappé, Moulines & Rydén :cite:`Cappe2005` toolbox: fixed-interval smoothing,
Baum-Welch (EM) parameter re-estimation, and the score / observed information
via the Fisher and Louis :cite:`Louis1982` identities.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   In [3]: obs = [0, 1, 0, 0, 1]

   In [4]: from sofic.generators.hmm_inference import forward, viterbi, sample

   In [5]: alpha = forward(eps, obs)

   In [6]: path = viterbi(eps, obs)

   In [7]: seq = sample(eps, n=10)

Smoothing returns the posterior state marginals given the whole observation
sequence; the two-slice marginals give the posterior over consecutive state
pairs.

.. ipython::

   In [8]: from sofic.generators.hmm_inference import smooth, two_slice_marginals

   In [9]: gamma = smooth(eps, obs)      # gamma[t, s] = P(X_t = s | Y)

   In [10]: xi = two_slice_marginals(eps, obs)

Baum-Welch re-estimates the model parameters from data while holding the
transition-graph topology fixed:

.. ipython::

   In [11]: from sofic.generators.hmm_inference import baum_welch

   In [12]: data, _states = sample(golden_mean(0.3), n=2000)

   In [13]: fitted, loglik_trace = baum_welch(golden_mean(0.6), data)

The score and observed information quantify the log-likelihood gradient and
parameter uncertainty at the current parameters:

.. ipython::

   In [14]: from sofic.generators.hmm_inference import score, standard_errors

   In [15]: g = score(eps, obs)

   In [16]: se = standard_errors(eps, obs)

API
===

Filtering, decoding, and sampling
---------------------------------

.. autofunction:: forward
.. autofunction:: backward
.. autofunction:: log_likelihood
.. autofunction:: viterbi
.. autofunction:: sample

Smoothing
---------

.. autofunction:: smooth
.. autofunction:: two_slice_marginals

Parameter estimation
--------------------

.. autofunction:: baum_welch

Score and observed information
------------------------------

.. autofunction:: score
.. autofunction:: observed_information
.. autofunction:: free_parameter_labels
.. autofunction:: standard_errors
