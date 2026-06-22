.. hmm_inference.rst
.. py:module:: pensive.generators.hmm_inference

*************
HMM Inference
*************

Forward-backward, Viterbi decoding, and sampling for hidden Markov models
:cite:`BaumPetrie1966,Rabiner1989,Viterbi1967`.

.. ipython::

   In [1]: from pensive.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   In [3]: obs = [0, 1, 0, 0, 1]

   In [4]: from pensive.generators.hmm_inference import forward, viterbi, sample

   In [5]: alpha = forward(eps, obs)

   In [6]: path = viterbi(eps, obs)

   In [7]: seq = sample(eps, n=10)

API
===

.. autofunction:: forward
.. autofunction:: backward
.. autofunction:: log_likelihood
.. autofunction:: viterbi
.. autofunction:: sample
