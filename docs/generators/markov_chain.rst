.. markov_chain.rst
.. py:module:: pensive.generators.markov

***********
MarkovChain
***********

A :class:`MarkovChain` is a visible-state Markov process: emissions coincide
with states. Finite-state Markov-chain terminology follows standard treatments
:cite:`KemenySnell1976`.

.. ipython::

   In [1]: from pensive.examples import golden_mean_markov

   In [2]: chain = golden_mean_markov(0.5)

   In [3]: chain.validate()

API
===

.. autoclass:: MarkovChain
