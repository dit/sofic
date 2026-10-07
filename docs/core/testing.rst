.. testing.rst
.. py:module:: sofic.testing

*******************
Testing strategies
*******************

:mod:`sofic.testing` provides `Hypothesis <https://hypothesis.readthedocs.io>`_
strategies that draw small, valid models for property-based tests. Hypothesis
is a test-only dependency (``pip install sofic[test]``); importing the module
without it raises :exc:`ImportError` only when a strategy is built.

.. code-block:: python

   from hypothesis import given

   from sofic.testing import mealy_hmms, nfas

   @given(nfas(max_states=3))
   def test_determinization_preserves_language(nfa):
       ...

   @given(mealy_hmms(max_states=3))
   def test_word_probabilities_sum_to_one(hmm):
       assert abs(sum(hmm.words_of_length(3).values()) - 1) < 1e-9

All strategies take keyword arguments bounding their size, so drawn models stay
small enough for brute-force reference checks.

API
===

.. autofunction:: dfas
.. autofunction:: nfas
.. autofunction:: buchi_automata
.. autofunction:: lassos
.. autofunction:: wheeler_nfas
.. autofunction:: epsilon_machines
.. autofunction:: markov_chains
.. autofunction:: mealy_hmms
.. autofunction:: sofic_shifts
.. autofunction:: sfts
.. autofunction:: vpas
.. autofunction:: nwas
.. autofunction:: mealy_transducers
