.. topological_markov_chain.rst
.. py:module:: pensive.shifts.tmc

************************
Topological Markov Chain
************************

A :class:`TopologicalMarkovChain` is a shift of finite type presented by an
adjacency matrix. Its Parry measure is the maximum-entropy stochastic generator
:cite:`Parry1964,LindMarcus1995`.

.. ipython::

   In [1]: import numpy as np; from pensive.shifts import TopologicalMarkovChain; adj = np.array([[1, 1], [1, 0]], dtype=int); tmc = TopologicalMarkovChain.from_adjacency(adj, symbol_alphabet=frozenset({0, 1}))

   @doctest float
   In [2]: tmc.topological_entropy()
   Out[2]: 0.48121182505960347

API
===

.. autoclass:: TopologicalMarkovChain
   :members: from_adjacency, parry_measure, to_sofic_shift, topological_entropy

.. autofunction:: pensive.shifts.parry_construction.parry_measure
