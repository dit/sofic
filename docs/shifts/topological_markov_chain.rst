.. topological_markov_chain.rst
.. py:module:: sofic.shifts.tmc

************************
Topological Markov Chain
************************

A :class:`TopologicalMarkovChain` is a shift of finite type presented by an
adjacency matrix. It is an *edge shift*: an entry ``k`` in the matrix is ``k``
distinct parallel edges, so :meth:`~TopologicalMarkovChain.topological_entropy`
is the ``log2`` spectral radius counting multiplicities (bits per symbol), even
when parallel edges share a label. Its Parry measure is the maximum-entropy
stochastic generator :cite:`Parry1964,LindMarcus1995`; when parallel edges share
a symbol the Parry HMM emits ``(symbol, k)`` for the ``k``-th copy so that its
entropy rate equals the topological entropy. Converting with
:meth:`~TopologicalMarkovChain.to_sofic_shift` forgets edge identities and gives
the entropy of the labeled (sofic) shift instead.

.. ipython::

   In [1]: import numpy as np; from sofic.shifts import TopologicalMarkovChain; adj = np.array([[1, 1], [1, 0]], dtype=int); tmc = TopologicalMarkovChain.from_adjacency(adj, symbol_alphabet=frozenset({0, 1}))

   @doctest float
   In [2]: tmc.topological_entropy()
   Out[2]: 0.6942419136306174

API
===

.. autoclass:: TopologicalMarkovChain
   :members: from_adjacency, parry_measure, to_sofic_shift, topological_entropy

.. autofunction:: sofic.shifts.parry_construction.parry_measure
