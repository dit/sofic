.. transition_graph.rst
.. py:module:: sofic.core
   :no-index:

****************
TransitionGraph
****************

:class:`TransitionGraph` wraps a :class:`networkx.MultiDiGraph` with a thin API
for adding states, adding transitions, and iterating edges. All ``sofic``
models store their structure in a ``TransitionGraph``.

.. ipython::

   In [1]: from sofic.core import TransitionGraph, ATTR_EMISSION, ATTR_PROB

   In [2]: g = TransitionGraph()

   In [3]: g.add_state("A")

   In [4]: g.add_state("B")

   In [5]: g.add_transition("A", "B", **{ATTR_EMISSION: 0, ATTR_PROB: 1.0})

   @doctest
   In [6]: len(list(g.transitions()))
   Out[6]: 1

Attribute constants
===================

.. autodata:: EPSILON
.. autodata:: ATTR_EMISSION
.. autodata:: ATTR_PROB
.. autodata:: ATTR_SYMBOL
.. autodata:: ATTR_QUASIPROB
.. autodata:: ATTR_OUTPUT
.. autodata:: KIND_CALL
.. autodata:: KIND_RETURN
.. autodata:: KIND_INTERNAL

API
===

.. autoclass:: Transition
.. autoclass:: TransitionGraph
   :members: add_state, add_transition, states, transitions, copy, reverse, nx
