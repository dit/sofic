.. channel_capacity.rst
.. py:module:: sofic.generators.capacity

****************
Channel Capacity
****************

A finite :doc:`ε-transducer <epsilon_transducer>` is a *unifilar finite-state
channel*: the pair ``(x, y)`` observed in causal state ``s`` determines the next
causal state. :mod:`sofic.generators.capacity` brackets the capacity of such a
channel, with feedback (the encoder sees past outputs) and without.

Every method returns a :class:`CapacityBounds` ``lower <= C <= upper`` in bits
per channel use. Exact answers are not available in general: the feedback
capacity of finite-state channels is not Turing-computable
:cite:`Grigorescu2024`, and the capacity without feedback is open even for
the trapdoor channel. ``CapacityBounds.value`` returns the midpoint once the
bracket is narrower than ``tol``.

The methods assume a strongly connected channel (checked), so the capacity
does not depend on the initial state. Inputs undefined at a state act as input
constraints for the feedback bounds; the no-feedback bounds require every
input to be admissible everywhere.

Feedback capacity
=================

With feedback, the encoder knows the causal state, and the feedback capacity
is the limit of normalized directed information :cite:`Permuter2008,Yang2005`.

Q-graphs
--------

A :class:`QGraph` maps output histories onto a finite set of contexts by
walking a deterministic graph with one edge per output symbol
:cite:`Sabag2017`. For any Q-graph,

.. math::

   C_{\mathrm{fb}} \le \max_{P_{X \mid S, Q}} I(X, S; Y \mid Q),

with ``(S, Q)`` stationary :cite:`Sabag2017`. This is a convex program in the
joint ``P(s, q, x)`` :cite:`Sabag2020`, solved with cvxpy when the optional
``capacity`` extra is installed (``pip install sofic[capacity]``) and with
scipy otherwise. On the lower side, any graph-based encoder
:math:`P(x \mid s, q)` is a valid feedback input; its directed-information rate
:math:`\bar h(Y) - \mathbb{E}_\pi H(Y \mid X, S)` is bracketed with block
entropies :cite:`Cover2006` and optimized. When the encoder is also
BCJR-invariant the single-letter rate :math:`I(X, S; Y \mid Q)` is achievable
:cite:`Sabag2017,Sabag2020`. :func:`qgraph_bounds` tries Markov Q-graphs of the
last ``k`` outputs plus any graphs you pass.

Markov Q-graphs are not always enough. The trapdoor channel
:cite:`Permuter2008` is closed by a three-node graph:

.. ipython::

   In [1]: import numpy as np

   In [2]: from sofic.examples.processes import trapdoor_et

   In [3]: from sofic.generators.capacity import QGraph, qgraph_bounds

   In [4]: trapdoor = trapdoor_et()

   In [5]: graph = QGraph(("0", "1"), np.array([[0, 1], [0, 2], [1, 2]]), ("a", "b", "c"))

   In [6]: bounds = qgraph_bounds(trapdoor, qgraphs=[graph], max_order=1)

   @doctest
   In [7]: round(bounds.upper, 6), round(np.log2((1 + np.sqrt(5)) / 2), 6)
   Out[7]: (0.694242, 0.694242)

   @doctest
   In [8]: bounds.gap < 1e-4
   Out[8]: True

For the Ising channel :cite:`Elishco2014` (``ising_channel_et``), the four-node
graph with table ``[[1, 2], [0, 2], [0, 3], [0, 2]]`` closes the bracket at
:math:`C_{\mathrm{fb}} \approx 0.575522`.

Duality bound
-------------

:func:`duality_upper_bound` fixes a test distribution :math:`R(y \mid q)` on a
Q-graph. For a unifilar channel the resulting bound is the optimal average
reward of a *finite* Markov decision process with state ``(s, q)``, action
``x`` and reward :math:`D(P_{Y \mid x, s} \| R_{Y \mid q})`
:cite:`Huleihel2025`. Every test distribution gives a valid bound; the
starting one is the output law of the Q-graph upper-bound optimizer, refined
by Nelder--Mead.

Value iteration
---------------

:func:`value_iteration_estimate` solves the belief-state dynamic program of
:cite:`Permuter2008,Yang2005` (state :math:`P(S \mid y^t)`, action
:math:`P(x \mid s)`, reward :math:`I(X, S; Y)`) on a gridded simplex. Its
bracket bounds the gain of the discretized problem only, so it is returned
with ``certified=False``; ``details["policy"]`` holds the greedy input law at
every grid belief.

:func:`feedback_capacity` dispatches on ``method`` (``"qgraph"``,
``"duality"``, ``"value_iteration"``, or ``"all"``).

Capacity without feedback
=========================

* :func:`markov_input_lower_bound` optimizes an order-``k`` Markov input and
  reports a block-entropy lower bound on its information rate. That rate is
  the objective of the generalized Blahut--Arimoto algorithm
  :cite:`Vontobel2008,Arnold2006`, maximized here directly.
* :func:`duality_upper_bound_no_feedback` evaluates the dual bound with a
  Q-graph test distribution, a deterministic dynamic program over
  :math:`P(S, Q \mid x^t)` :cite:`Huleihel2021`. Its value function is convex,
  so interpolating on a simplex grid over-estimates it and the gridded bound
  stays valid at any resolution.

For the trapdoor channel, an order-3 Markov input certifies ``C >= 0.5727``
and the order-1 Markov Q-graph at ``belief_resolution=24`` gives
``C <= 0.5936``. The best published bounds are ``0.572`` and ``0.584``
:cite:`Huleihel2021`. :func:`capacity` dispatches on ``method``
(``"markov_input"``, ``"duality"``, or ``"all"``).

API
===

.. autoclass:: CapacityBounds
   :members: gap, value

.. autoclass:: QGraph
   :members: markov, from_dfa, aligned

.. autofunction:: feedback_capacity
.. autofunction:: qgraph_bounds
.. autofunction:: qgraph_upper_bound
.. autofunction:: qgraph_lower_bound
.. autofunction:: encoder_rate_bounds
.. autofunction:: duality_upper_bound
.. autofunction:: value_iteration_estimate
.. autofunction:: capacity
.. autofunction:: markov_input_lower_bound
.. autofunction:: duality_upper_bound_no_feedback
.. autofunction:: channel_tensors
