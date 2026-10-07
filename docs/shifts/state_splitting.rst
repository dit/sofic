.. state_splitting.rst
.. py:module:: sofic.shifts.state_splitting

****************************************
State Splitting and Conjugacy Invariants
****************************************

Out- and in-splitting a graph replaces a state by copies that share its
out-edges (respectively in-edges) according to a partition, giving an edge shift
conjugate to the original :cite:`LindMarcus1995` (§2.4). Amalgamation is the
inverse operation. By the Decomposition Theorem every conjugacy of edge shifts
is a composition of splitting and amalgamation codes (§7.1).

Edges of a :class:`~sofic.shifts.tmc.TopologicalMarkovChain` carry
multiplicities, so a single edge is named ``(source, target, key, copy)``;
:func:`out_edges` and :func:`in_edges` list them, and a partition is a mapping
``{state: [part, ...]}`` of those edges. The division and edge matrices ``D``
and ``E`` of a splitting satisfy ``A = DE`` and ``A' = ED`` (Theorem 2.4.12).

.. ipython::

   In [1]: import numpy as np; from sofic.shifts import TopologicalMarkovChain, out_edges, out_split, out_split_matrices

   In [2]: full = TopologicalMarkovChain.from_adjacency(np.array([[2]]))

   In [3]: (state,) = full.states(); partition = {state: [[edge] for edge in out_edges(full, state)]}

   In [4]: d, e = out_split_matrices(full, partition)

   @doctest
   In [5]: (e @ d).tolist()
   Out[5]: [[1, 1], [1, 1]]

The complete out-splitting of the full 2-shift is its 2-block presentation
(Example 2.4.5).

Conjugacy invariants
====================

The Bowen-Franks group ``BF(A) = Z^r / Z^r (I - A)`` is read from the Smith form
of ``I - A`` (Definition 7.4.15, Theorem 7.4.17); the sign of ``det(I - A)``
completes the Parry-Sullivan / Bowen-Franks flow-equivalence invariant (§13.6).
The Jordan form away from zero ``J^x(A)`` keeps the Jordan blocks of nonzero
eigenvalues (Definition 7.4.9, Theorem 7.4.10). Both, together with the zeta
function (:doc:`zeta_function`), are unchanged by state splitting.

.. ipython::

   In [6]: from sofic.shifts import bowen_franks_group, jordan_form_away_from_zero

   @doctest
   In [7]: bowen_franks_group(np.array([[4, 1], [1, 0]]))
   Out[7]: BowenFranksGroup(invariant_factors=(4,), det_sign=-1)

   @doctest
   In [8]: jordan_form_away_from_zero(np.array([[3, 1, 1], [2, 2, 1], [1, 2, 2]]))
   Out[8]: ((1, 2), (5, 1))

API
===

.. autofunction:: out_edges
.. autofunction:: in_edges
.. autofunction:: out_split
.. autofunction:: in_split
.. autofunction:: out_split_matrices
.. autofunction:: in_split_matrices
.. autofunction:: out_amalgamate
.. autofunction:: in_amalgamate

.. autoclass:: sofic.shifts.invariants.BowenFranksGroup
.. autofunction:: sofic.shifts.invariants.bowen_franks_group
.. autofunction:: sofic.shifts.invariants.jordan_form_away_from_zero
