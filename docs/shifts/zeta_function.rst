.. zeta_function.rst

*************************************
Periodic Points and the Zeta Function
*************************************

The number ``p_n`` of points ``x`` with ``sigma^n x = x`` and the zeta function
``zeta(t) = exp(sum_n p_n t^n / n)`` are conjugacy invariants
:cite:`LindMarcus1995` (§6.4). All counts are exact Python integers; the zeta
function is a sympy rational function (install ``sofic[symbolic]``).

* A :class:`~sofic.shifts.tmc.TopologicalMarkovChain` is an edge shift, so
  ``p_n = tr A^n`` with edge multiplicities and ``zeta(t) = 1 / det(I - tA)``
  (Theorem 6.4.6).
* A :class:`~sofic.shifts.sft.ShiftOfFiniteType` given by forbidden words of
  length at most ``L`` is recoded as the edge shift on allowed ``L``-blocks,
  and the same trace formula applies.
* A :class:`~sofic.shifts.sofic.SoficShift` (or a presentation-only SFT) uses
  Manning's formula on a right-resolving presentation: with ``A_j`` the
  ``j``-th signed subset matrix, ``p_n = sum_j (-1)^(j+1) tr A_j^n`` and
  ``zeta(t) = prod_j det(I - tA_j)^((-1)^j)`` (Theorem 6.4.8). Counting cycles
  of the presentation itself overcounts points that have several
  presentations, such as ``0^infinity`` in the even shift.

.. ipython::

   In [1]: from sofic.shifts import SoficShift

   In [2]: even = SoficShift(symbol_alphabet=frozenset("ab"))

   In [3]: even.add_transition(0, 0, "a"); even.add_transition(0, 1, "b"); even.add_transition(1, 0, "b");

   @doctest
   In [4]: [even.periodic_points(n) for n in range(1, 7)]
   Out[4]: [2, 2, 5, 6, 12, 17]

   In [5]: even.zeta_function()

API
===

.. autofunction:: sofic.shifts.algorithms.periodic_points
.. autofunction:: sofic.shifts.algorithms.zeta_function
.. autofunction:: sofic.shifts.algorithms.signed_subset_matrices
.. autofunction:: sofic.shifts.algorithms.reciprocal_characteristic_polynomial
.. automethod:: sofic.shifts.tmc.TopologicalMarkovChain.periodic_points
.. automethod:: sofic.shifts.tmc.TopologicalMarkovChain.zeta_function
.. automethod:: sofic.shifts.sft.ShiftOfFiniteType.periodic_points
.. automethod:: sofic.shifts.sft.ShiftOfFiniteType.zeta_function
.. automethod:: sofic.shifts.sofic.SoficShift.periodic_points
.. automethod:: sofic.shifts.sofic.SoficShift.zeta_function
