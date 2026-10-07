.. covers.rst
.. py:module:: sofic.shifts.covers

******
Covers
******

Fischer and Krieger covers are canonical resolving presentations of a sofic
shift :cite:`Fischer1975,Krieger1984,LindMarcus1995`. Following Lind and Marcus,
*right* means right-resolving (deterministic reading forward):

* :class:`RightFischerCover` -- the unique minimal right-resolving presentation
  of an irreducible sofic shift. Reducible shifts raise
  :exc:`~sofic.exceptions.SoficValidationError`, since their minimal
  right-resolving presentation need not be unique.
* :class:`RightKriegerCover` -- the future cover, with one vertex per follower
  set of a left-infinite ray. Defined for every sofic shift; for an irreducible
  shift the Fischer cover is its unique terminal component.
* :class:`LeftFischerCover`, :class:`LeftKriegerCover` -- the left-resolving
  mirror images, built from the reversed shift.

Both constructions are exact: the Fischer cover is the terminal component of the
follower-merged subset construction, and the Krieger cover's vertices are the
images of path relations lying on cycles of the finite relation monoid.

.. ipython::

   In [1]: from sofic.graph import ATTR_SYMBOL

   In [2]: from sofic.shifts import RightKriegerCover, SoficShift

   In [3]: even = SoficShift(symbol_alphabet=frozenset("01"))

   In [4]: for s, t, a in [("A", "A", "0"), ("A", "B", "1"), ("B", "A", "1")]:
      ...:     even.graph.add_transition(s, t, **{ATTR_SYMBOL: a})

   In [5]: len(list(RightKriegerCover.from_presentation(even).states()))

API
===

.. autoclass:: LeftFischerCover
   :members: from_presentation
.. autoclass:: RightFischerCover
   :members: from_presentation
.. autoclass:: LeftKriegerCover
   :members: from_presentation
.. autoclass:: RightKriegerCover
   :members: from_presentation

.. autofunction:: sofic.shifts.cover_construction.left_fischer_cover
.. autofunction:: sofic.shifts.cover_construction.right_fischer_cover
.. autofunction:: sofic.shifts.cover_construction.left_krieger_cover
.. autofunction:: sofic.shifts.cover_construction.right_krieger_cover
