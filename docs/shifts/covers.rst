.. covers.rst
.. py:module:: sofic.shifts.covers

******
Covers
******

Fischer and Krieger covers convert a Sofic shift into unifilar presentations
:cite:`Fischer1975,Krieger1984,LindMarcus1995`.

* :class:`LeftFischerCover`, :class:`RightFischerCover` — implemented.
* :class:`LeftKriegerCover`, :class:`RightKriegerCover` — construction raises
  :exc:`NotImplementedError`.

.. ipython::

   In [1]: from sofic.examples import golden_mean_shift_parry; parry = golden_mean_shift_parry()

   In [2]: parry.validate()

API
===

.. autoclass:: LeftFischerCover
   :members: from_sofic
.. autoclass:: RightFischerCover
   :members: from_sofic
.. autoclass:: LeftKriegerCover
.. autoclass:: RightKriegerCover

.. autofunction:: sofic.shifts.cover_construction.left_fischer_from_sofic
.. autofunction:: sofic.shifts.cover_construction.right_fischer_from_sofic
