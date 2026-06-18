.. shift_of_finite_type.rst
.. py:module:: pensive.shifts.sft

*********************
Shift of Finite Type
*********************

A :class:`ShiftOfFiniteType` is defined by forbidden words over a finite
alphabet.

.. ipython::

   In [1]: from pensive.shifts import ShiftOfFiniteType

   In [2]: sft = ShiftOfFiniteType.from_forbidden_words(
      ...:     forbidden={(1, 1)},
      ...:     symbol_alphabet=frozenset({0, 1}),
      ...: )

   In [3]: sft.validate()

API
===

.. autoclass:: ShiftOfFiniteType
   :members: from_forbidden_words, from_presentation

.. autofunction:: pensive.shifts.sft_construction.from_forbidden_words
