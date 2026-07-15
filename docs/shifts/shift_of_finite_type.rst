.. shift_of_finite_type.rst
.. py:module:: sofic.shifts.sft

*********************
Shift of Finite Type
*********************

A :class:`ShiftOfFiniteType` is defined by forbidden words over a finite
alphabet, following standard symbolic-dynamics terminology :cite:`LindMarcus1995`.

.. ipython::

   In [1]: from sofic.shifts import ShiftOfFiniteType

   In [2]: sft = ShiftOfFiniteType.from_forbidden_words(
      ...:     forbidden={(1, 1)},
      ...:     symbol_alphabet=frozenset({0, 1}),
      ...: )

   In [3]: sft.validate()

API
===

.. autoclass:: ShiftOfFiniteType
   :members: from_forbidden_words, from_presentation

.. autofunction:: sofic.shifts.sft_construction.from_forbidden_words
