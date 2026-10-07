.. product_alphabet_shift.rst
.. py:module:: sofic.shifts.product_alphabet_shift

**********************
Product-Alphabet Shift
**********************

A :class:`ProductAlphabetShift` is the topological support of a transducer: a
sofic subshift of the product shift on ``X x Y`` whose input and output projections are the
transducer's domain and range subshifts :cite:`LindMarcus1995`. It is the
symbolic-dynamics reading of a transducer, complementary to the
:doc:`sliding block code <sliding_block_code>`.

.. ipython::

   In [1]: from sofic import ProductAlphabetShift

   In [2]: from sofic.examples.processes import gm_to_even

   In [3]: rel = ProductAlphabetShift.from_transducer(gm_to_even())

   @doctest
   In [4]: sorted(rel.output_alphabet())
   Out[4]: ['0', '1']

   In [5]: input_shift = rel.input_shift()

Build one with :meth:`~ProductAlphabetShift.from_transducer` and recover the coordinate
shifts with :meth:`~ProductAlphabetShift.input_shift` and
:meth:`~ProductAlphabetShift.output_shift`.

API
===

.. autoclass:: ProductAlphabetShift
   :members: from_transducer, input_shift, output_shift, input_alphabet, output_alphabet
