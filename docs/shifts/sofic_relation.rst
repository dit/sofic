.. sofic_relation.rst
.. py:module:: sofic.shifts.sofic_relation

**************
Sofic Relation
**************

A :class:`SoficRelation` is the topological support of a transducer: a subshift
of the product shift on ``X x Y`` whose input and output projections are the
transducer's domain and range subshifts :cite:`LindMarcus1995`. It is the
symbolic-dynamics reading of a transducer, complementary to the
:doc:`sliding block code <sliding_block_code>`.

.. ipython::

   In [1]: from sofic import SoficRelation

   In [2]: from sofic.examples.processes import GMtoEven

   In [3]: rel = SoficRelation.from_transducer(GMtoEven())

   @doctest
   In [4]: sorted(rel.output_alphabet())
   Out[4]: ['0', '1']

   In [5]: input_shift = rel.input_shift()

Build one with :meth:`~SoficRelation.from_transducer` and recover the coordinate
shifts with :meth:`~SoficRelation.input_shift` and
:meth:`~SoficRelation.output_shift`.

API
===

.. autoclass:: SoficRelation
   :members: from_transducer, input_shift, output_shift, input_alphabet, output_alphabet
