.. textile.rst
.. py:module:: sofic.shifts.textile

**************
Textile System
**************

A :class:`TextileSystem` encodes a factor map between subshifts as a pair of
labelings over a shared edge graph — one reading the domain (input) symbols, one
the range (output) symbols. This is Nasu's machine form of a sliding block code
:cite:`Nasu1995`; operationally it is a topological transducer whose input
labeling, when right-resolving, induces a
:doc:`sliding block code <sliding_block_code>` from the input shift to the output
shift.

.. ipython::

   In [1]: from sofic import TextileSystem

   In [2]: from sofic.examples.processes import SlidingNOR

   In [3]: textile = TextileSystem.from_transducer(SlidingNOR())

   @doctest
   In [4]: textile.induced_code().memory
   Out[4]: 1

API
===

.. autoclass:: TextileSystem
   :members: from_transducer, to_transducer, to_sofic_relation, input_shift, output_shift, induced_code
