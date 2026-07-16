.. sliding_block_code.rst
.. py:module:: sofic.shifts.sliding_block_code

*******************
Sliding Block Code
*******************

A :class:`SlidingBlockCode` is the symbolic-dynamics form of a transducer: a
factor map ``Phi`` between shift spaces induced by a local rule on a window of
``memory + 1 + anticipation`` consecutive symbols. The Curtis-Hedlund-Lyndon
theorem characterizes exactly the shift-commuting continuous maps this way
:cite:`LindMarcus1995`.

.. ipython::

   In [1]: from sofic.shifts.sliding_block_code import SlidingBlockCode, full_shift

   In [2]: nor = SlidingBlockCode({('0', '0'): '1', ('0', '1'): '0', ('1', '0'): '0', ('1', '1'): '0'}, memory=1, anticipation=0)

   @doctest
   In [3]: nor.apply_word(['0', '0', '1', '0'])
   Out[3]: ('1', '0', '0')

   In [4]: image = nor.apply(full_shift({'0', '1'}))

   @doctest
   In [5]: from sofic import SoficShift; isinstance(image, SoficShift)
   Out[5]: True

:meth:`~SlidingBlockCode.apply` builds the image subshift via the higher-block
construction; :meth:`~SlidingBlockCode.compose` composes codes; and
:meth:`~SlidingBlockCode.to_transducer` realizes the code as a Mealy machine.
The inverse bridge is
:meth:`sofic.automata.transducers.MealyMachine.to_sliding_block_code`.

API
===

.. autoclass:: SlidingBlockCode
   :members: apply, apply_word, compose, is_right_resolving, to_transducer

.. autofunction:: full_shift
