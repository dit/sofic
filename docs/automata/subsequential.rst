.. subsequential.rst
.. py:module:: sofic.automata.subsequential

************************************
Subsequential and Weighted Transducers
************************************

The deterministic and weighted branches of the finite-state transducer
hierarchy :cite:`Mohri2009`.

* :class:`SubsequentialTransducer` — an input-deterministic (sequential)
  transducer with a per-state final-output string, appended on reaching the end
  of input.
* :class:`WeightedFiniteStateTransducer` — a transducer whose edges carry
  weights in a semiring. The ``probability`` semiring uses ``(+, x, 0, 1)``; the
  ``tropical`` semiring uses ``(min, +, +inf, 0)`` (shortest-path / Viterbi
  weights). It generalizes the stochastic Mealy machine underlying the
  :doc:`ε-transducer <../generators/epsilon_transducer>`.

.. ipython::

   In [1]: from sofic.automata.subsequential import SubsequentialTransducer

   In [2]: t = SubsequentialTransducer(input_alphabet=frozenset('ab'), output_alphabet=frozenset('xy'), initial_states=frozenset({'q0'}), final_output={'q0': ('y',)})

   In [3]: t.graph.add_state('q0'); _ = t.add_transition('q0', 'q0', 'a', 'x')

   @doctest
   In [4]: sorted(t.transduce('aa'))
   Out[4]: [('x', 'x', 'y')]

   In [5]: from sofic.automata.subsequential import WeightedFiniteStateTransducer

   In [6]: from sofic.examples.processes import BinaryChannel

   In [7]: w = WeightedFiniteStateTransducer.from_transducer(BinaryChannel(0.1, 0.2))

   @doctest float
   In [8]: w.weight(['0'], ['0'])
   Out[8]: 0.9

The predicates :func:`sofic.properties.is_sequential_transducer` and
:func:`sofic.properties.is_subsequential_transducer` test these structural
properties, and :meth:`sofic.generators.epsilon_transducer.EpsilonTransducer.to_wfst`
/ :meth:`~sofic.generators.epsilon_transducer.EpsilonTransducer.from_wfst`
convert between the weighted and computational-mechanics views.

API
===

.. autoclass:: SubsequentialTransducer
   :members: transduce, is_subsequential
.. autoclass:: WeightedFiniteStateTransducer
   :members: from_transducer, weight

.. autofunction:: sofic.properties.is_sequential_transducer
.. autofunction:: sofic.properties.is_subsequential_transducer
.. autofunction:: sofic.properties.is_unifilar_transducer
