.. transducers.rst
.. py:module:: pensive.automata.transducers

***********
Transducers
***********

Finite-state transducers map input strings to output strings, with the usual
Mealy transition-output and Moore state-output conventions
:cite:`Mealy1955,Moore1956`.

* :class:`MealyMachine` — output on transitions.
* :class:`MooreMachine` — output on states.

Epsilon moves use :data:`pensive.graph.EPSILON` explicitly.  An epsilon input
transition advances the transducer without consuming an input symbol; an
epsilon output transition emits no output symbol.  ``transduce`` expands
epsilon closures before and after each consumed input symbol, and raises
``InfiniteTransductionError`` when a productive epsilon-input cycle would give
infinitely many finite-output strings for one finite input.

Mealy transitions may optionally carry ``ATTR_PROB``.  Topological operations
treat missing probabilities as weight ``1``; stochastic helpers and composition
preserve and normalize probabilities by transducer row ``(state, input)`` when
requested.

Composition
===========

The composition helpers mirror the common ``cmpy`` transducer operations:

* :func:`pensive.automata.transducer_operations.compose_tt` serially composes
  transducers.
* :func:`pensive.automata.transducer_operations.compose_tg` composes a
  transducer with a stochastic generator and returns a joint input/output
  generator.
* :func:`pensive.automata.transducer_operations.transduce_generator` returns
  the output-only generator induced by driving a transducer with a generator.
* :func:`pensive.automata.transducer_operations.cartesian_product_tt` and
  :func:`pensive.automata.transducer_operations.cartesian_product_gg` build
  tuple-symbol Cartesian products.

For convenience, :class:`MealyMachine` also exposes ``compose``,
``joint_machine``, and ``transduce_generator`` instance methods.

API
===

.. autoclass:: Transducer
   :members: transduce, is_deterministic, is_complete, alphabets
.. autoclass:: MealyMachine
   :members: add_transition, transduce, complete, input_machine, output_machine, compose, joint_machine, transduce_generator, labeled_transition_matrices
.. autoclass:: MooreMachine
   :members: add_transition, set_output, transduce

.. autofunction:: pensive.automata.transducer_simulation.transduce_mealy
.. autofunction:: pensive.automata.transducer_simulation.transduce_moore
.. autofunction:: pensive.automata.transducer_operations.compose_tt
.. autofunction:: pensive.automata.transducer_operations.compose_tg
.. autofunction:: pensive.automata.transducer_operations.transduce_generator
.. autofunction:: pensive.automata.transducer_operations.cartesian_product_tt
.. autofunction:: pensive.automata.transducer_operations.cartesian_product_gg
