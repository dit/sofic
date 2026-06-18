.. transducers.rst
.. py:module:: pensive.automata.transducers

***********
Transducers
***********

Finite-state transducers map input strings to output strings.

* :class:`MealyMachine` — output on transitions.
* :class:`MooreMachine` — output on states.

API
===

.. autoclass:: Transducer
.. autoclass:: MealyMachine
.. autoclass:: MooreMachine

.. autofunction:: pensive.automata.transducer_simulation.transduce_mealy
.. autofunction:: pensive.automata.transducer_simulation.transduce_moore
