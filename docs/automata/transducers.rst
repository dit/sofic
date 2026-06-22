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

API
===

.. autoclass:: Transducer
.. autoclass:: MealyMachine
.. autoclass:: MooreMachine

.. autofunction:: pensive.automata.transducer_simulation.transduce_mealy
.. autofunction:: pensive.automata.transducer_simulation.transduce_moore
