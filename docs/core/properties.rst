.. properties.rst
.. py:module:: pensive.properties

**********
Properties
**********

Structural predicates for validating and classifying state machines. The
determinism, unifilarity, and right-resolving checks mirror terminology from
finite automata, symbolic dynamics, and computational mechanics
:cite:`RabinScott1959,LindMarcus1995,Crutchfield1994`.

.. ipython::

   In [1]: from pensive.examples import golden_mean; from pensive.properties import is_unifilar_emissions

   In [2]: eps = golden_mean(0.5)

   @doctest
   In [3]: is_unifilar_emissions(eps)
   Out[3]: True

API
===

.. autofunction:: is_unifilar_labeled
.. autofunction:: is_unifilar_symbols
.. autofunction:: is_unifilar_emissions
.. autofunction:: is_deterministic_automaton
.. autofunction:: is_deterministic_markov
.. autofunction:: is_deterministic_transducer
