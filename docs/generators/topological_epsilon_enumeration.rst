.. topological_epsilon_enumeration.rst
.. py:module:: pensive.generators.topological_epsilon_enumeration

****************************************
Topological ε-Machine Enumeration
****************************************

Enumerate canonical topological ε-machines via IDFA strings (research
utilities). The implementation follows the accessible-DFA enumeration used by
Johnson, Crutchfield, Ellison, and McTague, building on the ICDFA string
representation :cite:`Johnson2010,Almeida2007`.

API
===

.. autofunction:: iter_topological_epsilon_machines
.. autofunction:: iter_topological_epsilon_strings
.. autofunction:: count_topological_epsilon_machines
.. autofunction:: is_topological_epsilon_string
.. autofunction:: is_canonical_topological_epsilon
