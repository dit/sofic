.. icdfa.rst
.. py:module:: sofic.automata.icdfa

*****
ICDFA
*****

Enumeration of initial-connected DFAs (ICDFAs) and accessible IDFAs for
combinatorial studies of automata and topological ε-machines. The complete
ICDFA string representation follows Almeida, Moreira, and Reis; the incomplete
accessible-DFA ranking/enumeration is used in topological ε-machine enumeration
:cite:`Almeida2007,Johnson2010`.

API
===

.. autoclass:: ICDFAString

.. autofunction:: iter_icdfa
.. autofunction:: iter_icdfa_empty_strings
.. autofunction:: icdfa_string_to_dfa
.. autofunction:: dfa_to_icdfa_string
.. autofunction:: count_icdfa
.. autofunction:: count_icdfa_empty

.. autofunction:: sofic.automata.idfa.iter_idfa_strings
.. autofunction:: sofic.automata.idfa.rank_idfa_string
.. autofunction:: sofic.automata.idfa.unrank_idfa_string
.. autofunction:: sofic.automata.idfa.count_accessible_idfa
