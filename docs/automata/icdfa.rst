.. icdfa.rst
.. py:module:: sofic.automata.enumeration.icdfa

*****
ICDFA
*****

Enumeration of initial-connected DFAs (ICDFAs) and accessible IDFAs for
combinatorial studies of automata and topological ε-machines. The complete
ICDFA string representation follows Almeida, Moreira, and Reis; the incomplete
accessible-DFA ranking/enumeration is used in topological ε-machine enumeration
:cite:`Almeida2007,Johnson2010`. Missing transitions are written ``-1``; as
:cite:`Almeida2007` notes, the defining rules R1 and R2 still characterize the
strings, so every non-flag position -- including those before the first flag
-- may be ``-1``, and the strings biject with non-isomorphic accessible
incomplete DFAs.

API
===

.. autoclass:: ICDFAString

.. autofunction:: iter_icdfa
.. autofunction:: iter_icdfa_empty_strings
.. autofunction:: icdfa_string_to_dfa
.. autofunction:: dfa_to_icdfa_string
.. autofunction:: count_icdfa
.. autofunction:: count_icdfa_empty

.. autofunction:: sofic.automata.enumeration.idfa.iter_idfa_strings
.. autofunction:: sofic.automata.enumeration.idfa.rank_idfa_string
.. autofunction:: sofic.automata.enumeration.idfa.unrank_idfa_string
.. autofunction:: sofic.automata.enumeration.idfa.count_accessible_idfa
