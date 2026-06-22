.. rfsa.rst
.. py:module:: pensive.automata.rfsa

***
RFSA
***

Residual finite state automata (:class:`ResidualFiniteStateAutomaton`) and their
canonical form (:class:`CanonicalRFSA`) follow the residual-language theory of
Denis, Lemay, and Terlutte :cite:`Denis2002`. The extraction helpers documented
here expose RFSA-oriented entry points without claiming a fully minimized RFSA
pipeline beyond the implemented automata-backed construction.

API
===

.. autoclass:: ResidualFiniteStateAutomaton
.. autoclass:: CanonicalRFSA
   :members: from_language, from_observation_table
