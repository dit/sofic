.. observation_table.rst
.. py:module:: pensive.automata.observation

*****************
Observation Table
*****************

:class:`ObservationTable` supports Angluin-style learning and canonical
extraction of minimal DFAs, RFSAs, and átomatons.

API
===

.. autoclass:: ObservationTable

.. autofunction:: pensive.automata.canonical_extraction.observation_to_canonical_rfsa
.. autofunction:: pensive.automata.canonical_extraction.observation_to_atomaton
.. autofunction:: pensive.automata.canonical_extraction.observation_to_minimal_dfa
