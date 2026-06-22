.. buchi.rst
.. py:module:: pensive.automata.buchi

*****************
Büchi Automata
*****************

:class:`BuchiAutomaton` recognizes ω-languages over infinite words
:cite:`Buchi1962`.

API
===

.. autoclass:: BuchiAutomaton
   :members: accepts_lasso, accepts_omega

.. autofunction:: pensive.automata.buchi_simulation.accepts_lasso_buchi
.. autofunction:: pensive.automata.buchi_simulation.accepts_omega_buchi
