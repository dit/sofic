.. buchi.rst
.. py:module:: sofic.automata.buchi

*****************
Büchi Automata
*****************

:class:`BuchiAutomaton` recognizes ω-languages over infinite words
:cite:`Buchi1962`.

API
===

.. autoclass:: BuchiAutomaton
   :members: accepts_lasso, accepts_omega, accepted_lasso, is_empty

.. autofunction:: sofic.automata.buchi_simulation.accepts_lasso_buchi
.. autofunction:: sofic.automata.buchi_simulation.accepts_omega_buchi
