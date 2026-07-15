.. languages.rst
.. py:module:: sofic.automata.languages.base

*****************
Regular Languages
*****************

The :mod:`sofic.automata.languages` subpackage provides regular-language
algebra via the :class:`RegularLanguage` protocol. Quotients, residuals, and
atoms use the standard regular-language viewpoint
:cite:`Kleene1956,Nerode1958,BrzozowskiTamm2011`.

.. ipython::

   In [1]: from sofic.automata import DFA, AutomatonLanguage
   In [2]: from sofic.automata.languages import union

API
===

.. autoclass:: RegularLanguage
.. autoclass:: AutomatonLanguage
   :members: from_automaton, automaton
.. autoclass:: ExplicitLanguage
   :members: alphabet

.. autofunction:: sofic.automata.languages.operations.union
.. autofunction:: sofic.automata.languages.operations.intersection
.. autofunction:: sofic.automata.languages.operations.complement
.. autofunction:: sofic.automata.languages.operations.difference
.. autofunction:: sofic.automata.languages.operations.concat
.. autofunction:: sofic.automata.languages.operations.kleene_star
.. autofunction:: sofic.automata.languages.operations.reverse
.. autofunction:: sofic.automata.languages.quotients.left_quotient
.. autofunction:: sofic.automata.languages.quotients.right_quotient
.. autofunction:: sofic.automata.languages.quotients.left_quotients
.. autofunction:: sofic.automata.languages.quotients.residuals
.. autofunction:: sofic.automata.languages.residuals.prime_residuals
.. autofunction:: sofic.automata.languages.atoms.atoms
.. autofunction:: sofic.automata.languages.atoms.prime_atoms
