.. languages.rst
.. py:module:: pensive.automata.languages.base

*****************
Regular Languages
*****************

The :mod:`pensive.automata.languages` subpackage provides regular-language
algebra via the :class:`RegularLanguage` protocol. Quotients, residuals, and
atoms use the standard regular-language viewpoint
:cite:`Kleene1956,Nerode1958,BrzozowskiTamm2011`.

.. ipython::

   In [1]: from pensive.automata import DFA, AutomatonLanguage
   In [2]: from pensive.automata.languages import union

API
===

.. autoclass:: RegularLanguage
.. autoclass:: AutomatonLanguage
.. autoclass:: ExplicitLanguage

.. autofunction:: pensive.automata.languages.operations.union
.. autofunction:: pensive.automata.languages.operations.intersection
.. autofunction:: pensive.automata.languages.operations.complement
.. autofunction:: pensive.automata.languages.operations.reverse
.. autofunction:: pensive.automata.languages.quotients.left_quotient
.. autofunction:: pensive.automata.languages.quotients.right_quotient
.. autofunction:: pensive.automata.languages.quotients.residuals
.. autofunction:: pensive.automata.languages.atoms.atoms
.. autofunction:: pensive.automata.languages.atoms.prime_atoms
