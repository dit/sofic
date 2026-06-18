.. exceptions.rst
.. py:module:: pensive.exceptions

**********
Exceptions
**********

``pensive`` raises typed exceptions when models fail validation:

* :class:`PensiveValidationError` — general structural or semantic failure
* :class:`NonDeterministicError` — DFA determinism violated
* :class:`StochasticValidationError` — invalid probability masses
* :class:`UnifilarityError` — unifilarity invariant violated
* :class:`QuasiStochasticValidationError` — quasi-stochastic invariant violated

API
===

.. autoclass:: PensiveError
.. autoclass:: PensiveValidationError
.. autoclass:: NonDeterministicError
.. autoclass:: StochasticValidationError
.. autoclass:: UnifilarityError
.. autoclass:: QuasiStochasticValidationError
