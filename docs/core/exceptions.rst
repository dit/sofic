.. exceptions.rst
.. py:module:: sofic.exceptions

**********
Exceptions
**********

``sofic`` raises typed exceptions when models fail validation:

* :class:`SoficValidationError` — general structural or semantic failure
* :class:`NonDeterministicError` — DFA determinism violated
* :class:`StochasticValidationError` — invalid probability masses
* :class:`UnifilarityError` — unifilarity invariant violated
* :class:`QuasiStochasticValidationError` — quasi-stochastic invariant violated

API
===

.. autoclass:: SoficError
.. autoclass:: SoficValidationError
.. autoclass:: NonDeterministicError
.. autoclass:: StochasticValidationError
.. autoclass:: UnifilarityError
.. autoclass:: QuasiStochasticValidationError
