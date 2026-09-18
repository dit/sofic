.. exceptions.rst
.. py:module:: sofic.exceptions

**********
Exceptions
**********

``sofic`` raises typed exceptions when models fail validation:

* :class:`SoficValidationError` — general structural or semantic failure
* :class:`NonDeterministicError` — DFA determinism violated
* :class:`StochasticValidationError` — invalid probability masses
* :class:`MixedStateExplosionError` — mixed-state presentation did not close
* :class:`UnifilarityError` — unifilarity invariant violated
* :class:`QuasiStochasticValidationError` — quasi-stochastic invariant violated

Infinite mixed-state presentations
==================================

:class:`MixedStateExplosionError` is not a "bad model" error. A stationary
process can have a finite forward ε-machine and *infinitely many* retrodictive
causal states, so :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_time_reversed`
has no presentation to return and raises rather than degrading to an
approximation. Causal irreversibility :math:`\Delta C_\mu = C_\mu - C_\mu^{-}`
is then :math:`-\infty`; see :cite:`Crutchfield2009` and :cite:`Ellison2009` for
the finite-state theory.

It subclasses :class:`StochasticValidationError`, so existing handlers continue
to catch it. Raising the ``max_states`` cap will not help when the belief set is
genuinely infinite.

API
===

.. autoclass:: SoficError
.. autoclass:: SoficValidationError
.. autoclass:: NonDeterministicError
.. autoclass:: StochasticValidationError
.. autoclass:: MixedStateExplosionError
.. autoclass:: UnifilarityError
.. autoclass:: QuasiStochasticValidationError
