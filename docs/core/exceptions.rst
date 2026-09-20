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
approximation. This is the "explosive irreversibility" of
:cite:`Ellison2011`, whose Sec. VI B 3 gives a ternary process with two
recurrent forward causal states and countably infinitely many reverse ones.

Note that an infinite reverse presentation does *not* make
:math:`C_\mu^{-}` infinite: :math:`C_\mu^{-}` is the entropy of the
retrodictive stationary distribution, which converges whenever those weights
decay fast enough, as the geometric weights of that example do. What explodes is
the cardinality of the presentation, which is why there is nothing to return.
See :cite:`Crutchfield2009` and :cite:`Ellison2009` for the finite-state theory.

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
