.. quasi_realization.rst
.. py:module:: sofic.generators.quasi_realization

*****************
Quasi-Realization
*****************

A :class:`QuasiRealization` is a matrix presentation :math:`(\pi, D_o, \tau)` of
a quasistochastic generator. The representation follows the linear
observable-operator/quasi-realization view of finite-alphabet stochastic
processes :cite:`Jaeger2000`.

API
===

.. autoclass:: QuasiRealization
.. autoclass:: QuasiStochasticModel

.. autofunction:: sofic.generators.quasi_inference.transition_matrices
.. autofunction:: sofic.generators.quasi_inference.stationary_quasidistribution
.. autofunction:: sofic.generators.quasi_inference.word_probability
