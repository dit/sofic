.. predictive_rate_distortion.rst
.. py:module:: sofic.generators.predictive_rd

**************************
Predictive rate-distortion
**************************

How much of a process's predictive information survives when the observer may
only store a limited amount of memory? Predictive rate-distortion answers this
with lossy causal states. Optimal causal inference (Still, Crutchfield &
Ellison :cite:`still2010optimal`) and predictive rate-distortion (Marzen &
Crutchfield :cite:`Marzen2016`) compress the past into features :math:`R` via
the information bottleneck (Tishby, Pereira & Bialek
:cite:`tishby2000information`). The causal state :math:`S` is a sufficient
statistic of the past, so it is enough to compress :math:`S`:

.. math::

   \min_{q(r \mid s)} \; I[S; R] - \beta \, I[R; X_{0:L}],
   \qquad R - S - X_{0:L} .

The rate :math:`I[S;R]` lies in :math:`[0, C_\mu]` and the retained predictive
information :math:`I[R; X_{0:L}]` in :math:`[0, I[S; X_{0:L}]]`, where
:math:`I[S; X_{0:L}] \uparrow \mathbf{E}` as :math:`L \to \infty`. The
distortion reported is the predictive information lost,
:math:`I[S; X_{0:L}] - I[R; X_{0:L}]`.

Algorithm
=========

* **Future morphs** :math:`P(X_{0:L} = w \mid S = s) = \langle \delta_s | T^{(w)}
  | \mathbf{1} \rangle` are exact. Futures are built by prepending symbols
  to the column vectors :math:`T^{(w)}\mathbf{1}`. Words whose columns are
  proportional induce the same posterior over :math:`S`, so they are merged.
  The merge is lossless for the bottleneck, and long futures stay cheap for
  processes whose reverse-time mixed states are finite.
* **Default future length**: the smallest :math:`L` with :math:`\mathbf{E} -
  I[S; X_{0:L}] \le` ``tol`` (``1e-6``), capped at ``max_future_length``
  (a ``RuntimeWarning`` is issued if the cap is reached first).
* **Self-consistent iteration** (Blahut–Arimoto, Cover & Thomas
  :cite:`Cover2006`, §10.8; Tishby et al. :cite:`tishby2000information`):

  .. math::

     q(r \mid s) \propto q(r) \, e^{-\beta D_{KL}[P(X_{0:L} \mid s) \,\|\, q(X_{0:L} \mid r)]},
     \quad q(r) = \sum_s \pi(s) q(r \mid s),
     \quad q(w \mid r) = \sum_s P(w \mid s) \, q(s \mid r),

  with :math:`|R| = |S|`.
* **Annealing and restarts**: the :math:`\beta` grid is swept upward. Each
  :math:`\beta` is started from a perturbation of the previous optimum, from
  the one-to-one encoder :math:`R = S`, and from ``restarts`` random encoders
  (seeded by ``seed``). The candidate with the lowest Lagrangian wins. For
  :math:`\beta \le 1` the optimum is always the trivial :math:`R` (zero rate);
  as :math:`\beta \to \infty` the rate tends to :math:`C_\mu`, because the
  length-:math:`L` morphs of distinct causal states differ once
  :math:`I[S; X_{0:L}] = \mathbf{E}`.

Example
=======

.. code-block:: python

   from sofic.examples import golden_mean
   from sofic.generators.predictive_rd import predictive_rate_distortion

   curve = predictive_rate_distortion(golden_mean(0.5), 40)
   curve.rate                   # I[S; R] in bits, rises from 0 to C_mu
   curve.relevant_information   # I[R; X_{0:L}], rises from 0 to E
   curve.distortion             # I[S; X_{0:L}] - I[R; X_{0:L}]

API
===

.. autoclass:: PredictiveRateDistortionCurve
   :members:

.. autofunction:: predictive_rate_distortion
