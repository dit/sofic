.. nonunifilar_entropy_rate.rst
.. py:module:: sofic.generators.measures
   :no-index:

*************************************
Entropy Rate of Non-Unifilar Models
*************************************

For a unifilar presentation the entropy rate is the state-averaged
next-symbol entropy. For a non-unifilar hidden Markov model there is no closed
form :cite:`blackwell1957entropy`: the observer's belief about the hidden state
(the mixed state) generically ranges over an infinite set
:cite:`jurgens2021shannon`. :meth:`~sofic.generators.base.HiddenMarkovModel.entropy_rate`
therefore takes a ``method``:

``"exact"``
    The unifilar closed form, applied to the mixed-state presentation
    :cite:`Ellison2009` when the presentation is non-unifilar and its mixed
    states close within ``max_states``.
``"bounds"``
    The Cover & Thomas sandwich (Thm. 4.5.1 :cite:`Cover2006`, written with
    0-based indices),

    .. math::

        H[X_n \mid X_{0:n}, S_0] \;\le\; h_\mu \;\le\; H[X_n \mid X_{0:n}],

    with the lower bound nondecreasing and the upper bound nonincreasing in
    :math:`n`, iterated until the gap is below ``tol``. Convergence is often
    exponential :cite:`Travers2013`.
``"blackwell"``
    A Monte Carlo average of :math:`H[X \mid \eta_t]` along a sampled
    mixed-state trajectory :cite:`blackwell1957entropy,jurgens2021shannon`.
``"auto"`` (default)
    ``"exact"`` when the mixed states close, otherwise ``"bounds"`` (warning
    if they have not met ``tol``). It never silently returns a stochastic
    estimate.

The simple nonunifilar source has countably many mixed states, so ``"auto"``
falls back to the bounds:

.. ipython::

   In [1]: from sofic.examples import sns

   In [2]: from sofic.generators.measures import entropy_rate_bounds, entropy_rate_blackwell

   In [3]: entropy_rate_bounds(sns(), 10)

   In [4]: sns().entropy_rate()

   In [5]: entropy_rate_blackwell(sns(), n_samples=20_000, seed=0)

Statistical complexity dimension
================================

When the mixed states do not close, the statistical complexity diverges. Its
divergence rate, the statistical complexity dimension :math:`d_\mu`, is the
information dimension of the Blackwell measure on the mixed states
:cite:`jurgens2021divergent`. The mixed-state dynamic is an iterated function
system on the simplex, and :math:`d_\mu` is bounded above by (and, under the
open set condition, equal to) its Lyapunov dimension. That dimension is the
Kaplan–Yorke formula with the entropy rate as the leading exponent. A process
with finitely many causal states has :math:`d_\mu = 0`.

.. ipython::

   In [6]: from sofic.generators.complexity_dimension import statistical_complexity_dimension

   In [7]: statistical_complexity_dimension(sns(), n_samples=5_000, seed=0).dimension

API
===

.. autofunction:: sofic.generators.measures.entropy_rate
.. autofunction:: sofic.generators.measures.entropy_rate_bounds
.. autofunction:: sofic.generators.measures.entropy_rate_blackwell
.. autoclass:: sofic.generators.measures.EntropyRateEstimate
.. autofunction:: sofic.generators.measures.mixed_state_walk
.. autofunction:: sofic.generators.measures.batch_means_stderr
.. autofunction:: sofic.generators.complexity_dimension.statistical_complexity_dimension
.. autofunction:: sofic.generators.complexity_dimension.ifs_lyapunov_dimension
.. autoclass:: sofic.generators.complexity_dimension.StatisticalComplexityDimension
