.. renyi.rst
.. py:module:: sofic.generators.renyi

**********************************************
Rényi Entropy Rates and Thermodynamic Spectrum
**********************************************

The Rényi entropy rate of order :math:`\alpha \in [0, \infty]` of a stationary
process, in bits per symbol, is

.. math::

   h_\alpha = \lim_{n \to \infty} \tfrac{1}{n} H_\alpha[X_{0:n}],
   \qquad H_\alpha[X_{0:n}] = \tfrac{1}{1 - \alpha} \log_2 \sum_w P(w)^\alpha,

where :math:`w` ranges over the words of length :math:`n`. The pressure is
:math:`\mathcal{P}(\beta) = \lim_n \tfrac{1}{n} \log_2 \sum_w P(w)^\beta`, so
:math:`h_\alpha = \mathcal{P}(\alpha) / (1 - \alpha)`.

Transfer matrix formula
=======================

On a unifilar presentation with labeled matrices :math:`T^{(x)}`, restricted to
the states of positive stationary probability,

.. math::

   \mathcal{P}(\beta) = \log_2 \rho(A_\beta), \qquad
   A_\beta = \sum_x \big(T^{(x)}\big)^{\circ \beta}, \qquad \beta \ge 0,

where :math:`\circ \beta` raises the positive entries to the power
:math:`\beta` and :math:`\rho` is the spectral radius. For Markov chains this is
:cite:`Rached2001`. For a unifilar presentation, each start state :math:`s`
emits a word :math:`w` along at most one path, with probability
:math:`p(w \mid s)`, and :math:`P(w) = \sum_s \pi_s p(w \mid s)` over the
:math:`N` support states. For :math:`\beta \ge 0`,

.. math::

   N^{-1} \sum_s \big(\pi_s p(w \mid s)\big)^\beta \le P(w)^\beta
   \le N^\beta \sum_s \big(\pi_s p(w \mid s)\big)^\beta .

Summing over :math:`w` gives :math:`(\pi^{\circ \beta})^\top A_\beta^n \mathbf{1}`
up to these constant factors. Both outer vectors are positive on the support, so
the growth rate is :math:`\rho(A_\beta)`. Negative :math:`\beta` is not
supported, because then each word is weighted by its *least* likely start
state.

Special orders:

* :math:`h_0` is the topological entropy of the process support: :math:`A_0` is
  the support adjacency matrix.
* :math:`h_1 = h_\mu`, the Shannon entropy rate, is the limit of the formula at
  :math:`\alpha \to 1`. It is computed by
  :meth:`~sofic.generators.base.HiddenMarkovModel.entropy_rate`.
* :math:`h_\infty = -\lim_n \tfrac{1}{n} \log_2 \max_w P(w)` is the min-entropy
  rate. It equals minus the maximum mean :math:`\log_2` edge probability over
  cycles of the support graph, computed with Karp's algorithm.

:math:`h_\alpha` is non-increasing in :math:`\alpha`.

Non-unifilar generators are first converted with
:meth:`EpsilonMachine.from_hmm(model, max_states=...)
<sofic.generators.epsilon_machine.EpsilonMachine.from_hmm>`. If the mixed-state
presentation does not close within ``max_states`` this raises
:class:`~sofic.exceptions.MixedStateExplosionError`.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: from sofic.generators.renyi import renyi_entropy_rate

   @doctest float
   In [3]: renyi_entropy_rate(golden_mean(0.5), 0.0)
   Out[3]: 0.6942419136306174

   @doctest float
   In [4]: renyi_entropy_rate(golden_mean(0.5), float("inf"))
   Out[4]: 0.5

Large deviations
================

Let :math:`U_n = -\tfrac{1}{n} \log_2 P(X_{0:n})`. Its scaled cumulant generating
function in base 2 is :math:`\lim_n \tfrac{1}{n} \log_2 \mathbb{E}[2^{n t U_n}] = \mathcal{P}(1 - t)`.
By the Gärtner-Ellis theorem :cite:`DemboZeitouni1998`, the rate function of
:math:`U_n` is the Legendre transform

.. math:: I(u) = \sup_\beta \big[(1 - \beta) u - \mathcal{P}(\beta)\big].

:func:`rate_function` takes the supremum over :math:`\beta \in [0, \beta_{\max}]`
on a grid, refined with :func:`scipy.optimize.minimize_scalar`. That gives
:math:`I` on :math:`[h_\infty, u_0]`, where :math:`u_0 = -\mathcal{P}'(0)`.
:math:`I` is non-negative and vanishes at :math:`u = h_\mu`. It is ``inf`` below
:math:`h_\infty`, and ``nan`` above :math:`u_0`, where :math:`\beta < 0` would
be needed. Near :math:`h_\infty` the optimal :math:`\beta` diverges, so values
there are lower bounds that tighten as ``beta_max`` grows. For an i.i.d. source
the result is Cramér's rate function.

API
===

.. autofunction:: renyi_entropy_rate

.. autofunction:: pressure

.. autofunction:: rate_function
