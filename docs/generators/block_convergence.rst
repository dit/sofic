.. block_convergence.rst

Block convergence (Anatomy of a Bit)
====================================

James, Ellison & Crutchfield :cite:`James2011` analyze stationary processes through
finite-block information curves.  Each curve ``F(ℓ) = F[X_{0:ℓ}]`` approaches a
linear asymptote ``F(ℓ) ∼ F_sub + ℓ f_μ`` with extensive rate ``f_μ`` and
subextensive intercept ``F_sub``.

Curves
------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Definition
     - Rate
     - Y-intercept
   * - ``H(ℓ)``
     - Block entropy
     - ``h_μ``
     - ``E``
   * - ``T(ℓ)``
     - Block total correlation
     - ``ρ_μ``
     - ``−E``
   * - ``R(ℓ)``
     - Residual entropy
     - ``r_μ``
     - ``E_R``
   * - ``B(ℓ)``
     - Binding (dual total correlation)
     - ``b_μ``
     - ``E_B``
   * - ``Q(ℓ)``
     - Enigmatic information ``T − B``
     - ``q_μ``
     - ``E_Q``
   * - ``W(ℓ)``
     - Local exogenous ``B + T``
     - ``w_μ``
     - ``E_W``
   * - ``I(ℓ)``
     - Block coinformation
     - ``i_μ → 0`` (finitary)
     - ``I → 0``
   * - ``J(ℓ)``
     - Block CAEKL mutual information :cite:`chan2015multivariate`
     - ``j_μ``
     - ``J_∞``

Example
-------

.. ipython::

   In [1]: from pensive.examples import golden_mean, even_process, noisy_random_phase_slip

   In [2]: eps = golden_mean(0.5)

   In [3]: diag = eps.block_convergence_diagram(max_length=6)

   @doctest float
   In [4]: diag.h_mu
   Out[4]: 0.6666666666666665

   @doctest float
   In [5]: diag.rho_mu
   Out[5]: 0.25162916738782304

   In [6]: diag.validate_identities()

   In [7]: nrps = noisy_random_phase_slip()

   @doctest float
   In [8]: nrps.block_convergence_estimates(max_length=8).h_mu
   Out[8]: 0.5

API
---

.. autofunction:: pensive.generators.block_convergence.block_convergence_diagram

.. autofunction:: pensive.generators.block_convergence.block_convergence_estimates

.. autofunction:: pensive.generators.block_convergence.plot_block_convergence_diagram

.. autoclass:: pensive.generators.block_convergence.BlockConvergenceDiagram
   :members: plot, validate_identities

.. autoclass:: pensive.generators.block_convergence.BlockConvergenceEstimates
   :members: information_anatomy
