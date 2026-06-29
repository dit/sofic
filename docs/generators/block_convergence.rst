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

Exact vs estimated rates
------------------------

James (2011) curves ``H``, ``T``, ``R``, ``B``, ``Q``, and ``W`` satisfy block
identities checked by :meth:`~pensive.generators.block_convergence.BlockConvergenceDiagram.validate_identities`.
When a bidirectional ε-machine is available, the promoted rates ``h_μ``,
``ρ_μ``, ``b_μ``, and ``r_μ`` are **exact** (from the step distribution); see
:doc:`information_anatomy`.

The CAEKL curve ``J(ℓ)`` is separate: for each ``ℓ`` it is **exact** given the
ε-machine word distribution and ``dit``'s partition-minimization definition of
CAEKL.  There is no bidirectional closed form analogous to ``ρ_μ = I[X₀:S⁺₀]``.
The extensive rate ``j_μ`` is promoted from finite differences ``J(ℓ)-J(ℓ-1)``;
when :attr:`~pensive.generators.block_convergence.BlockConvergenceEstimates.caekl_rate_converged`
is ``True``, that rate is certified from a stable affine tail.

Multivariate ordering :cite:`chan2015multivariate` gives ``J(ℓ) ≤ B(ℓ) ≤ T(ℓ)``
at each block length, so (when limits exist) ``j_μ ≤ b_μ ≤ ρ_μ``.  There is no
general identity relating ``j_μ`` to the entropy rate ``h_μ`` (e.g. a fair coin
has ``h_μ = 1`` and ``j_μ = 0``).

CAEKL partition minimization costs grow quickly with ``ℓ`` (Bell-number partitions);
pass ``max_caekl_length`` to
:func:`~pensive.generators.block_convergence.block_convergence_estimates`
to cap how far ``J(ℓ)`` is computed when ``max_length`` is large.

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

.. autofunction:: pensive.generators.block_convergence.block_caekl

.. autofunction:: pensive.generators.block_convergence.block_convergence_diagram

.. autofunction:: pensive.generators.block_convergence.block_convergence_estimates

.. autofunction:: pensive.generators.block_convergence.plot_block_convergence_diagram

.. autoclass:: pensive.generators.block_convergence.BlockConvergenceDiagram
   :members: plot, validate_identities, j_mu, J_inf, caekl_rate_converged

.. autoclass:: pensive.generators.block_convergence.BlockConvergenceEstimates
   :members: information_anatomy
