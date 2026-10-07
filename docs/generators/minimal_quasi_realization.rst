.. minimal_quasi_realization.rst
.. py:module:: sofic.generators.minimal_quasi_realization

*************************
Minimal Quasi-Realization
*************************

A hidden Markov model with symbol-labeled matrices :math:`T^{(x)}`, start vector
:math:`\pi`, and final vector :math:`\mathbf{1}` is a *linear representation* of
its word probabilities,

.. math::

   P(x_0 x_1 \cdots x_{L-1}) = \pi \, T^{(x_0)} T^{(x_1)} \cdots T^{(x_{L-1})} \, \mathbf{1}.

Many representations, of different dimensions, define the same process.
Schützenberger's minimization of weighted automata :cite:`Schutzenberger1961`
finds the smallest: only the part of state space that is both *reachable*,
spanned by the row vectors :math:`\pi T^{(w)}`, and *observable*, spanned by the
column vectors :math:`T^{(w)} \mathbf{1}`, matters. Its dimension is the
**process rank**, which equals the rank of the Hankel matrix

.. math::

   H_{u, v} = P(uv)

over all pairs of words :math:`u, v`, and is the dimension of Upper's
generalized-state space :cite:`Upper1997`. It is a lower bound on the number of
states of every HMM presentation of the process, including the ε-machine, and
can be strictly smaller than all of them: the minimal representation is in
general only a quasi-realization with signed entries.

:func:`minimal_quasi_realization` grows bases :math:`R` (rows) and :math:`O`
(columns) of the reachable and observable spaces breadth-first over words,
factors the finite Hankel core :math:`M = R O = C F` with
:math:`k = \operatorname{rank} M`, and returns the projected representation

.. math::

   D^{(x)} = C^{+} R \, T^{(x)} \, O F^{+}

as a :class:`~sofic.generators.quasi_realization.QuasiRealization` of
dimension :math:`k`. A final change of basis makes the final vector ``tau`` the
all-ones vector, so ``pi`` sums to one. Models whose probabilities are sympy
expressions are reduced exactly and keep object-dtype sympy entries; numeric
models use orthonormal Krylov bases and drop singular values of :math:`M`
below ``tol`` times the largest. :func:`process_rank` (also
:meth:`HiddenMarkovModel.process_rank
<sofic.generators.base.HiddenMarkovModel.process_rank>`) returns :math:`k`.

Word indices are 0-based: a length-:math:`L` word is :math:`x_{0:L} = x_0 \cdots x_{L-1}`.

.. ipython::

   In [1]: from sofic.examples import golden_mean, iid

   In [2]: from sofic.generators.minimal_quasi_realization import minimal_quasi_realization

   In [3]: iid(3).process_rank()

   In [4]: qr = minimal_quasi_realization(golden_mean(0.5))

   In [5]: qr.pi.size, qr.pi.sum()

API
===

.. autofunction:: minimal_quasi_realization
.. autofunction:: process_rank
