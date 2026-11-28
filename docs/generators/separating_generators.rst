.. separating_generators.rst
.. py:module:: sofic.generators.separating_generators

**************************************
Cryptic Order and Exact Generators
**************************************

Every generator state separates the forward and reverse causal states, so
:math:`C_g \ge \op{G}{S^+ : S^-}` :cite:`LohrAy2009`, the exact common information :cite:`Kumar2014` of the
causal-state joint :math:`P`. :mod:`sofic.generators.separating_generators` computes :math:`C_g` exactly when the
cryptic order is at most 1, and bounds it above by lookahead generators when the cryptic order is larger
(:doc:`generative_models` describes the rest of the search).

Cryptic order at most 1
=======================

The persistent crypticity :math:`p.\chi^+ = \H{S^+_1 \mid S^-_0, X_0}` equals :math:`\H{S^+_1 \mid X_{0:}}`,
because the reverse causal state :math:`S^-_0` shields the future from the past. That is
:math:`\chi^+ - \H{S^+_0 \mid X_0, S^+_1}`, Theorem 1 of Mahoney, Ellison and Crutchfield :cite:`Mahoney2009` at
:math:`k = 1`, so :math:`p.\chi^+` vanishes exactly when the cryptic order :math:`\kappa^+` is at most 1. The
forward and reverse cryptic orders are equal, the cryptic-order counterpart of the time symmetry of the Markov
order :cite:`Ellison2011`. When :math:`\chi^+ = 0` the past depends on the future only through :math:`S^+`, so
:math:`S^-` is a relabeling of :math:`S^+` and :math:`\chi^- = 0`. For orders of at least 1, the two persistent
crypticities of each block process (below) vanish together, so :math:`\max(\kappa^+, 1) = \max(\kappa^-, 1)`. Both
persistent crypticities therefore vanish exactly when the cryptic order is at most 1, and then every separating
variable of :math:`P` is a generator state and :math:`C_g = \op{G}{S^+ : S^-}`.

:func:`theorem_generator` writes the generator down. For a separating decomposition
:math:`P = \sum_w \pi_w Q_w R_w^\top` let :math:`c(w \mid a, b) = \pi_w Q_w(a) R_w(b) / P(a, b)`, and let
:math:`\phi(b, x) = S^+_1` be the forward state after :math:`X_0 = x` from :math:`S^-_0 = b`, a function because
:math:`p.\chi^+ = 0`. The kernel

.. math::

   K(x, v \mid w) = \sum_{b, b'} R_w(b) \Pr(X_0 = x, S^-_1 = b' \mid S^-_0 = b)\, c(v \mid \phi(b, x), b')

satisfies the coupled generator equations, and its word process is the source's.

The support bound
-----------------

A decomposition :math:`P = \sum_w A_w R_w^\top`, with row masses :math:`A_w = \pi_w Q_w` and column distributions
:math:`R_w`, has two moves that keep it exact and do not raise the entropy of the weights
(:func:`reduce_decomposition`). Two terms with proportional row masses or equal column factors merge. When the
column factors of the terms covering a row are linearly dependent, mass moves along the dependency until one of
those cells empties, at whichever end has less entropy (the entropy is concave along the move). At the end the
terms covering each row have independent column factors, and each row has at most one term supported on that row
alone, so at most

.. math::

   K(P) = \left\lfloor \frac{|\mathrm{supp}\, P| + \min(|S^+|, |S^-|)}{2} \right\rfloor

terms remain (:func:`support_bound_states`; the moves run on the side with fewer rows or columns). Kumar, Li and
El Gamal :cite:`Kumar2014` bound the number of values by :math:`|S^+||S^-|` and :math:`2^{\min(|S^+|, |S^-|)} - 1`.
So at cryptic order at most 1 some minimal generator has at most :math:`K(P)` states.

Computing the exact common information
--------------------------------------

:func:`separating_decomposition` works on each connected component of the support. A component in one row or
column is a single term. A component without :math:`2 \times 2` rectangles is solved exactly by
:func:`~sofic.generators.generator_bounds.one_wide_exact_common_information`. Otherwise SCIP minimizes the entropy
over decompositions with the component's :math:`K` terms
(:func:`~sofic.generators.global_bound.separating_global_bound`). The components combine as
:math:`\H{C} + \sum_c \Pr(c)\, G_c`, since the component label :math:`C` is a common function of both sides.

SCIP needs a second opinion. On Kumar, Li and El Gamal's Example 2, the erasure joint
:math:`\begin{pmatrix} 0.05 & 0 & 0.45 \\ 0 & 0.05 & 0.45 \end{pmatrix}`, presolve declares the model with
:math:`k = 2` infeasible, though :math:`W = S^+` is feasible. On the :math:`3 \times 3` joint with five cells in
``tests/test_separating_generators.py``, optimization-based bound tightening (OBBT) at :math:`k = 3` cuts off the
optimum and SCIP reports "optimal" at :math:`1.447803`, above a feasible :math:`1.343763`. With presolve and OBBT
off (``presolve=False, obbt=False``) both come out right. Each component therefore runs both settings, an
"infeasible" counts as no bound (a decomposition with :math:`K` terms always exists), and the smaller dual bound is
the component's lower bound. SCIP's primal solutions satisfy the constraints only to about :math:`10^{-6}`, which
can put their entropy below the true optimum, so each is reduced, projected onto exact decompositions by least
squares on its support, and descended, alongside multistart descents; the best exact decomposition is kept.
Without PySCIPOpt the multistart descents alone give the upper bound and such components contribute 0 to the
lower bound.

In the pipeline
---------------

:func:`~sofic.generators.generative_search.generative_model` logs the cryptic orders and the block persistent
crypticities in its ``persistent criteria`` stage. When both persistent crypticities vanish it runs the
``exact path`` stage after the ε-machine candidates: :func:`separating_decomposition`, :func:`reduce_decomposition`
and :func:`theorem_generator`, offering the generator as a candidate. Its lower bound enters ``lower``. It is
labelled ``separating global bound (numerical)``, with ``lower_certified`` False, whenever SCIP solved a
component; the certified bounds that run later can replace it. When the exact path leaves the bracket open,
the certified heaviest-term bound runs next (:doc:`generative_models`). The restricted global bound
(``global_bound=True``) defaults to :math:`K(P)` states, lowered to fit its variable cap, and at cryptic order at
most 1 with :math:`K(P)` states it enters ``lower`` too.

On the 510 processes of cryptic order at most 1 in a survey dataset of 808 two- and three-state ε-machines on two
and three symbols, the exact path reproduced every certified value to :math:`10^{-6}` bits, with at most
:math:`K(P)` states, generator residuals below :math:`3 \times 10^{-11}` and word-process errors below
:math:`2 \times 10^{-11}`. Of the 1,140 components, 970 were single rows or columns, 115 rectangle-free and 55 went
to SCIP, where the numerical lower bound came within :math:`10^{-5}` bits.

Block processes and lookahead generators
========================================

A generator whose state determines its next :math:`L - 1` symbols is the same thing as a generator of the
sliding-block process :math:`Y_t = X_{t:t+L}`. The forward causal states of :math:`Y_L` are pairs
:math:`(X_{t:t+L-1}, S^+_{t+L-1})`, and its persistent crypticity is :math:`\H{S^+_L \mid X_{0:}}`. That vanishes
exactly when :math:`L` reaches the cryptic order, which Mahoney et al. define as the least such :math:`L`
:cite:`Mahoney2011`. At :math:`L = \kappa` the construction above applies to :math:`Y_L`, and keeping the first
symbol of each emission gives a generator of the source, so

.. math::

   C_g \le \H{X_{0:L-1}} + \sum_u \Pr(u)\, \op{G}{S^+_{L-1} : S^-_0 \mid X_{0:L-1} = u}.

:func:`lookahead_generator` builds :math:`Y_L`'s ε-machine, runs :func:`separating_decomposition` and
:func:`theorem_generator` on it, projects, and checks the word process against the source exactly. On the
:math:`(2, \infty)` process (``restricted_golden_mean(2)``, cryptic order 2) it gives :math:`3/2 = C_g` with 3
states. On the butterfly process (cryptic order 2) it gives :math:`3.1219`, with 9 states, above
:math:`C_g = C_\mu = \log_2 5 = 2.3219` (the support-type bound certifies the ε-machine): the construction can
lose to the ε-machine. On the survey dataset the lookahead value equals the certified :math:`C_g` on 32 of the 49
closed processes of cryptic order 2 and on none of the closed processes of cryptic order 3, whose minimal
generators do not determine their next two symbols. It takes at most about a second there, and the pipeline offers it as a
candidate at cryptic orders 2 and 3; with a 60 s budget it lowered the pipeline's upper bound on 4 of 81 such
processes and tied on 45.

The support bound, the reduction, the kernel of :func:`theorem_generator` and the lookahead construction are not
from the literature, and none of :cite:`Mahoney2009,Mahoney2011,Crutchfield2009,Ellison2011` states the equality
of the cryptic orders.

API
===

.. autofunction:: support_bound_states
.. autofunction:: reduce_decomposition
.. autofunction:: separating_decomposition
.. autoclass:: SeparatingDecomposition
.. autofunction:: theorem_generator
.. autofunction:: lookahead_generator
.. autoclass:: LookaheadGenerator
