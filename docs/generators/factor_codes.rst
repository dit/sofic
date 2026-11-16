.. factor_codes.rst
.. py:module:: sofic.generators.factor_codes

*******************************
Factor codes on processes
*******************************

A sliding block code :math:`\Phi` with memory :math:`m`, anticipation :math:`a`
and window :math:`w = m + 1 + a` (:class:`~sofic.shifts.sliding_block_code.SlidingBlockCode`)
maps a process :math:`X` to its *image process* :math:`Y = \Phi(X)`
:cite:`LindMarcus1995` (§1.5):

.. math::

   Y_t = \Phi(X_{t:t+w}),

which is :math:`(\Phi X)_{t+m}` in the two-sided indexing of Lind & Marcus; for a
stationary :math:`X` the shift by :math:`m` is immaterial. The law of
:math:`Y_{0:n}` is the pushforward of the law of :math:`X_{0:n+w-1}`:

.. math::

   \Pr(Y_{0:n} = y_{0:n}) = \sum_{x_{0:n+w-1} \,:\, \Phi(x_{0:n+w-1}) = y_{0:n}} \Pr(X_{0:n+w-1} = x_{0:n+w-1}).

:meth:`SlidingBlockCode.apply_to_process <sofic.shifts.sliding_block_code.SlidingBlockCode.apply_to_process>`
(equivalently :func:`image_process`) builds a Mealy HMM for :math:`Y` by composing
the code's sliding-window transducer with the generator
(:func:`~sofic.automata.transducer_operations.transduce_generator`). States are
pairs :math:`(q, x_{t-w+1:t})` of a generator state and the last :math:`w - 1`
symbols; the initial law is that of :math:`(Q_{w-1}, X_{0:w-1})`, so the composed
transducer starts synchronized with the source and the first output is
:math:`Y_0 = \Phi(X_{0:w})`. A ``ValueError`` is raised if the generator can
emit a :math:`w`-block missing from the code's ``block_map``.

Invariants
==========

A stationary coding never increases the entropy rate, :math:`h_\mu(\Phi(X)) \le
h_\mu(X)` :cite:`Gray1990`, so a conjugacy (an invertible code) preserves it; the
topological analogue is :cite:`LindMarcus1995` (§4.1). Neither the excess entropy
nor the statistical complexity is a conjugacy invariant. The cleanest witness is
the higher block code.

Higher block presentations
==========================

The higher block code :math:`\beta_k` :cite:`LindMarcus1995` (§1.4) recodes
:math:`X` to overlapping :math:`k`-tuples,

.. math::

   Y_t = X_{t:t+k} = (X_t, \ldots, X_{t+k-1}),

so :math:`\Pr(Y_{0:n} = y_{0:n}) = \Pr(X_{0:n+k-1} = x_{0:n+k-1})` when the tuples
overlap consistently and spell :math:`x_{0:n+k-1}`, and zero otherwise.
:meth:`HiddenMarkovModel.higher_block <sofic.generators.base.HiddenMarkovModel.higher_block>`
returns this presentation; it is unifilar whenever the source is. The 1-block
code :math:`y \mapsto y[0]` inverts :math:`\beta_k`, so :math:`\beta_k` is a
conjugacy and :math:`h_\mu(Y) = h_\mu(X)`. However, since
:math:`H[Y_{0:L}] = H[X_{0:L+k-1}]` and
:math:`\mathbf{E} = \lim_L (H[X_{0:L}] - L\, h_\mu)` :cite:`CrutchfieldFeldman2003`,

.. math::

   \mathbf{E}(Y) = \mathbf{E}(X) + (k - 1)\, h_\mu(X),

and, because a :math:`Y`-past fixes both the causal state of :math:`X` and the
last :math:`k - 1` symbols (which the next tuple repeats),

.. math::

   C_\mu(Y) = H[\mathcal{S}_0, X_{-(k-1):0}] = C_\mu(X) + H[X_{-(k-1):0} \mid \mathcal{S}_0],

with :math:`\mathcal{S}_0` the causal state of :math:`X` after :math:`X_{-1}`.
Both identities are checked in ``tests/test_factor_codes.py``.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: from sofic.shifts.sliding_block_code import SlidingBlockCode

   In [3]: machine = golden_mean(0.5)

   In [4]: pairs = machine.higher_block(2)

   In [5]: sorted(pairs.observation_alphabet)

   In [6]: float(pairs.entropy_rate()), float(machine.entropy_rate())

   In [7]: first = SlidingBlockCode({(y,): y[0] for y in pairs.observation_alphabet})

   In [8]: first.apply_to_process(pairs).is_equal_process(machine)

API
===

.. autofunction:: image_process
.. autofunction:: higher_block
