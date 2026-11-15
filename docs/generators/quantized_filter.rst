.. py:module:: sofic.generators.quantized_filter

****************************************
Finite-state quantized filtering
****************************************

A known HMM can have infinitely many reachable Bayesian beliefs. Instead of
trying to enumerate them exactly, :func:`quantized_filter` rounds the filter
onto a finite log-odds grid and builds only the labels reachable from its
specified initial belief. The construction follows Feng and Meng
:cite:`feng2026recursive`, Theorem 3.1, Theorem 4.3, and Appendix A.4.

This is an opt-in approximation, not a replacement for
:doc:`mixed_state_presentation`. It does not learn the HMM, minimize the number
of states, or solve the :doc:`predictive_rate_distortion` bottleneck.

.. warning::

   The bounds below are theorem-derived bounds in real arithmetic. This
   implementation uses float64 and checks rounding distances numerically;
   ``result.bounds.rigorous`` is **False**. It is not an outward-rounded
   interval certificate. Precision too fine to resolve, nonpositive rounded
   beliefs, and reachable-state cap overruns raise ``ValueError``.

   Only numeric :class:`~sofic.generators.moore.MooreHMM` models with strictly
   positive hidden transitions are accepted. Emissions may be zero. A positive
   *summed* transition matrix of an arbitrary Mealy HMM does not imply the
   required contraction, so converting an arbitrary Mealy model is not a way
   to obtain these guarantees.

Three different operations
==========================

The observer's retained label identifies a representative predictive belief
:math:`z`. It can be used in three ways :cite:`feng2026recursive`:

* **Event query:** supply the likelihood vector :math:`v_B(i)=P(B\mid S_t=i)`
  for a future event, and decode :math:`zv_B` with ``event_probability``.
  Its log error is bounded independently of the event's horizon. Constructing
  the likelihood vector is the caller's responsibility.
* **Recursive generation:** ``to_generator()`` returns a fixed-start,
  unifilar :class:`~sofic.generators.mealy.MealyHMM`. It emits from the
  representative belief and rounds again after each output. Conditional
  errors accumulate, so a fixed *total* loss requires finer precision at
  longer horizons.
* **Private-state simulation:** ``private_generator(state=...)`` initializes
  a copy of the original HMM from a retained belief. It samples a hidden
  state once and then follows the original dynamics. Only initialization
  uses the approximation; its event-law error is horizon-independent
  (Proposition 4.4). The paper's :math:`K+m` label count assumes the observation
  and simulation phases reuse storage. Keeping both live concurrently is a
  different memory model.

``state_after(history)`` begins at the exact initial label 0. ``step(label,
next_symbol)`` maintains the observer online while retaining only a label.
Unknown or impossible observations are rejected. Bounds concern histories
starting from the specified initial law, not arbitrary resets of the source
alone. The arrays exposed by the result are read-only, and source models are
copied so later model edits cannot change the observer's meaning.

Accuracy and units
==================

All public tolerances, decoded log errors, and KL losses use **bits**. The
paper instead uses nats for losses. A requested bit budget :math:`\varepsilon`
is converted internally to :math:`\epsilon=(\log 2)\varepsilon` nats.

For the departure-emission Moore convention, let :math:`P` be the hidden
transition matrix and :math:`D_x` the diagonal emission matrix. The predictive
belief :math:`u_t=P(S_t\mid X_{0:t})` evolves as

.. math::

   u_{t+1}=\Psi_{X_t}(u_t),\qquad
   \Psi_x(u)=\frac{uD_xP}{uD_x\mathbf{1}}.

Positive transitions make this a contraction in the natural-log Hilbert
metric :math:`d_H`, with Birkhoff coefficient :math:`\kappa<1`:

.. math::

   d_H(\Psi_x(u),\Psi_x(z))\leq\kappa d_H(u,z).

If each rounding introduces at most :math:`\eta`, then
:math:`d_H(u_t,z_t)\leq\eta/(1-\kappa)`. The constructor chooses a grid spacing
at most :math:`(1-\kappa)\xi`, where

.. math::

   \xi=\begin{cases}
      \epsilon, & \text{event queries},\\
      \epsilon/T, & \text{uniform word log error},\\
      \log(1+\sqrt{\epsilon/T}), & \text{total forward KL}.
   \end{cases}

Use ``loss="event"`` without a horizon, or ``loss="word"`` / ``loss="kl"``
with a positive ``horizon=T``. The latter KL bound is uniform over supplied
observation histories, which is stronger than averaging over histories.
``bounds.error_bound`` reports the bound for the requested criterion;
``word_log_error_bound(T)`` and ``total_kl_bound(T)`` can also inspect a
constructed observer at another horizon.

With :math:`L=\log(p_{\max}/p_{\min})`, the full grid has at most

.. math::

   1+\left(1+\left\lceil\frac{2L}{\eta}\right\rceil\right)^{m-1}

labels, including the exact initial label. Only reachable labels are built;
identical transition rows have an exact one- or two-label shortcut. The
constructor honors an explicit model initial law or ``start=...``; stationarity
is used only when no initial law is supplied. ``grid_states`` reports the
unpruned construction, ``n_states`` the reachable count, and ``memory_bits``
is :math:`\lceil\log_2 K\rceil`, **not** stationary state entropy, :math:`C_\mu`,
:math:`C_g`, or an information-bottleneck rate.

A per-symbol KL allowance :math:`a` corresponds to total budget :math:`aT`.
Then the required grid precision need not change with :math:`T`. The horizon
cost is not a universal barrier to fixed-memory models with a fixed perplexity
tolerance (Section 6.1 of :cite:`feng2026recursive`).

Example: queries versus generation
========================================

The two-regime model of the paper has hidden flip probability 0.2 and emission
noise 0.1. The following uses existing HMM word-probability and sampling APIs:

.. code-block:: python

   import math
   from itertools import product

   import numpy as np
   from sofic import MooreHMM, quantized_filter
   from sofic.generators.matrices import emission_tensors

   source = MooreHMM(
       observation_alphabet=frozenset("01"),
       initial_distribution={0: 0.5, 1: 0.5},
   )
   for state in (0, 1):
       source.add_transition(state, state, 0.8)
       source.add_transition(state, 1 - state, 0.2)
   source.set_emission_distribution(0, {"0": 0.9, "1": 0.1})
   source.set_emission_distribution(1, {"0": 0.1, "1": 0.9})

   # Match the paper's 0.1-nat event-error budget explicitly.
   observer = quantized_filter(source, 0.1 / math.log(2))
   assert (observer.grid_states, observer.n_states, observer.memory_bits) == (72, 29, 5)
   assert not observer.bounds.rigorous
   history = tuple("0110100111")
   label = observer.state_after(history)

   # Query the event "no alarm in the next 12 observations".
   _, matrices = emission_tensors(source)
   likelihood = np.linalg.matrix_power(matrices["0"], 12) @ np.ones(2)
   event_probability = observer.event_probability(likelihood, state=label)

   # The same approximate future law can be sampled with private hidden states.
   private = observer.private_generator(state=label)
   observations, hidden_states = private.sample(100, rng=np.random.default_rng(0))
   assert np.isclose(private.word_probability("0" * 12), event_probability)

   # In contrast, this generator rounds again after every visible output.
   rounded = observer.to_generator()
   exact_belief = source.forward(history, normalize=True)[-1]
   words = list(product("01", repeat=8))
   p = np.array([source.word_probability(w, start=exact_belief) for w in words])
   q = np.array([rounded.word_probability(w, start=label) for w in words])
   total_kl_bits = float(p @ np.log2(p / q))
   assert total_kl_bits <= observer.bounds.total_kl_bound(8) + 1e-12

   # Choose precision for a particular total-KL forecast budget instead.
   forecast = quantized_filter(source, 0.1 / math.log(2), loss="kl", horizon=100)
   assert (forecast.grid_states, forecast.n_states) == (225, 51)
   assert forecast.bounds.error_bound <= 0.1 / math.log(2) + 1e-12

These reachable-state counts reproduce the log-grid rows of Table 3 using
float64 comparisons, not the paper's interval checks. ``to_generator()``
returns an ordinary serializable Mealy HMM; it does not claim to preserve the
source exactly or to be a minimal epsilon-machine.

The explicit finite-word KL calculation grows exponentially in the forecast
horizon. For larger problems, the KL chain rule :cite:`Cover2006` gives the
same total by summing expected conditional KLs along **source** prefixes,
running the candidate's own updates along each prefix. Monte Carlo estimates
of those expectations are not uniform error certificates. Existing
:doc:`relative_entropy_rate` functions instead concern stationary, asymptotic
bits per symbol.

API
===

.. autofunction:: quantized_filter

.. autoclass:: QuantizedFilter
   :members:

.. autoclass:: QuantizedFilterBounds
   :members:
