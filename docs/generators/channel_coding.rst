.. channel_coding.rst
.. py:currentmodule:: sofic.generators

****************************************
Channel Coding over Channels with Memory
****************************************

The channel codes in :mod:`dit.coding` (linear block, LDPC, polar and
convolutional codes) evaluate themselves against any object implementing dit's
channel protocol. A stochastic :class:`~sofic.automata.transducers.MealyMachine`
-- including every :doc:`ε-transducer <epsilon_transducer>` -- with transitions
:math:`T(y, s' \mid s, x)` is a finite-state channel, and its
``__dit_channel__`` method wraps it as a :class:`FiniteStateChannel`. So the
machine can be handed to a dit code as is:

.. code-block:: python

   from dit.coding import hamming
   from sofic.examples.processes import trapdoor

   code = hamming(3)
   code.decode((0, 1, 1, 0, 1, 0, 0), channel=trapdoor())
   code.probability_of_error(trapdoor(), method="exact")
   code.capacity_gap(trapdoor())  # (lower, upper)

What the wrapper provides
=========================

- **Likelihoods.** :meth:`FiniteStateChannel.log_likelihood` is the forward
  algorithm :math:`\log \pi^\top \prod_t T^{(x_t, y_t)} \mathbf{1}`, exact for
  any finite-state channel, unifilar or not. Maximum-likelihood decoding and the
  exact block-error probability are therefore exact over a channel with memory.
- **Sampling.** :meth:`FiniteStateChannel.sample` draws the state path from the
  initial law, for Monte Carlo error estimates.
- **Soft information.** :meth:`FiniteStateChannel.symbol_log_likelihoods` runs
  the forward-backward (BCJR) recursions :cite:`Bahl1974` with the inputs at
  the other positions uniform and i.i.d., giving :math:`\log p(y^n \mid x_i)`.
  Belief propagation, successive cancellation and soft Viterbi decode from
  these per-position metrics. This is separate detection and decoding: it
  ignores the correlations the channel induces between positions, so those
  decoders are approximate over a channel with memory.
- **Capacity.** :meth:`FiniteStateChannel.capacity` returns the
  ``(lower, upper)`` bracket of :func:`sofic.generators.capacity.capacity`, the
  capacity *without* feedback (the codes do not use feedback). It requires a
  strongly connected unifilar channel; see :doc:`channel_capacity`.

Polar codes are designed from a channel's Bhattacharyya parameter, which a
channel with memory does not have. Design the polar code on a memoryless
channel and pass the finite-state channel to ``decode``.

Symbols
=======

dit's binary codes use the bits ``0`` and ``1``, while the example channels are
labeled with the strings ``"0"`` and ``"1"``. By default the wrapper maps digit
strings to integers; pass ``input_map`` and ``output_map`` to
:class:`FiniteStateChannel` for any other relabeling.

A one-state machine is a memoryless channel: ``binary_channel(p, p)`` gives the
same likelihoods, soft information and error probabilities as dit's
``binary_symmetric_channel(p)``.

API
===

.. autoclass:: FiniteStateChannel
   :members:
