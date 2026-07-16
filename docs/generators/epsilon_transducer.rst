.. epsilon_transducer.rst
.. py:module:: sofic.generators.epsilon_transducer

**************
ε-Transducer
**************

An :class:`EpsilonTransducer` is the input/output generalization of the
:doc:`ε-machine <epsilon_machine>`: the minimal unifilar stochastic Mealy
machine presenting a channel that transforms one process into another
:cite:`Barnett2015`. It is the tuple ``(X, Y, S, T)`` of input alphabet, output
alphabet, causal states, and conditional-symbol transition probabilities
``T(y, s' | s, x)``.

Structurally it extends the probability-aware :class:`~sofic.automata.transducers.MealyMachine`
with a causal-state initial distribution and channel information measures.
Unifilarity means the observed pair ``(x, y)`` determines the successor causal
state.

Construction
============

Minimize a joint-unifilar stochastic transducer to its causal states:

.. ipython::

   In [1]: from sofic import EpsilonTransducer

   In [2]: from sofic.examples.processes import BinaryChannel, GMtoEven

   In [3]: eps = EpsilonTransducer.from_channel(BinaryChannel(0.1, 0.2))

   @doctest
   In [4]: len(list(eps.states()))
   Out[4]: 1

   @doctest
   In [5]: EpsilonTransducer.from_channel(GMtoEven()).is_unifilar()
   Out[5]: True

The channel can also be read off a joint ``(input, output)`` generator with
:meth:`~EpsilonTransducer.from_joint_generator`, or reconstructed from paired
sample sequences with :meth:`~EpsilonTransducer.from_paired_sequences` (the
transCSSR algorithm, :doc:`epsilon_transducer_inference`).

Channel measures
================

Structural quantities of a channel are defined relative to a driving input
process :cite:`Barnett2015`. Each measure drives the transducer with a supplied
input generator, forms the joint ``(input, output)`` process, and reads off the
quantity, reusing the directional-flow estimators of :doc:`directional_flow`.

* :func:`channel_statistical_complexity` — ``H[S]`` under the stationary
  causal-state law induced by the input.
* :func:`driven_entropy_rate` — entropy rate of the induced output process.
* :func:`directed_information` / :func:`transfer_entropy` — input-to-output
  information flow of the driven joint process.

API
===

.. autoclass:: EpsilonTransducer
   :members: from_channel, from_iohmm, from_joint_generator, from_paired_sequences, is_unifilar, causal_states, statistical_complexity, driven_entropy_rate, directed_information, transfer_entropy, from_wfst

.. currentmodule:: sofic.generators.channel_measures

.. autofunction:: channel_statistical_complexity
.. autofunction:: driven_entropy_rate
.. autofunction:: directed_information
.. autofunction:: transfer_entropy
.. autofunction:: driven_joint_generator
