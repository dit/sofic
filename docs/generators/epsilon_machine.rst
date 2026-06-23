.. epsilon_machine.rst
.. py:module:: pensive.generators.epsilon_machine

***********
ε-Machine
***********

An :class:`EpsilonMachine` is a unifilar Mealy HMM — the minimal causal
presentation of a stationary process :cite:`Crutchfield1994,Loomis2019`.

.. math::

   h_\mu = \H{X_0 \mid X_{-\infty:0}}, \qquad
   C_\mu = \H{S^+}

Build from an HMM via partition refinement:

.. ipython::

   In [1]: from pensive.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   @doctest float
   In [3]: eps.entropy_rate()
   Out[3]: 0.6666666666666665

   @doctest float
   In [4]: eps.statistical_complexity()
   Out[4]: 0.9182958340544896

   @doctest
   In [5]: eps.markov_order()
   Out[5]: 1

Block entropy diagrams follow the entropy-convergence view of Markov order,
cryptic order, and crypticity :cite:`Mahoney2011`:

.. ipython::

   In [6]: diagram = eps.block_entropy_diagram(max_length=4)

   @doctest
   In [7]: diagram.markov_order
   Out[7]: 1

   In [8]: ax = eps.plot_block_entropy_diagram(max_length=4)

When a bidirectional ε-machine is unavailable or intentionally avoided, finite
block entropies also provide explicit estimates for ``h_mu``, ``E``, ``r_mu``,
``b_mu``, synchronization, crypticity, and related convergence curves:

.. ipython::

   In [9]: estimates = eps.block_entropy_estimates(max_length=4)

   @doctest float
   In [10]: estimates.h_mu
   Out[10]: 0.6666666666666665

   In [11]: estimates.information_anatomy()

See also :doc:`bidirectional_epsilon_machine`, :doc:`information_anatomy`, and
:doc:`epsilon_inference` (sample-based reconstruction).

API
===

.. autoclass:: EpsilonMachine
   :members: from_hmm, from_sequence, from_time_reversed, to_bidirectional, block_entropy_diagram, block_entropy_estimates, plot_block_entropy_diagram, approximate_entropy_rate, approximate_excess_entropy, approximate_information_anatomy, statistical_complexity, bidirectional_statistical_complexity, excess_entropy, predicted_information, bound_information, ephemeral_information, information_anatomy, crypticity, bidirectional_crypticity, markov_order, is_markov, cryptic_order, is_exactly_synchronizable

.. autoclass:: pensive.generators.block_entropy.BlockEntropyDiagram
   :members: plot
.. autoclass:: pensive.generators.block_entropy.BlockEntropyEstimates
   :members: information_anatomy
