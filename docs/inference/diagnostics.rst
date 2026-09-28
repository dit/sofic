.. diagnostics.rst
.. py:module:: sofic.inference.diagnostics

*****************************
Reconstruction diagnostics
*****************************

A reconstruction algorithm always returns *some* machine. These tools ask
whether that machine fits the data, and whether its structure is supported by the
data or produced by one particular sample and one setting of the tuning
parameters.

Goodness of fit
===============

:func:`goodness_of_fit` is a parametric bootstrap :cite:`Efron1993`. It
simulates sequences as long as the data from the fitted machine and compares a
length-``L`` word statistic of the data with its distribution over the
simulations. Two statistics are available:

* ``"g"`` — the G statistic of the observed word counts against the machine's
  stationary word probabilities;
* ``"entropy_rate"`` — the gap between the plug-in conditional entropy and the
  machine's.

Because the null distribution is simulated, overlapping windows need no
correction. A small p-value means the machine misses structure. For CSSR that
usually means ``Lmax`` is shorter than the source's synchronization length.

.. code-block:: python

   from sofic.generators.epsilon_inference import cssr
   from sofic.inference.diagnostics import goodness_of_fit

   machine = cssr(data, Lmax=1)
   goodness_of_fit(machine, data, L=6).pvalue   # small for the even process
   machine = cssr(data, Lmax=4)
   goodness_of_fit(machine, data, L=6).pvalue   # large

Observed words the machine forbids are listed in ``forbidden_words``.

Structural stability
====================

:func:`structure_stability` reconstructs from many resamples and counts how
often each topology (compared up to isomorphism by :func:`topology_key`)
reappears. The default ``resample="subsample"`` uses random contiguous segments
:cite:`Politis1999`, which contain no artificial junctions. The stationary
bootstrap (``resample="block"``, :cite:`Politis1994`) joins blocks, which creates
words the source never emits and can add spurious states. For example, on
even-process data it returns 6–12-state machines where subsampling returns the
true 2 states.

:func:`reconstruction_sweep` reconstructs over a grid of ``alpha`` and ``Lmax``.
A structure that persists over a range of settings is better supported than one
that appears at a single setting.

API
===

.. autofunction:: goodness_of_fit

.. autoclass:: GoodnessOfFit

.. autofunction:: structure_stability

.. autoclass:: StructureStability
   :members: reference_fraction, modal_topology, n_resamples

.. autofunction:: reconstruction_sweep

.. autofunction:: topology_key
