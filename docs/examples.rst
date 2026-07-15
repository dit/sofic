.. examples.rst
.. py:module:: sofic.examples

********
Examples
********

The :mod:`sofic.examples` module is a catalog of canonical ε-machines,
symbolic shifts, and related generators from the computational mechanics and
symbolic dynamics literature.

Catalog
-------

Bernoulli and coin processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`bernoulli` — biased i.i.d. coin
* :func:`fair_coin` — fair coin
* :func:`alternating_biased_coins` — alternating bias pattern

Golden mean family
~~~~~~~~~~~~~~~~~~

* :func:`golden_mean` — forbid-``11`` shift convention (not the paper's forbid-``00``)
* :func:`golden_mean_forward`, :func:`golden_mean_reverse` — directional presentations
* :func:`golden_mean_bidirectional` — bidirectional ε-machine (Ellison et al. Fig. 4)
* :func:`golden_mean_markov`, :func:`golden_mean_shift_parry` — Markov and Parry variants
* :func:`restricted_golden_mean` — restricted variant

Other literature processes
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`even_process` — even parity process
* :func:`noisy_random_phase_slip` — NRPS prototype (James et al. :cite:`James2011`, Fig.~11c)
* :func:`butterfly_process`, :func:`nemo_process`
* :func:`ellison_fig9_forward`, :func:`ellison_fig9_reverse` — Ellison et al. :cite:`Ellison2011`
* :func:`ellison_fig15_bidirectional` — bidirectional presentation

Tent map (Misiurewicz point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`tent_map_misiurewicz_hmm` — non-unifilar HMM
* :func:`tent_map_misiurewicz_forward`, :func:`tent_map_misiurewicz_reverse`
* :func:`tent_map_misiurewicz_bidirectional` — information anatomy reference
* :func:`tent_map_misiurewicz_information_expected` — expected measure dict

Sofic-Dyck shifts
~~~~~~~~~~~~~~~~~

* :func:`dyck_shift_order` — one-state Dyck shift of order ``k`` :cite:`BealBlockeletDima2015`
* :func:`motzkin_shift` — Motzkin shift from Beal, Blockelet & Dima Fig. 1
* :func:`sofic_dyck_fig1_shift` — two-state sofic-Dyck shift from Fig. 1
* :func:`sofic_dyck_nondeterminizable_shift` — example with no deterministic presentation
* :func:`sofic_dyck_zeta_example_shift` — zeta-function example shift

Anatomy of a Bit prototypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

James et al. :cite:`James2011` use three canonical processes for block-convergence
figures:

* :func:`even_process`
* :func:`golden_mean`
* :func:`noisy_random_phase_slip`

Process library
---------------

In addition to the curated ε-machines above, :mod:`sofic.examples.processes`
ports a large library of parametrized process factories (``GoldenMean``,
``Even``, ``Nemo``, ``IID``, ``Ising``, ``Ehrenfest``, the periodic and
``Misiurewicz`` families, and many more). Each is a function that returns a
generator, defaulting to an :class:`~sofic.generators.epsilon_machine.EpsilonMachine`
but accepting a ``machine_type`` argument:

.. ipython::

   In [1]: from sofic.examples import GoldenMean, Even, Nemo

   In [2]: gm = GoldenMean(bias=0.5)

   @doctest float
   In [3]: gm.entropy_rate()
   Out[3]: 0.6666666666666665

The module also exposes registries — ``processes.process_list`` and
``processes.transducer_list`` — that enumerate every factory, which is handy for
parametrized tests and sweeps:

.. ipython::

   In [4]: from sofic.examples import processes

   In [5]: len(processes.process_list) > 0
   Out[5]: True

A parallel set of transducer factories (``BitFlip``, ``Parity``, ``Delay``,
``BinaryChannel``, …) lives alongside the processes and produces
:class:`~sofic.automata.transducers.MealyMachine` instances.

Symbolic-shift examples (:mod:`sofic.examples.shifts`) provide the
sofic-Dyck factories listed above; access them via
``from sofic.examples import dyck_shift_order`` or the ``shifts`` module.

Example
-------

.. ipython::

   In [1]: from sofic.examples import golden_mean, even_process, bernoulli

   In [2]: for factory in (golden_mean, even_process, bernoulli):
      ...:     model = factory(0.5)
      ...:     model.validate()

   @doctest float
   In [3]: golden_mean(0.5).entropy_rate()
   Out[3]: 0.6666666666666665

API
---

.. autofunction:: bernoulli
.. autofunction:: fair_coin
.. autofunction:: golden_mean
.. autofunction:: golden_mean_forward
.. autofunction:: golden_mean_reverse
.. autofunction:: golden_mean_bidirectional
.. autofunction:: golden_mean_markov
.. autofunction:: golden_mean_shift_parry
.. autofunction:: even_process
.. autofunction:: noisy_random_phase_slip
.. autofunction:: alternating_biased_coins
.. autofunction:: restricted_golden_mean
.. autofunction:: butterfly_process
.. autofunction:: nemo_process
.. autofunction:: ellison_fig9_forward
.. autofunction:: ellison_fig9_reverse
.. autofunction:: ellison_fig15_bidirectional
.. autofunction:: tent_map_misiurewicz_hmm
.. autofunction:: tent_map_misiurewicz_forward
.. autofunction:: tent_map_misiurewicz_reverse
.. autofunction:: tent_map_misiurewicz_bidirectional
.. autofunction:: tent_map_misiurewicz_a
.. autofunction:: tent_map_misiurewicz_information_expected
.. autofunction:: dyck_shift_order
.. autofunction:: motzkin_shift
.. autofunction:: sofic_dyck_fig1_shift
.. autofunction:: sofic_dyck_nondeterminizable_shift
.. autofunction:: sofic_dyck_zeta_example_shift
