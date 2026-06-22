.. examples.rst
.. py:module:: pensive.examples.epsilon_machines

********
Examples
********

The :mod:`pensive.examples` module is a catalog of canonical ε-machines and
related generators from the computational mechanics literature.

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
* :func:`butterfly_process`, :func:`nemo_process`
* :func:`ellison_fig9_forward`, :func:`ellison_fig9_reverse` — Ellison et al. :cite:`Ellison2011`
* :func:`ellison_fig15_bidirectional` — bidirectional presentation

Tent map (Misiurewicz point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`tent_map_misiurewicz_hmm` — non-unifilar HMM
* :func:`tent_map_misiurewicz_forward`, :func:`tent_map_misiurewicz_reverse`
* :func:`tent_map_misiurewicz_bidirectional` — information anatomy reference
* :func:`tent_map_misiurewicz_information_expected` — expected measure dict

Example
-------

.. ipython::

   In [1]: from pensive.examples import golden_mean, even_process, bernoulli

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
