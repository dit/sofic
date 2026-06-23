.. information_anatomy.rst

********************
Information Anatomy
********************

James, Burke & Crutchfield (2013) decompose the process entropy rate into
predicted, bound, and ephemeral components :cite:`James2013`:

.. math::

   h_\mu = b_\mu + r_\mu, \qquad
   \rho_\mu = \I{X_0 : S^+_0}, \qquad
   b_\mu = \H{X_0 \mid S^+_0, S^-_1}, \qquad
   r_\mu = \I{X_0 : S^-_1 \mid S^+_0}

.. ipython::

   In [1]: from pensive.examples import tent_map_misiurewicz_bidirectional

   In [2]: bidir = tent_map_misiurewicz_bidirectional()

   @doctest float
   In [3]: bidir.predicted_information()
   Out[3]: 0.19828318592750582

   @doctest float
   In [4]: bidir.bound_information()
   Out[4]: 0.17491140867742372

   @doctest float
   In [5]: bidir.ephemeral_information()
   Out[5]: 0.6482610470171268

   In [6]: anatomy = bidir.information_anatomy()

API
===

Use :meth:`~pensive.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.information_anatomy`
on bidirectional models. :class:`~pensive.generators.epsilon_machine.EpsilonMachine`
also exposes these quantities by building its bidirectional presentation.

When that construction is unavailable, use
:meth:`~pensive.generators.epsilon_machine.EpsilonMachine.approximate_information_anatomy`
or :meth:`~pensive.generators.epsilon_machine.EpsilonMachine.block_entropy_estimates`.
These finite-block estimates do not replace the exact bidirectional quantities;
they report the current block-length approximation to ``h_mu``, ``E``,
``rho_mu``, ``b_mu``, ``r_mu``, and related convergence curves.
