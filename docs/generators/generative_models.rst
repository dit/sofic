.. generative_models.rst
.. py:module:: sofic.generators.minimal_generative_model

*****************
Generative Models
*****************

The statistical complexity :math:`C_\mu = \H{S^+}` is the cost of *prediction*:
the memory a unifilar (ε-machine) presentation must carry. A process can often
be *generated* with less memory by a non-unifilar machine. ``sofic`` builds
these minimal generators from a
:class:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine`,
each realizing a different **common information** between the forward causal
state :math:`S^+` and the reverse causal state :math:`S^-`, restricted to
auxiliaries that are the states of a generator (see `Lifted generators`_).

The generative complexity :math:`C_g = \H{G}` of a presentation with generative
states :math:`G` obeys :math:`C_g \le C_\mu`. The three generator
constructions here sit in the order

.. math::

   \op{I}{S^+ : S^-} \;\le\; \text{(Wyner)} \;\le\; \text{(exact)} \;\le\;
   \text{(functional)} \;\le\; \min(C_\mu^+, C_\mu^-),

where the Wyner value is :math:`\op{I}{(S^+, S^-) : G}` and the others are
:math:`\H{G}` (the first two up to optimizer accuracy).

* :func:`minimal_generative_model` — the **exact common information**
  :cite:`Kumar2014`: the minimum-entropy generator state :math:`G` rendering
  :math:`S^+` and :math:`S^-` conditionally independent, found by a
  stochastic optimizer. Its entropy bounds :math:`C_g` from above.
* :func:`wyner_generative_model` — the **Wyner common information**
  :cite:`Wyner1975`, minimizing :math:`\op{I}{(S^+, S^-) : G}` over the same
  generator states.
* :func:`functional_generative_model` — the **functional common information**:
  the smallest *deterministic* function of :math:`(S^+, S^-)` that separates
  the two causal states and is a generator state, found by a partition search.
* :func:`gacs_korner_generative_model` — the **Gács–Körner common information**
  :cite:`GacsKorner1973`: the combinatorial meet :math:`S^+ \wedge S^-`, the
  largest variable that is simultaneously a deterministic function of both.
  This captures only the conserved "core" (phase / ergodic structure) and is
  often trivial for mixing processes.

Each returns a (generally non-unifilar) :class:`~sofic.generators.mealy.MealyHMM`
subclass whose ``generative_complexity()`` is its state entropy. The exact,
Wyner, and functional models reproduce the source process. The Gács–Körner model
does **not**: the meet need not render :math:`S^+` and :math:`S^-` conditionally
independent, so it is the common-information variable presented as an HMM
rather than a generator — ``is_equal_process(source)`` is ``False`` in general
(the golden mean collapses to an i.i.d. coin), and its ``entropy_rate()`` is its
own, not the source's.

.. code-block:: python

   from sofic.examples import golden_mean_bidirectional

   bidir = golden_mean_bidirectional(0.5)

   gk = bidir.gacs_korner_generative_model()   # deterministic, combinatorial
   gk.generative_complexity()                   # H[S+ ∧ S-]

   exact = bidir.minimal_generative_model()     # stochastic optimizer
   exact.generative_complexity()                # <= C_mu

The optimizers start from the corresponding ``dit`` common-information routines
and require ``dit`` (a core dependency). The convenience methods
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.minimal_generative_model`
and friends on an :class:`~sofic.generators.epsilon_machine.EpsilonMachine`
build the bidirectional presentation first.

Lifted generators
=================

The common informations of :math:`S^+` and :math:`S^-` constrain only
:math:`\op{I}{S^+ : S^- \mid G} = 0`. The HMM built from a channel applies
:math:`p(g \mid s^+, s^-)` at both ends of every bidirectional edge (the
*lift*), and that HMM generates the source process, with :math:`G` jointly
distributed with the past and future as the channel prescribes, only if the
lifted state sequence is Markov:

.. math::

   \op{I}{G_0 X_0 : S^-_1 \mid G_1} = 0, \qquad \op{I}{S^+_0 : X_0 G_1 \mid G_0} = 0.

The exact, Wyner, and functional constructors impose both conditions, so every
point they search is a generator state (Ellison et al. :cite:`Ellison2009` for
the bidirectional machine). The exact value, which
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.generative_complexity`
returns, bounds the generative complexity of Löhr & Ay :cite:`LohrAy2009` from
above. ``markov_residual`` reports the largest constraint information (bits) at
the returned channel.

For first-order Markov processes the constrained and unconstrained values
coincide. They differ already for the second-order Markov process in which
every 1 is followed by at least two 0s: its unconstrained exact (and functional)
optimum, of entropy 1.406, is a deterministic variable whose value
:math:`(C, R_{01})` ("the past ends in 10 and the future starts with 01") can
only be preceded by a value that does not know the past ended in 10, so no HMM
has it as its state; the constrained value is 1.5. The stochastic optimizer is
local and seeded (``starts``, ``rng``); on harder processes more starts can
find lower values.

The unconstrained values remain properties of the process — they equal the
common informations of the past and future — and are available as measures:

.. code-block:: python

   bidir.wyner_common_information()   # C[S+ : S-] = C[past : future]
   bidir.exact_common_information()   # G[S+ : S-] = G[past : future]

API
===

Constructors
------------

.. autofunction:: minimal_generative_model
.. autofunction:: wyner_generative_model
.. autofunction:: functional_generative_model
.. autofunction:: gacs_korner_generative_model

Measures
--------

.. autofunction:: wyner_common_information
.. autofunction:: exact_common_information

Models
------

.. autoclass:: MinimalGenerativeModel
   :members: generative_complexity, entropy_rate
.. autoclass:: WynerGenerativeModel
   :members: generative_complexity, entropy_rate
.. autoclass:: FunctionalGenerativeModel
   :members: generative_complexity, entropy_rate
.. autoclass:: GacsKornerGenerativeModel
   :members: generative_complexity, entropy_rate
