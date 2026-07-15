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
state :math:`S^+` and the reverse causal state :math:`S^-`.

The generative complexity :math:`C_g = \H{G}` of a presentation with generative
states :math:`G` obeys :math:`C_g \le C_\mu`, and the four constructions here
sit in a fixed order:

.. math::

   K[S^+ : S^-] \;\le\; \op{I}{S^+ : S^-} \;\le\;
   \text{(exact)} \;\le\; \text{(Wyner)}, \quad \text{(functional)}

* :func:`minimal_generative_model` — the **exact common information**
  :cite:`Kumar2014`: the minimum-entropy auxiliary :math:`G` rendering
  :math:`S^+` and :math:`S^-` conditionally independent. This gives the
  smallest-state generator and is found by a stochastic optimizer.
* :func:`wyner_generative_model` — the **Wyner common information**
  :cite:`Wyner1975`, minimizing :math:`\op{I}{(S^+, S^-) : G}` subject to
  conditional independence.
* :func:`functional_generative_model` — the **functional common information**:
  the smallest *deterministic* function of :math:`(S^+, S^-)` that separates
  the two causal states.
* :func:`gacs_korner_generative_model` — the **Gács–Körner common information**
  :cite:`GacsKorner1973`: the combinatorial meet :math:`S^+ \wedge S^-`, the
  largest variable that is simultaneously a deterministic function of both.
  This captures only the conserved "core" (phase / ergodic structure) and is
  often trivial for mixing processes.

Each returns a (generally non-unifilar) :class:`~sofic.generators.mealy.MealyHMM`
subclass whose ``generative_complexity()`` is the corresponding common
information, and which reproduces the source process.

.. code-block:: python

   from sofic.examples import golden_mean_bidirectional

   bidir = golden_mean_bidirectional(0.5)

   gk = bidir.gacs_korner_generative_model()   # deterministic, combinatorial
   gk.generative_complexity()                   # H[S+ ∧ S-]

   exact = bidir.minimal_generative_model()     # stochastic optimizer
   exact.generative_complexity()                # <= C_mu

The optimizers mirror the corresponding ``dit`` common-information routines and
require ``dit`` (a core dependency). The convenience methods
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.minimal_generative_model`
and friends on an :class:`~sofic.generators.epsilon_machine.EpsilonMachine`
build the bidirectional presentation first.

API
===

Constructors
------------

.. autofunction:: minimal_generative_model
.. autofunction:: wyner_generative_model
.. autofunction:: functional_generative_model
.. autofunction:: gacs_korner_generative_model

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
