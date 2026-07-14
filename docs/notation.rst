.. notation.rst

********
Notation
********

``pensive`` is a scientific tool, and much of this documentation uses
mathematical expressions for computational mechanics and information theory.

Graph-backed models
===================

Every model in ``pensive`` is a :class:`~pensive.core.StateMachine`: a labeled
directed multigraph with typed node and edge attributes.

* **States** are hashable labels (strings, integers, tuples, etc.).
* **Transitions** are directed edges with attribute dictionaries.
* **Alphabets** depend on model type: input symbols, emissions, or outputs.

Edge attribute keys (from :mod:`pensive.core`) include:

* ``ATTR_SYMBOL`` — input symbol on automaton edges
* ``ATTR_EMISSION`` — emitted symbol on generator edges
* ``ATTR_PROB`` — transition probability
* ``ATTR_QUASIPROB`` — signed quasi-probability (n-machines)
* ``ATTR_OUTPUT`` — transducer output symbol
* ``EPSILON`` — sentinel for ε-transitions (NFAs)

Computational mechanics
=======================

For a stationary stochastic process with causal states :math:`S^+` and
retrodictive states :math:`S^-` :cite:`Crutchfield1994`:

* :math:`h_\mu` — entropy rate
* :math:`C_\mu` — statistical complexity :math:`\H{S^+}`
* :math:`E` — excess entropy :math:`\I{S^+ : S^-}`
* :math:`\chi` — crypticity :math:`C_\pm - E`
* :math:`C_\pm` — bidirectional statistical complexity
* :math:`\rho_\mu` — predicted information rate
* :math:`b_\mu` — bound information rate
* :math:`r_\mu` — ephemeral information rate
* :math:`R` — Markov order (topological synchronization)
* :math:`k_\chi` — cryptic order (retrodiction depth)

Information anatomy satisfies :math:`h_\mu = b_\mu + r_\mu`
:cite:`James2013`.

Generative complexity and common information
============================================

Alongside the *predictive* complexity :math:`C_\mu = \H{S^+}`, ``pensive``
computes *generative* complexities: the minimal state entropy of a (possibly
non-unifilar) generator of the process. Each corresponds to a common
information between the forward and reverse causal states :math:`S^+` and
:math:`S^-`:

* :math:`C_g` — generative complexity, the state entropy :math:`\H{G}` of a
  generative presentation
* exact common information (minimal generative model) :cite:`Kumar2014`
* Wyner common information :cite:`Wyner1975`
* Gács–Körner common information :cite:`GacsKorner1973`
* functional common information

See :doc:`generators/generative_models`.

Directional information flow
============================

For bivariate generators emitting paired symbols :math:`(X, Y)`, ``pensive``
computes transfer entropy :cite:`Schreiber2000`, directed information, and the
intrinsic/shared/synergistic decomposition of information flow. See
:doc:`generators/directional_flow`.

Information-theoretic quantities require ``dit`` :cite:`James2018`, which is a
core dependency of ``pensive``.

Unifilarity
===========

The word **unifilar** appears in two contexts:

* :class:`~pensive.automata.unifilar.UnifilarAutomaton` — right-resolving on
  **input symbols** (formal language theory).
* :class:`~pensive.generators.epsilon_machine.EpsilonMachine` — row-unifilar on
  **emissions** (computational mechanics / causal states).

These are distinct predicates; an ε-machine is unifilar in the generator sense.
