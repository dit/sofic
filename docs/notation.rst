.. notation.rst

********
Notation
********

``pensive`` is a scientific tool, and much of this documentation uses
mathematical expressions for computational mechanics and information theory.

Graph-backed models
===================

Every model in ``pensive`` is a :class:`~pensive.base.StateMachine`: a labeled
directed multigraph with typed node and edge attributes.

* **States** are hashable labels (strings, integers, tuples, etc.).
* **Transitions** are directed edges with attribute dictionaries.
* **Alphabets** depend on model type: input symbols, emissions, or outputs.

Edge attribute keys (from :mod:`pensive.graph`) include:

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

Unifilarity
===========

The word **unifilar** appears in two contexts:

* :class:`~pensive.automata.unifilar.UnifilarAutomaton` — right-resolving on
  **input symbols** (formal language theory).
* :class:`~pensive.generators.epsilon_machine.EpsilonMachine` — row-unifilar on
  **emissions** (computational mechanics / causal states).

These are distinct predicates; an ε-machine is unifilar in the generator sense.
