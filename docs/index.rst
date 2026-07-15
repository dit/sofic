.. sofic documentation master file
.. py:module:: sofic

******************************************************************
:mod:`sofic`: stochastic symbol generators
******************************************************************

:mod:`sofic` is a Python package for hidden Markov models, symbolic dynamics,
finite state machines, and other stochastic symbol generators.

Introduction
------------

Many natural and engineered processes produce sequences of symbols whose
statistics are governed by latent structure: hidden states, transition rules,
or algebraic constraints on allowed paths. ``sofic`` collects algorithms and
data structures for representing, simulating, and analyzing such generators in
a consistent, composable Python API built on NumPy, SciPy, and NetworkX.

Every model is a graph-backed state machine, so the same objects support
construction, validation, simulation, visualization, (de)serialization, and a
large library of structural and information-theoretic measures. The model
families group into three presentations of a process and a set of tools for
inferring them from data:

* **Stochastic generators** (:mod:`sofic.generators`) — Markov chains, hidden
  Markov models (Mealy and Moore presentations), ε-machines and their
  bidirectional/mixed-state relatives, probabilistic finite automata, stack
  HMMs, and signed quasiprobabilistic generators. These assign probabilities to
  sequences and expose the full computational-mechanics toolkit.
* **Finite automata** (:mod:`sofic.automata`) — DFAs, NFAs, transducers,
  regular-language algebra, residual/átomaton canonical forms, Büchi automata,
  and visibly pushdown / nested-word automata. These recognize or transform
  languages.
* **Symbolic shifts** (:mod:`sofic.shifts`) — shifts of finite type, sofic
  shifts, topological Markov chains, Dyck and sofic-Dyck shifts, and their
  covers. These describe the *support* (set of allowed sequences) of a process.
* **Inference** (:mod:`sofic.inference`) — Bayesian structural inference for
  Markov chains, ε-machines, and stack HMMs via exact conjugate Dirichlet
  evidences, with optional PyMC backends.

The package is designed to sit alongside the :mod:`dit` ecosystem
:cite:`James2018` for information-theoretic analysis of the processes these
models describe.

For a quick tour, see the :ref:`Quickstart <quickstart>`.

Contents:

.. toctree::
   :maxdepth: 2

   generalinfo
   notation
   core/core
   generators/generators
   automata/automata
   shifts/shifts
   inference/inference
   viz
   examples
   zreferences

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
