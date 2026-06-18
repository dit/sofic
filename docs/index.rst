.. pensive documentation master file
.. py:module:: pensive

******************************************************************
:mod:`pensive`: stochastic symbol generators
******************************************************************

:mod:`pensive` is a Python package for hidden Markov models, symbolic dynamics,
finite state machines, and other stochastic symbol generators.

Introduction
------------

Many natural and engineered processes produce sequences of symbols whose
statistics are governed by latent structure: hidden states, transition rules,
or algebraic constraints on allowed paths. ``pensive`` collects algorithms and
data structures for representing, simulating, and analyzing such generators in
a consistent, composable Python API built on NumPy, SciPy, and NetworkX.

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
   viz
   examples
   dit_bridge
   zreferences

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
