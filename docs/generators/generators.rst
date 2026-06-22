.. generators.rst

**********
Generators
**********

The :mod:`pensive.generators` package provides stochastic and quasiprobabilistic
symbol generators: hidden Markov models, ε-machines, mixed-state presentations,
and related constructions. The stochastic-generator foundations are hidden
Markov models, computational mechanics, and symbolic process presentations
:cite:`Rabiner1989,Crutchfield1994`.

Stochastic models
=================

.. toctree::
   :maxdepth: 1

   hidden_markov_model
   hidden_markov_stack_model
   markov_chain
   probabilistic_finite_automaton

Computational mechanics
=======================

.. toctree::
   :maxdepth: 1

   epsilon_machine
   bidirectional_epsilon_machine
   mixed_state_presentation
   edge_machine
   information_anatomy

Constructions and conversions
=============================

.. toctree::
   :maxdepth: 1

   constructions
   conversions

Inference
=========

.. toctree::
   :maxdepth: 1

   hmm_inference
   epsilon_inference

Advanced
========

.. toctree::
   :maxdepth: 1

   nmachine
   quasi_realization
   synchronization
   topological_epsilon_enumeration
