.. generators.rst

**********
Generators
**********

The :mod:`sofic.generators` package provides stochastic and quasiprobabilistic
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
   epsilon_transducer
   block_convergence
   bidirectional_epsilon_machine
   mixed_state_presentation
   nonunifilar_entropy_rate
   edge_machine
   information_anatomy
   relative_entropy_rate
   support
   conditioning
   omega_probability
   renyi
   canonical_residual_hmm
   correlations
   predictive_rate_distortion
   symbolic_hmm
   generative_models
   directional_flow
   alternative_complexity
   wheeler_epsilon

Constructions and conversions
=============================

.. toctree::
   :maxdepth: 1

   constructions
   conversions
   lumping
   bisimulation
   factor_codes
   state_splitting

Advanced
========

.. toctree::
   :maxdepth: 1

   nmachine
   quasi_realization
   minimal_quasi_realization
   synchronization
   topological_epsilon_enumeration
