``pensive`` is a Python package for hidden Markov models, symbolic dynamics,
finite state machines, and other stochastic symbol generators.

Introduction
------------

Many natural and engineered processes produce sequences of symbols whose
statistics are governed by latent structure: hidden states, transition rules,
or algebraic constraints on allowed paths. ``pensive`` collects algorithms and
data structures for representing, simulating, and analyzing such generators in
a consistent, composable Python API.

The package builds on NumPy, SciPy, and NetworkX and is designed to sit alongside
the `dit <https://github.com/dit/dit>`_ ecosystem for information-theoretic
analysis of the processes these models describe.

Quick start
-----------

Build a bidirectional ε-machine for the golden mean process (Ellison et al.,
arXiv:0905.3587, Fig.~4) and compute information measures with ``dit``::

   from pensive.examples import golden_mean, golden_mean_bidirectional

   eps = golden_mean(0.5)
   dist = eps.state_distribution()
   print("h  =", eps.entropy_rate())
   print("Cμ =", eps.statistical_complexity())

   bidir = golden_mean_bidirectional(0.5)
   # Or: BidirectionalEpsilonMachine.from_forward(golden_mean_forward(0.5))

   print(bidir.draw())  # requires graphviz (``pip install pensive[viz]``)
   print("C± =", bidir.statistical_complexity())
   print("E  =", bidir.excess_entropy())
   print("χ  =", bidir.crypticity())

Topological synchronization orders (James et al., arXiv:1010.5545) depend only on
ε-machine graph structure::

   print("R  =", eps.markov_order())
   print("kχ =", eps.cryptic_order())

``cryptic_order()`` is the integer cryptic order; ``crypticity()`` in
``BidirectionalEpsilonMachine`` is the information measure χ = C± − E.

For exact Fig.~4(c) state labels ``(A, C)``, ``(A, D)``, ``(B, C)``, use
``golden_mean_bidirectional`` or pass ``golden_mean_forward`` and
``golden_mean_reverse`` explicitly to ``from_pair``. The helper
``golden_mean()`` uses the forbid-``11`` shift convention (not the paper's
forbid-``00`` process), though ``from_forward`` still yields a
three-state bidirectional presentation for either convention.

Information anatomy (ρ_μ, b_μ, r_μ)
-----------------------------------

James, Burke & Crutchfield (2013), supplement to *Chaos Forgets and Remembers*,
give a closed-form pipeline for bound and ephemeral information rates from a
bidirectional ε-machine.  With ``dit`` installed::

   from pensive.examples import tent_map_misiurewicz_bidirectional

   bidir = tent_map_misiurewicz_bidirectional()
   print("h_μ =", bidir.entropy_rate())
   print("ρ_μ =", bidir.predicted_information())
   print("r_μ =", bidir.ephemeral_information())
   print("b_μ =", bidir.bound_information())
   print(bidir.information_anatomy())

Edge machines (generator presentations) convert a non-unifilar HMM into a
Mealy generator whose states index labeled transitions::

   from pensive.examples import tent_map_misiurewicz_hmm

   edge = tent_map_misiurewicz_hmm().to_edge_machine()

Optional extras:

* ``pensive[measures]`` — ``dit`` integration for entropy, complexity, crypticity
* ``pensive[viz]`` — Graphviz diagrams in terminals and Jupyter
* ``pensive[test]`` — pytest, hypothesis, and graphviz for the test suite
* ``pensive[dev]`` — linting, type checking, docs, and all of the above

Basic Information
-----------------

Documentation
*************

https://pensive.readthedocs.io

Repository
**********

https://github.com/dit/pensive

Dependencies
************

+-------------------------------------------------------------------+
| * Python 3.11+                                                    |
| * `networkx <https://networkx.github.io/>`_                       |
| * `numpy <http://www.numpy.org/>`_                                |
| * `scipy <https://www.scipy.org/>`_                               |
+-------------------------------------------------------------------+

Development
***********

Clone the repository and install development dependencies with ``uv``::

   git clone https://github.com/dit/pensive.git
   cd pensive
   uv sync --extra dev

Run tests with ``uv run pytest``. See :doc:`generalinfo` in the Sphinx docs
for linting, type checking, and documentation builds.

License
-------

``pensive`` is distributed under the BSD 3-Clause License; see ``LICENSE.txt``.
