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
