=======
sofic
=======

``sofic`` is a Python package for hidden Markov models, symbolic dynamics,
finite state machines, and other stochastic symbol generators.

Basic Information
-----------------

Documentation
~~~~~~~~~~~~~~

https://sofic.readthedocs.io

Repository
~~~~~~~~~~

https://github.com/dit/sofic

Dependencies
~~~~~~~~~~~~

* Python 3.11+
* `networkx <https://networkx.github.io/>`_
* `numpy <http://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `pyyaml <https://pyyaml.org/>`_
* `dit <https://github.com/dit/dit>`_ (information-theoretic measures)

Development
~~~~~~~~~~~

Clone the repository and install development dependencies with ``uv``:

.. code-block:: bash

   git clone https://github.com/dit/sofic.git
   cd sofic
   uv sync --extra dev

Run tests with ``uv run pytest``. See the ``generalinfo`` page in the Sphinx
docs for linting, type checking, and documentation builds.

Introduction
------------

Many natural and engineered processes produce sequences of symbols whose
statistics are governed by latent structure: hidden states, transition rules,
or algebraic constraints on allowed paths. ``sofic`` collects algorithms and
data structures for representing, simulating, and analyzing such generators
behind a single, composable Python API.

Every model is a graph-backed state machine, so the same objects support
construction, validation, simulation, visualization, (de)serialization, and a
large library of structural and information-theoretic measures. The three
main families are:

* **Stochastic generators** (``sofic.generators``) — Markov chains, hidden
  Markov models (Moore and Mealy presentations), ε-machines, probabilistic
  finite automata, mixed-state presentations, and quasiprobabilistic
  generators. These assign probabilities to sequences.
* **Finite automata** (``sofic.automata``) — DFAs, NFAs, transducers
  (Mealy/Moore machines), regular languages, Büchi automata, visibly pushdown
  and nested-word automata, and residual finite-state automata. These recognize
  or transform languages.
* **Symbolic shifts** (``sofic.shifts``) — shifts of finite type, sofic
  shifts, topological Markov chains, and Dyck/sofic-Dyck shifts. These describe
  the *support* (set of allowed sequences) of a process.

The package builds on NumPy, SciPy, and NetworkX and is designed to sit
alongside the `dit <https://github.com/dit/dit>`_ ecosystem for
information-theoretic analysis of the processes these models describe.

Installation
------------

.. code-block:: bash

   pip install sofic

Optional extras:

* ``sofic[viz]`` — Graphviz diagrams in terminals and Jupyter
* ``sofic[bayes]`` — PyMC/ArviZ backends for Bayesian inference
* ``sofic[test]`` — pytest, hypothesis, and graphviz for the test suite
* ``sofic[docs]`` — Sphinx, IPython, and matplotlib for the docs
* ``sofic[dev]`` — linting, type checking, docs, and all of the above

Quickstart
----------

The basic workflow is the same for every model: build (or load) a generator,
call ``validate()``, then compute properties or convert to another
presentation. States can be any hashable object and every model exposes
``states()``, ``transitions()``, ``draw()``, and ``to_yaml()``.

Stochastic generators
~~~~~~~~~~~~~~~~~~~~~~~

**Markov chain** — a visible-state process. Edges carry ``P(target | source)``:

.. code-block:: python

   from sofic import MarkovChain

   mc = MarkovChain(initial_distribution={"sunny": 0.5, "rainy": 0.5})
   mc.add_transition("sunny", "sunny", 0.9)
   mc.add_transition("sunny", "rainy", 0.1)
   mc.add_transition("rainy", "sunny", 0.5)
   mc.add_transition("rainy", "rainy", 0.5)
   mc.validate()

   mc.stationary_distribution()   # array([0.8333, 0.1667])
   mc.entropy_rate()              # 0.5575 bits/symbol
   mc.is_deterministic()          # False

**Moore HMM** — hidden states with an emission law ``P(observation | state)`` on
states and ``P(target | source)`` on edges:

.. code-block:: python

   from sofic import MooreHMM

   moore = MooreHMM(
       observation_alphabet=frozenset({"H", "T"}),
       initial_distribution={0: 0.5, 1: 0.5},
   )
   moore.add_transition(0, 0, 0.5)
   moore.add_transition(0, 1, 0.5)
   moore.add_transition(1, 0, 0.5)
   moore.add_transition(1, 1, 0.5)
   moore.set_emission_distribution(0, {"H": 0.9, "T": 0.1})
   moore.set_emission_distribution(1, {"H": 0.1, "T": 0.9})
   moore.validate()

   moore.stationary_distribution()             # array([0.5, 0.5])
   moore.log_likelihood(["H", "H", "T", "T"])  # -2.7726 (log2)
   observations, hidden = moore.sample(6)      # simulate the process

**Mealy HMM** — hidden states with a joint transition/emission law
``P(target, symbol | source)`` on edges. Here is the golden-mean process
(consecutive ``1``\ s are forbidden):

.. code-block:: python

   from sofic import MealyHMM

   gm = MealyHMM(
       observation_alphabet=frozenset({0, 1}),
       initial_distribution={"A": 2 / 3, "B": 1 / 3},
   )
   gm.add_transition("A", "A", 0, 0.5)   # source, target, symbol, probability
   gm.add_transition("A", "B", 1, 0.5)
   gm.add_transition("B", "A", 0, 1.0)
   gm.validate()

   gm.entropy_rate()             # 0.6667 bits/symbol
   gm.is_unifilar()              # True
   gm.word_probability([1, 0, 1])  # 0.1667

Many canonical models ship in ``sofic.examples``, so the golden mean is
also just ``from sofic.examples import golden_mean; gm = golden_mean(0.5)``.

**ε-machine** — the minimal unifilar (causal-state) presentation of a stationary
process. Build one from any HMM, from an observed sequence, or directly, and
read off computational-mechanics quantities:

.. code-block:: python

   from sofic import EpsilonMachine

   eps = EpsilonMachine.from_hmm(gm)          # minimize an HMM presentation
   # eps = EpsilonMachine.from_sequence(data, method="cssr", Lmax=4)  # infer
   # eps = EpsilonMachine.from_sequence(data, method="spectral", prefix_length=3, rank=2)

   eps.statistical_complexity()   # 0.9183 bits  (C_mu)
   eps.entropy_rate()             # 0.6667 bits/symbol  (h_mu)
   eps.markov_order()             # 1
   eps.cryptic_order()            # 1

Finite automata
~~~~~~~~~~~~~~~

**DFA** — a deterministic recognizer. This one accepts strings with an even
number of ``b``\ s (states are added before their transitions so determinism can
be checked as you go):

.. code-block:: python

   from sofic import DFA

   dfa = DFA(
       input_alphabet=frozenset({"a", "b"}),
       initial_states=frozenset({"even"}),
       accepting_states=frozenset({"even"}),
   )
   dfa.graph.add_state("even")
   dfa.graph.add_state("odd")
   dfa.add_transition("even", "even", "a")
   dfa.add_transition("even", "odd", "b")
   dfa.add_transition("odd", "odd", "a")
   dfa.add_transition("odd", "even", "b")
   dfa.validate()

   dfa.recognizes("abba")   # True  (two b's)
   dfa.recognizes("abbb")   # False (three b's)
   dfa.minimize()           # DFA(2 states, 4 transitions)
   dfa.to_regex()           # 'a*|a*b(?:...)*ba*'

**NFA** — nondeterministic, with first-class ε-transitions. Determinize to a DFA
when you need one:

.. code-block:: python

   from sofic import NFA

   nfa = NFA(
       input_alphabet=frozenset({"a", "b"}),
       initial_states=frozenset({"q0"}),
       accepting_states=frozenset({"q2"}),
   )
   nfa.add_transition("q0", "q0", "a")
   nfa.add_transition("q0", "q0", "b")
   nfa.add_transition("q0", "q1", "a")
   nfa.add_transition("q1", "q2", "b")
   nfa.validate()

   nfa.recognizes("ab")   # True
   dfa = nfa.determinize()

**Transducer (Mealy machine)** — maps input words to output words. This one
inverts bits:

.. code-block:: python

   from sofic import MealyMachine

   inv = MealyMachine(
       input_alphabet=frozenset({"0", "1"}),
       output_alphabet=frozenset({"0", "1"}),
       initial_states=frozenset({"q"}),
   )
   inv.add_transition("q", "q", "0", "1")   # source, target, input, output
   inv.add_transition("q", "q", "1", "0")
   inv.validate()

   inv.transduce("0110")   # {('1', '0', '0', '1')}

All automata also support ``union``, ``intersection``, ``complement``,
``difference``, ``concat``, and ``kleene_star`` (see the
``LabeledAutomaton`` base class).

Symbolic shifts
~~~~~~~~~~~~~~~

**Shift of finite type** — the set of sequences avoiding a finite list of
forbidden words. The golden-mean shift forbids ``11``:

.. code-block:: python

   from sofic import ShiftOfFiniteType

   sft = ShiftOfFiniteType.from_forbidden_words(
       {("1", "1")},
       symbol_alphabet=frozenset({"0", "1"}),
   )
   sft.validate()

   sft.forbidden_words()        # frozenset({('1', '1')})
   sorted(sft.factor_language(3))  # allowed length-3 words
   sft.is_unifilar()            # True (right-resolving presentation)

**Topological Markov chain** — an adjacency-matrix presentation; compute the
topological entropy or extract the measure of maximal entropy:

.. code-block:: python

   import numpy as np
   from sofic import TopologicalMarkovChain

   tmc = TopologicalMarkovChain.from_adjacency(
       np.array([[1, 1], [1, 0]], dtype=float),   # golden-mean adjacency
       symbol_alphabet=frozenset({0, 1}),
   )
   tmc.validate()

   tmc.topological_entropy()   # 0.4812  (ln of the golden ratio)
   tmc.parry_measure()         # MealyHMM at maximal entropy

**Sofic shift** — a labeled-graph presentation (the shift-space analog of an
NFA). This is the even shift (even-length runs of ``0`` between ``1``\ s):

.. code-block:: python

   from sofic import SoficShift

   even = SoficShift(symbol_alphabet=frozenset({0, 1}))
   even.add_transition("even", "even", 0)
   even.add_transition("even", "odd", 1)
   even.add_transition("odd", "even", 1)
   even.validate()

   even.topological_entropy()   # 0.4812

Converting between model types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The presentations are related by a web of conversions. Probabilistic models can
drop their weights to become automata or shifts (keeping only their support);
HMMs can be minimized to ε-machines; automata can be determinized, minimized,
and turned into regular expressions:

.. code-block:: python

   # Between stochastic presentations
   gm.to_mealy()                       # any HMM -> Mealy HMM
   moore.to_mealy()                    # Moore -> Mealy (joint edge law)
   EpsilonMachine.from_hmm(gm)         # HMM -> minimal causal ε-machine
   eps.to_bidirectional()             # ε-machine -> bidirectional presentation
   eps.mixed_state_presentation()      # -> observer belief-state dynamics
   EpsilonMachine.from_time_reversed(eps)  # forward -> reverse ε-machine

   # Stochastic -> topological (drop probabilities, keep the support)
   gm.to_sofic_shift()                 # HMM support as a sofic shift
   gm.to_support_nfa()                 # HMM support as an NFA
   gm.to_support_dfa()                 # HMM support as a (determinized) DFA

   # Between automata
   nfa.determinize()                   # NFA -> DFA
   DFA.from_nfa(nfa)                   # NFA -> DFA (same result)
   dfa.minimize()                      # DFA -> minimal DFA
   dfa.to_regex()                      # automaton -> regular expression

   # Between / out of shifts
   tmc.to_sofic_shift()                # topological Markov chain -> sofic shift
   tmc.parry_measure()                 # shift -> max-entropy MealyHMM
   sofic.parry_measure()               # sofic shift -> max-entropy MealyHMM

Every model can also be round-tripped through YAML:

.. code-block:: python

   text = eps.to_yaml()
   restored = EpsilonMachine.from_yaml(text)
   # or: sofic.model_to_yaml(eps) / sofic.model_from_yaml(text)

Information anatomy (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``dit`` installed, a bidirectional ε-machine exposes the full stored- and
transient-information anatomy of a process (James, Burke & Crutchfield, 2013):

.. code-block:: python

   from sofic.examples import golden_mean_bidirectional, tent_map_misiurewicz_bidirectional

   bidir = golden_mean_bidirectional(0.5)
   bidir.statistical_complexity()   # 1.5850 bits  (C±)
   bidir.excess_entropy()           # 0.2516 bits  (E)
   bidir.crypticity()               # 1.3333 bits  (chi = C± - E)

   tent = tent_map_misiurewicz_bidirectional()
   tent.information_anatomy()  # {'rho_mu', 'bound_mu', 'ephemeral_mu',
                              #  'entropy_rate', 'excess_entropy', 'crypticity'}

License
-------

``sofic`` is distributed under the BSD 3-Clause License; see ``LICENSE.txt``.
