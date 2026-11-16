r"""Stochastic analogues of the canonical NFAs of a regular language, as generators of a process.

Regular languages have several canonical automata :cite:`Myers2015`: the minimal DFA, the átomaton
:cite:`BrzozowskiTamm2014`, the canonical residual finite-state automaton (RFSA) :cite:`Denis2002`, and its dual,
the maximized prime átomaton :cite:`MaarandTamm2022`. For a stationary process with a finite bidirectional
ε-machine :cite:`Ellison2009` the first two have standard analogues, and the last two have analogues built from
extreme points of belief polytopes:

* minimal DFA -- the forward ε-machine, states :math:`S^+`;
* átomaton (reverse of the minimal DFA of the reverse language) -- the reverse ε-machine run forward,
  states :math:`S^-` (:func:`atomaton_hmm`);
* canonical RFSA (prime residuals) -- the forward causal states whose future morph is not a mixture of the
  others' (:func:`canonical_rfsa_hmm`);
* maximized prime átomaton (reverse of the canonical RFSA of the reverse language) -- the same construction in
  reverse time, reversed back (:func:`maximized_prime_atomaton_hmm`).

A forward causal state :math:`s^+` has future morph :math:`\sum_b R_{s^+}(b) \Pr(\text{future} \mid S^- = b)`
with :math:`R_{s^+} = \Pr(S^- \mid S^+ = s^+)`, and the futures given distinct reverse causal states have
disjoint supports. So the morph of :math:`s^+` is not a nonnegative combination of the other morphs exactly when
:math:`R_{s^+}` is a vertex of :math:`\operatorname{conv}\{R_{s^+}\}`: this is the convex analogue of a prime
residual, as in probabilistic residual automata :cite:`Esposito2002`. Writing every row as
:math:`R_t = \sum_e C_{te} R_e`, the HMM on the vertices with joint transition matrices :math:`(T^{(x)} C)_{E,
\cdot}` emits the future morph of each vertex, so it generates the process with at most :math:`|S^+|` states. It
is in general not unifilar. The prime-átomaton analogue uses the forward beliefs
:math:`Q_{s^-} = \Pr(S^+ \mid S^- = s^-)` of the reverse causal states instead: it is a generator of the
reversed process, time-reversed with its stationary law, with at most :math:`|S^-|` states.

These are generators, so their state entropies bound the generative complexity :math:`C_g` of
:cite:`LohrAy2009` from above; no minimality is claimed. Identifying them with the canonical NFAs is an analogy
(extreme points of a cone in place of prime residuals) and is not from the literature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sofic.generators.mealy import MealyHMM

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.coupled_generators import _GeneratorProblem

__all__ = ["atomaton_hmm", "canonical_rfsa_hmm", "maximized_prime_atomaton_hmm"]


def _problem(machine: Any) -> _GeneratorProblem:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.coupled_generators import _GeneratorProblem
    from sofic.generators.epsilon_machine import EpsilonMachine

    if not isinstance(machine, BidirectionalEpsilonMachine):
        forward = machine if isinstance(machine, EpsilonMachine) else EpsilonMachine.from_hmm(machine)
        machine = BidirectionalEpsilonMachine.from_forward(forward)
    return _GeneratorProblem(machine)


def _labels(problem: _GeneratorProblem, side: int) -> list[Any]:
    return sorted({pair[side] for pair in problem.lifted.pairs}, key=repr)


def _as_hmm(problem: _GeneratorProblem, pi: np.ndarray, K: np.ndarray, names: list[Any], cutoff: float) -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset(problem.symbols))
    for name in names:
        hmm.graph.add_state(name)
    for w, x, v in zip(*np.nonzero(cutoff < K), strict=True):
        hmm.add_transition(names[w], names[v], problem.symbols[x], float(K[w, x, v]))
    total = float(pi.sum())
    hmm.initial_distribution = {name: float(p) / total for name, p in zip(names, pi, strict=True)}
    hmm.validate_stochastic()
    return hmm


def canonical_rfsa_hmm(
    machine: EpsilonMachine | BidirectionalEpsilonMachine, *, tol: float = 1e-7, cutoff: float = 1e-13
) -> MealyHMM:
    """Canonical-RFSA analogue: the generator on the forward causal states with extreme future morphs.

    States are named by those causal states; the initial distribution is stationary. ``tol`` is the L1
    tolerance of the extreme-point linear programs. See the module docstring.
    """
    problem = _problem(machine)
    (pi, _, _, K), E = problem.residual_seed(tol=tol)
    names = _labels(problem, 0)
    return _as_hmm(problem, pi, K, [names[e] for e in E], cutoff)


def maximized_prime_atomaton_hmm(
    machine: EpsilonMachine | BidirectionalEpsilonMachine, *, tol: float = 1e-7, cutoff: float = 1e-13
) -> MealyHMM:
    """Maximized-prime-átomaton analogue: the generator on the reverse causal states with extreme past morphs.

    States are named by those reverse causal states; the initial distribution is stationary. See the module
    docstring.
    """
    problem = _problem(machine)
    (pi, _, _, K), E = problem.prime_atomaton_seed(tol=tol)
    names = _labels(problem, 1)
    return _as_hmm(problem, pi, K, [names[e] for e in E], cutoff)


def atomaton_hmm(machine: EpsilonMachine | BidirectionalEpsilonMachine, *, cutoff: float = 1e-13) -> MealyHMM:
    """Átomaton analogue: the reverse ε-machine run forward in time (a co-unifilar generator on :math:`S^-`)."""
    problem = _problem(machine)
    pi, _, _, K = problem.reverse_seed()
    return _as_hmm(problem, pi, K, _labels(problem, 1), cutoff)


if TYPE_CHECKING:
    from sofic.generators.epsilon_machine import EpsilonMachine
