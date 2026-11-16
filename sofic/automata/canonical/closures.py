r"""Canonical NFAs from closures of the residuals under algebraic operations.

Myers, Adámek, Milius and Urbat :cite:`Myers2014,Myers2015` build canonical NFAs for a regular language
:math:`L` in two steps. First close the residuals (left quotients) :math:`u^{-1}L` under the operations of a
locally finite variety; with the derivatives :math:`K \mapsto a^{-1}K` as transitions this is the minimal
deterministic automaton of :math:`L` whose states form an algebra of that variety
(:func:`minimal_structured_dfa`). Then take a minimal set of generators of that algebra as the states of an
NFA, with :math:`K \xrightarrow{a} K'` whenever :math:`K'` occurs in the decomposition of :math:`a^{-1}K`.
Zetzsche, van Heerdt, Sammartino and Silva :cite:`Zetzsche2021` recast the second step as generators for
algebras over a monad, related by distributive law homomorphisms, so that the algebra of the deterministic
automaton and the side effect of the succinct one may differ:

=========================  ========================  ======================  =====================================
algebra of the DFA         closure of the residuals  states of the NFA       canonical automaton
=========================  ========================  ======================  =====================================
join-semilattice           unions                    join-irreducibles       canonical RFSA (*jiromaton*)
distributive lattice       unions, intersections     join-irreducibles       distromaton (:class:`Distromaton`)
Boolean algebra            Boolean operations        atoms                   átomaton
GF(2) vector space         symmetric difference      a basis                 minimal xor automaton
Boolean algebra as GF(2)   Boolean operations        a basis                 minimal xor-CABA automaton
=========================  ========================  ======================  =====================================

The universal automaton :cite:`LombardySakarovitch2008` uses every intersection of residuals rather than a
minimal generating set (:class:`UniversalAutomaton`).

Exact representation. Every language in these closures is a union of atoms of :math:`L`, and the atoms are in
bijection with the states of the minimal complete DFA :math:`D` of the reversed language :math:`L^R`
:cite:`BrzozowskiTamm2014`: the atom :math:`A_q` is the reverse of the set of words leading to :math:`q`. So a
language :math:`K` of the closure is stored exactly as the set of states :math:`q` with :math:`A_q \subseteq K`.
Writing :math:`\delta`, :math:`F` and :math:`s_0` for the transition function, accepting states and start state
of :math:`D`, a word :math:`w` lies in :math:`A_q` iff :math:`\delta(s_0, w^R) = q`, from which

* the residual :math:`u^{-1}L` is :math:`\{q : \delta(q, u^R) \in F\}`, and :math:`L` itself is :math:`F`;
* the derivative :math:`a^{-1}K` is the preimage :math:`\{q : \delta(q, a) \in K\}`;
* :math:`\varepsilon \in K` iff :math:`s_0 \in K`;
* union, intersection, complement and symmetric difference are the set operations.

Every construction below is exact and determinizes the reversed language once.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from sofic.automata.canonical.residual import ResidualTable, _language_automaton
from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.automata.xor import XorAutomaton
from sofic.states import sequential_labels

if TYPE_CHECKING:
    from sofic.automata.base import LabeledAutomaton
    from sofic.automata.languages.base import RegularLanguage

__all__ = [
    "Distromaton",
    "UniversalAutomaton",
    "Variety",
    "distromaton_from_language",
    "is_intersection_closed",
    "minimal_structured_dfa",
    "minimal_xor_automaton",
    "universal_automaton_from_language",
    "xor_caba_automaton",
]

Variety = Literal["semilattice", "lattice", "boolean", "gf2"]
_Set = frozenset[Hashable]


@dataclass(frozen=True)
class _AtomAlgebra:
    """Languages of the Boolean closure of the residuals of ``L``, as sets of atoms (see the module docstring)."""

    alphabet: tuple[Any, ...]
    atoms: tuple[Hashable, ...]
    start: Hashable
    language: _Set
    delta: dict[tuple[Hashable, Any], Hashable]

    @classmethod
    def of(cls, language: Any) -> _AtomAlgebra:
        aut = language.determinize() if isinstance(language, XorAutomaton) else _language_automaton(language)
        table = ResidualTable.from_automaton(aut.reverse())
        return cls(
            alphabet=table.alphabet,
            atoms=tuple(table._bfs_order()),
            start=table.start,
            language=table.accepting,
            delta=table.delta,
        )

    @property
    def everything(self) -> _Set:
        return frozenset(self.atoms)

    def derivative(self, language: _Set, symbol: Any) -> _Set:
        return frozenset(q for q in self.atoms if self.delta[(q, symbol)] in language)

    def has_epsilon(self, language: _Set) -> bool:
        return self.start in language

    def residuals(self) -> tuple[_Set, ...]:
        """The distinct residuals, in shortlex order of their shortest access words."""
        order = [self.language]
        seen = {self.language}
        for residual in order:
            for symbol in self.alphabet:
                target = self.derivative(residual, symbol)
                if target not in seen:
                    seen.add(target)
                    order.append(target)
        return tuple(order)

    def meets(self) -> tuple[_Set, ...]:
        """Every intersection of residuals, the empty intersection being :math:`\\Sigma^*`."""
        return _close(self.residuals(), frozenset.__and__, start=self.everything)

    def closure(self, variety: Variety) -> tuple[_Set, ...]:
        if variety == "boolean":
            subsets: list[_Set] = [frozenset()]
            for atom in self.atoms:
                subsets += [s | {atom} for s in subsets]
            return tuple(subsets)
        if variety == "gf2":
            return _close(self.residuals(), frozenset.__xor__)
        if variety == "semilattice":
            return _close(self.residuals(), frozenset.__or__)
        if variety == "lattice":
            return _close(self.meets(), frozenset.__or__)
        raise ValueError(f"unknown variety {variety!r}; expected 'semilattice', 'lattice', 'boolean' or 'gf2'")


def _close(generators: Iterable[_Set], operation: Any, start: _Set = frozenset()) -> tuple[_Set, ...]:
    """Every combination of a subfamily of ``generators`` under an associative, commutative ``operation``.

    ``start`` is the combination of the empty subfamily (the identity of ``operation``).
    """
    elements = [start]
    seen = {start}
    for generator in generators:
        for combined in [operation(e, generator) for e in elements]:
            if combined not in seen:
                seen.add(combined)
                elements.append(combined)
    return tuple(elements)


def _nfa_on(cls: type[NFA], algebra: _AtomAlgebra, generators: Sequence[_Set]) -> Any:
    """The succinct NFA of a union-generating set (Proposition 14 of :cite:`Zetzsche2021`).

    Every derivative of a generator, and :math:`L` itself, must be the union of the generators it contains.
    """
    name = dict(zip(generators, sequential_labels(len(generators)), strict=True))
    nfa = cls(
        input_alphabet=frozenset(algebra.alphabet),
        initial_states=frozenset(name[g] for g in generators if g <= algebra.language),
        accepting_states=frozenset(name[g] for g in generators if algebra.has_epsilon(g)),
    )
    for g in generators:
        nfa.graph.add_state(name[g])
    for g in generators:
        for symbol in algebra.alphabet:
            successor = algebra.derivative(g, symbol)
            for h in generators:
                if h <= successor:
                    nfa.add_transition(name[g], name[h], symbol)
    return nfa


class _GF2Basis:
    """Incremental Gaussian elimination over GF(2) on sets of atoms, encoded as bitmasks."""

    def __init__(self, atoms: Sequence[Hashable]) -> None:
        self._bit = {atom: 1 << i for i, atom in enumerate(atoms)}
        self._rows: dict[int, tuple[int, int]] = {}  # leading bit -> (row, mask of the basis vectors it sums)
        self.vectors: list[_Set] = []

    def _reduce(self, vector: _Set) -> tuple[int, int]:
        row = sum(self._bit[q] for q in vector)
        combination = 0
        while row and (lead := row.bit_length() - 1) in self._rows:
            pivot, pivot_combination = self._rows[lead]
            row ^= pivot
            combination ^= pivot_combination
        return row, combination

    def add(self, vector: _Set) -> bool:
        """Append ``vector`` to the basis if it is independent of the vectors so far."""
        row, combination = self._reduce(vector)
        if not row:
            return False
        self._rows[row.bit_length() - 1] = (row, combination ^ (1 << len(self.vectors)))
        self.vectors.append(vector)
        return True

    def coordinates(self, vector: _Set) -> list[int]:
        """Indices of the basis vectors whose symmetric difference is ``vector``."""
        row, combination = self._reduce(vector)
        if row:
            raise ValueError("vector is not in the span of the basis")
        return [i for i in range(len(self.vectors)) if combination >> i & 1]


def _xor_on(algebra: _AtomAlgebra, basis: _GF2Basis) -> XorAutomaton:
    names = sequential_labels(len(basis.vectors))
    xor = XorAutomaton(
        input_alphabet=frozenset(algebra.alphabet),
        initial_states=frozenset(names[i] for i in basis.coordinates(algebra.language)),
        accepting_states=frozenset(n for n, v in zip(names, basis.vectors, strict=True) if algebra.has_epsilon(v)),
    )
    for name in names:
        xor.graph.add_state(name)
    for name, vector in zip(names, basis.vectors, strict=True):
        for symbol in algebra.alphabet:
            for j in basis.coordinates(algebra.derivative(vector, symbol)):
                xor.add_transition(name, names[j], symbol)
    return xor


class Distromaton(NFA):
    r"""Distromaton of a regular language :cite:`Myers2014`.

    Its states are the join-irreducible languages of the distributive lattice generated by the residuals; the
    lattice is ordered by inclusion, which makes the distromaton a nondeterministic automaton over a poset. As an
    NFA it is state-minimal among NFAs whose accepted languages (over all sets of initial states) are closed under
    intersection (:cite:`Myers2014`, Theorem 4.13; :cite:`Zetzsche2021`, Corollary 37).
    """

    @classmethod
    def from_language(cls, language: RegularLanguage | LabeledAutomaton | XorAutomaton, **kwargs: Any) -> Distromaton:
        return distromaton_from_language(language)


class UniversalAutomaton(NFA):
    r"""Universal automaton of a regular language :cite:`LombardySakarovitch2008`.

    Its states are all intersections of residuals (the empty intersection is :math:`\Sigma^*`), with
    :math:`K \xrightarrow{a} K'` iff :math:`K' \subseteq a^{-1}K`. Every NFA for the language whose states are
    all accessible and co-accessible maps into it, so it contains a homomorphic image of every state-minimal NFA.
    """

    @classmethod
    def from_language(
        cls, language: RegularLanguage | LabeledAutomaton | XorAutomaton, **kwargs: Any
    ) -> UniversalAutomaton:
        return universal_automaton_from_language(language)


def distromaton_from_language(language: RegularLanguage | LabeledAutomaton | XorAutomaton) -> Distromaton:
    r"""Build the distromaton of a regular language :cite:`Myers2014,Zetzsche2021`.

    States are the join-irreducibles of the lattice generated by the residuals under union and intersection
    (together with :math:`\emptyset` and :math:`\Sigma^*`): the intersections of residuals that are not unions of
    smaller intersections. A state :math:`K` is initial iff :math:`K \subseteq L`, accepting iff
    :math:`\varepsilon \in K`, and :math:`K \xrightarrow{a} K'` iff :math:`K' \subseteq a^{-1}K`.

    In a finite distributive lattice of sets the join-irreducibles are the smallest members containing each point.
    Here the points are the atoms, so the state of atom :math:`A` is the intersection of the residuals that
    contain it (:math:`\Sigma^*` for the negative atom). There is one state per atom, *including* the negative
    atom when it is non-empty, so the distromaton has as many states as the minimal complete DFA of :math:`L^R`
    (:cite:`Myers2014`, Corollary 3.21) -- one more than :class:`~sofic.automata.canonical.atomaton.Atomaton`,
    which drops the negative atom, whenever that atom is non-empty. The state :math:`\Sigma^*` is then not
    accessible, but it belongs to the canonical automaton and is kept.
    """
    algebra = _AtomAlgebra.of(language)
    residuals = algebra.residuals()
    states: dict[_Set, None] = {}
    for atom in algebra.atoms:
        meet = algebra.everything
        for residual in residuals:
            if atom in residual:
                meet &= residual
        states[meet] = None
    return _nfa_on(Distromaton, algebra, list(states))


def universal_automaton_from_language(
    language: RegularLanguage | LabeledAutomaton | XorAutomaton,
) -> UniversalAutomaton:
    r"""Build the universal automaton of a regular language :cite:`LombardySakarovitch2008`.

    States are all intersections of residuals, as in :cite:`Myers2014`: :math:`K` is initial iff
    :math:`K \subseteq L`, accepting iff :math:`\varepsilon \in K`, and :math:`K \xrightarrow{a} K'` iff
    :math:`K' \subseteq a^{-1}K`. The states correspond to the maximal factorizations :math:`(X, K)` of
    :math:`L`, :math:`XK \subseteq L`. The empty intersection :math:`\Sigma^*` and, when it occurs, :math:`\emptyset`
    are kept, so the result need not be trim. The distromaton is the subautomaton on the join-irreducible
    intersections. The number of intersections can be exponential in the number of residuals.
    """
    algebra = _AtomAlgebra.of(language)
    return _nfa_on(UniversalAutomaton, algebra, algebra.meets())


def minimal_xor_automaton(language: RegularLanguage | LabeledAutomaton | XorAutomaton) -> XorAutomaton:
    r"""Build the minimal xor automaton of a regular language :cite:`VuilleminGama2010,Myers2014`.

    Its states form a basis of the GF(2) vector space spanned by the residuals under symmetric difference, and
    it accepts by parity (see :class:`~sofic.automata.xor.XorAutomaton`). Its size, the rank over GF(2) of the
    Hankel matrix :math:`[uv \in L]_{u, v}`, is minimal among all xor automata for :math:`L` (:cite:`Myers2014`,
    Theorem 4.10; :cite:`Zetzsche2021`, Corollary 35) and at most the size of the minimal DFA.

    The automaton is canonical only up to the choice of basis. Here the basis is chosen greedily among the
    residuals in shortlex order of their access words, so every state accepts a residual and :math:`L` itself is
    the single initial state (unless :math:`L = \emptyset`). This is the GF(2) instance of the minimal linear
    representation that :func:`~sofic.generators.minimal_quasi_realization.minimal_quasi_realization` computes
    over the reals.
    """
    algebra = _AtomAlgebra.of(language)
    basis = _GF2Basis(algebra.atoms)
    for residual in algebra.residuals():
        basis.add(residual)
    return _xor_on(algebra, basis)


def xor_caba_automaton(
    language: RegularLanguage | LabeledAutomaton | XorAutomaton,
    *,
    basis: Literal["residuals", "atoms"] = "residuals",
) -> XorAutomaton:
    r"""Build a minimal xor-CABA automaton of a regular language :cite:`Zetzsche2021`.

    The Boolean closure of the residuals, read as a GF(2) vector space under symmetric difference, has the atoms
    as a basis, so its dimension is the number of atoms (including the negative atom when non-empty). A
    *minimal xor-CABA automaton* is the xor automaton on a basis of that space (:cite:`Zetzsche2021`,
    Definition 28); it is size-minimal among xor automata whose accepted languages have the same GF(2) closure
    as Boolean closure (:cite:`Zetzsche2021`, Corollary 38). It stands to the minimal xor automaton as the
    átomaton stands to the canonical RFSA.

    ``basis`` chooses the basis:

    * ``"residuals"`` (default) extends the basis of :func:`minimal_xor_automaton` by atoms in order. The
      residual span is closed under derivatives, so the states accessible from the initial states are exactly
      the minimal xor automaton.
    * ``"atoms"`` uses the atoms. The result is then the átomaton together with the negative atom: atoms are
      disjoint, so derivatives decompose as in the átomaton, and the átomaton is co-deterministic with a single
      accepting state, so every word has at most one accepting path and parity acceptance agrees with NFA
      acceptance.
    """
    algebra = _AtomAlgebra.of(language)
    gf2 = _GF2Basis(algebra.atoms)
    if basis == "residuals":
        for residual in algebra.residuals():
            gf2.add(residual)
    elif basis != "atoms":
        raise ValueError(f"unknown basis {basis!r}; expected 'residuals' or 'atoms'")
    for atom in algebra.atoms:
        gf2.add(frozenset({atom}))
    return _xor_on(algebra, gf2)


def minimal_structured_dfa(language: RegularLanguage | LabeledAutomaton | XorAutomaton, variety: Variety) -> DFA:
    r"""The minimal deterministic automaton of a language whose states form an algebra in ``variety``.

    Construction 2.15 of :cite:`Myers2014`: the states are the closure of the residuals under the operations of
    the variety -- unions (``"semilattice"``), unions and intersections with :math:`\emptyset` and
    :math:`\Sigma^*` (``"lattice"``), Boolean operations (``"boolean"``), or symmetric difference (``"gf2"``) --
    with :math:`K \xrightarrow{a} a^{-1}K`, initial state :math:`L` and accepting states the languages containing
    :math:`\varepsilon`. It is complete. It is the minimal pointed bialgebra of :cite:`Zetzsche2021`, from which
    the canonical NFAs of the module docstring are read off by choosing generators. States are labeled
    sequentially. The closure can be exponentially larger than the minimal DFA (the Boolean closure always has
    :math:`2^{\#\text{atoms}}` elements).
    """
    algebra = _AtomAlgebra.of(language)
    elements = algebra.closure(variety)
    name = dict(zip(elements, sequential_labels(len(elements)), strict=True))
    dfa = DFA(
        input_alphabet=frozenset(algebra.alphabet),
        initial_states=frozenset({name[algebra.language]}),
        accepting_states=frozenset(name[e] for e in elements if algebra.has_epsilon(e)),
    )
    for element in elements:
        dfa.graph.add_state(name[element])
    for element in elements:
        for symbol in algebra.alphabet:
            dfa.add_transition(name[element], name[algebra.derivative(element, symbol)], symbol)
    return dfa


def is_intersection_closed(language: RegularLanguage | LabeledAutomaton | XorAutomaton) -> bool:
    r"""Whether every intersection of two residuals of the language is a union of residuals :cite:`Myers2014`.

    For such languages the canonical RFSA is a state-minimal NFA, and the distromaton and the átomaton (counting
    the negative atom) have at most one more state (:cite:`Myers2014`, Theorem 4.4 and Corollary 4.6). Every
    singleton :math:`\{w\}` is intersection-closed (:cite:`Myers2014`, Example 4.3), since its distinct residuals
    are singletons or empty and so pairwise disjoint; the remark after Theorem 4.4 there, that :math:`\{aa\}` is not, conflicts with
    that example and with the definition.
    """
    algebra = _AtomAlgebra.of(language)
    residuals = algebra.residuals()
    return all(
        frozenset().union(*(r for r in residuals if r <= left & right)) == left & right
        for i, left in enumerate(residuals)
        for right in residuals[i + 1 :]
    )
