"""Five-variable information-anatomy I-diagram over ``Pr(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)``.

The signed I-measure (:cite:`yeung1991new`) of the five bidirectional random
variables has ``2⁵ − 1 = 31`` atoms.  Each atom is the *conditional
co-information* of the variables that are "inside" the region given the ones
that are "outside", and equals one region of the five-set information diagram.
For a nonempty subset ``S`` of the five random variables,

.. math::

    a_S = I\\!\\left(X_i : i \\in S \\;\\middle|\\; X_j : j \\notin S\\right),

which reduces to a conditional entropy when ``|S| = 1`` and to a (possibly
negative) conditional co-information otherwise.

Because the presentation is unifilar in both time directions
(``S⁺₁ = φ⁺(S⁺₀, X₀)`` and ``S⁻₀ = φ⁻(S⁻₁, X₀)``), the four ephemeral atoms of
the information anatomy (:cite:`James2011`) collapse onto *single* diagram
atoms, so the anatomy can be read straight off the diagram:

* ephemeral ``r_μ = a{X₀} + a{X₀,S⁺₁} + a{S⁻₀,X₀} + a{S⁻₀,X₀,S⁺₁}`` — the
  gauge, forward, reverse and joint branches;
* bound ``b_μ`` — the atoms with ``X₀`` and ``S⁻₁`` inside and ``S⁺₀`` outside;
* elusive ``σ_μ`` — the atoms with ``S⁺₀`` and ``S⁻₁`` inside and ``X₀``
  outside (:cite:`ara2016elusive`);
* predicted ``ρ_μ = I[X₀ : S⁺₀]`` — the atoms with ``X₀`` and ``S⁺₀`` inside.

:func:`information_diagram` computes every atom, classifies it into one of the
:data:`ROLE_ORDER` anatomy roles, lays the atoms out in a fixed
(role-then-cardinality) order, and reports the named anatomy totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

# Anatomy variable ordering; matches ``bidirectional_step_distribution``.
_S_PLUS_0, _S_MINUS_0, _X_0, _S_PLUS_1, _S_MINUS_1 = 0, 1, 2, 3, 4
_ALL_INDICES: tuple[int, ...] = (_S_PLUS_0, _S_MINUS_0, _X_0, _S_PLUS_1, _S_MINUS_1)

#: Display names for the five random variables, indexed as in the step joint.
VARIABLE_NAMES: tuple[str, ...] = ("S⁺₀", "S⁻₀", "X₀", "S⁺₁", "S⁻₁")

#: Fixed anatomy roles, in the order atoms are laid out (never sorted by value).
ROLE_ORDER: tuple[str, ...] = (
    "r_gauge",  # rμ: X₀ only (transient / pure output relabeling)
    "r_fwd",  # rμ: X₀ + next forward state
    "r_rev",  # rμ: X₀ + previous reverse state
    "r_joint",  # rμ: X₀ + both next states
    "bound",  # bμ: shared with the future S⁻₁, not the past S⁺₀
    "predictive",  # present shared with the past S⁺₀, not the future
    "shared",  # co-information of present with both past and future
    "elusive",  # σμ: past↔future information bypassing the present
    "structure",  # causal-state structure not touching the present symbol
)

#: Anatomy-zone symbol per role. Every present-containing atom (and the elusive
#: atoms) carries a ``{zone} {branch}`` name; pure state-structure atoms get none.
#: ``rμ`` ephemeral, ``bμ`` bound, ``ρμ`` predictive (present·past, not future),
#: ``cμ`` co-information core (present·past·future), ``σμ`` elusive.
_ZONE_SYMBOL: dict[str, str | None] = {
    "r_gauge": "rμ",
    "r_fwd": "rμ",
    "r_rev": "rμ",
    "r_joint": "rμ",
    "bound": "bμ",
    "predictive": "ρμ",
    "shared": "cμ",
    "elusive": "σμ",
    "structure": None,
}


def _atom_symbol(indices: tuple[int, ...]) -> str | None:
    """A ``{zone} {branch}`` anatomy name, e.g. ``rμ gauge`` or ``bμ joint``.

    The zone comes from present/past/future membership (the atom's role); the
    branch comes from which next-states are inside — ``gauge`` (neither),
    ``fwd`` (S⁺₁), ``rev`` (S⁻₀), ``joint`` (both) — mirroring the four ephemeral
    atoms. Pure state-structure atoms (present, past and future all absent, or
    only one of past/future) have no anatomy symbol and return ``None``.
    """
    zone = _ZONE_SYMBOL[_classify(indices)]
    if zone is None:
        return None
    next_fwd = _S_PLUS_1 in indices
    prev_rev = _S_MINUS_0 in indices
    branch = "joint" if next_fwd and prev_rev else "fwd" if next_fwd else "rev" if prev_rev else "gauge"
    return f"{zone} {branch}"


#: Maps each role to the ``totals`` key holding its aggregate value.
ROLE_TOTAL_KEY: dict[str, str] = {
    "r_gauge": "r_gauge",
    "r_fwd": "r_fwd",
    "r_rev": "r_rev",
    "r_joint": "r_joint",
    "bound": "b_mu",
    "predictive": "predictive",
    "shared": "shared",
    "elusive": "sigma_mu",
    "structure": "structure",
}


def _require_dit() -> Any:
    from sofic.generators.measures import require_dit

    return require_dit("information diagrams")


def _classify(indices: tuple[int, ...]) -> str:
    """Assign an I-diagram atom to exactly one anatomy role by its membership."""
    inside = set(indices)
    present = _X_0 in inside
    past = _S_PLUS_0 in inside
    future = _S_MINUS_1 in inside
    if present:
        if past and future:
            return "shared"
        if future:
            return "bound"
        if past:
            return "predictive"
        next_fwd = _S_PLUS_1 in inside
        prev_rev = _S_MINUS_0 in inside
        if next_fwd and prev_rev:
            return "r_joint"
        if next_fwd:
            return "r_fwd"
        if prev_rev:
            return "r_rev"
        return "r_gauge"
    if past and future:
        return "elusive"
    return "structure"


def _ordering_key(indices: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
    return (ROLE_ORDER.index(_classify(indices)), len(indices), indices)


def _coerce(dist: Any, value: Any) -> Any:
    """Return a float for numeric distributions, the raw expression if symbolic."""
    if hasattr(dist, "is_symbolic") and dist.is_symbolic():
        return value
    return float(value)


def _is_zero(value: Any, tol: float) -> bool:
    try:
        return abs(float(value)) <= tol
    except (TypeError, ValueError):
        return False  # symbolic expression: keep it


@dataclass(frozen=True)
class IDiagramAtom:
    """One atom of the five-variable information diagram.

    Attributes:
        indices: Random-variable indices that are *inside* the region.
        variables: The matching :data:`VARIABLE_NAMES`.
        value: The atom's I-measure (a conditional co-information); ``float`` for
            numeric machines, a sympy expression for symbolic ones.
        role: One of :data:`ROLE_ORDER`.
    """

    indices: tuple[int, ...]
    variables: tuple[str, ...]
    value: Any
    role: str

    @property
    def label(self) -> str:
        """Set-builder label, e.g. ``{X₀,S⁺₁}``."""
        return "{" + ",".join(self.variables) + "}"

    @property
    def conditional_expression(self) -> str:
        """The atom's I-measure as a conditional co-information, e.g.
        ``I[X₀:S⁺₁|S⁺₀,S⁻₀,S⁻₁]`` (a conditional entropy ``H[…|…]`` when a
        single variable is inside). This is the term's exact name."""
        inside = ":".join(self.variables)
        outside = ",".join(VARIABLE_NAMES[i] for i in _ALL_INDICES if i not in self.indices)
        head = "H" if len(self.indices) == 1 else "I"
        body = inside if not outside else f"{inside}|{outside}"
        return f"{head}[{body}]"

    @property
    def symbol(self) -> str | None:
        """A ``{zone} {branch}`` anatomy name (e.g. ``rμ gauge``, ``bμ joint``)
        for present-containing and elusive atoms; ``None`` for pure
        state-structure atoms. See :func:`_atom_symbol`."""
        return _atom_symbol(self.indices)

    def value_float(self) -> float:
        """The atom value as a float (raises for symbolic values)."""
        return float(self.value)


@dataclass
class InformationDiagram:
    """The full five-variable information diagram with anatomy roles and totals.

    Attributes:
        atoms: Nonzero (or all, when ``show_zero``) atoms in fixed role order.
        variable_names: The five variable display names.
        totals: Named anatomy quantities and per-role sums (see keys below).

    The ``totals`` dictionary carries the per-role sums (``r_gauge``, ``r_fwd``,
    ``r_rev``, ``r_joint``, ``bound``, ``predictive``, ``shared``, ``elusive``,
    ``structure``) plus the named anatomy quantities ``r_mu``, ``b_mu``,
    ``sigma_mu``, ``rho_mu``, ``h_mu``, ``H[X0]`` and the internal
    (causal-state) Markov-chain entropy rates ``h_imc`` (= ``H[S⁺₁|S⁺₀]``) and
    ``h_imc_reverse`` (= ``H[S⁻₀|S⁻₁]``).
    """

    atoms: list[IDiagramAtom]
    variable_names: tuple[str, ...]
    totals: dict[str, Any]

    def by_role(self) -> dict[str, list[IDiagramAtom]]:
        """Group the retained atoms by role, in :data:`ROLE_ORDER`."""
        groups: dict[str, list[IDiagramAtom]] = {role: [] for role in ROLE_ORDER}
        for atom in self.atoms:
            groups[atom.role].append(atom)
        return groups

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        t = self.totals
        return (
            "InformationDiagram("
            f"atoms={len(self.atoms)}, "
            f"h_mu={t['h_mu']:.4f}, r_mu={t['r_mu']:.4f}, "
            f"b_mu={t['b_mu']:.4f}, sigma_mu={t['sigma_mu']:.4f})"
        )


def _resolve_step_distribution(source: Any) -> Any:
    """Extract the five-variable step joint from a machine or pass a distribution."""
    if hasattr(source, "step_distribution"):
        return source.step_distribution()
    if hasattr(source, "to_bidirectional"):
        return source.to_bidirectional().step_distribution()
    return source


def _compute_totals(records: list[tuple[tuple[int, ...], Any]]) -> dict[str, Any]:
    role_sum: dict[str, Any] = dict.fromkeys(ROLE_ORDER, 0)
    for indices, value in records:
        role_sum[_classify(indices)] += value
    r_mu = role_sum["r_gauge"] + role_sum["r_fwd"] + role_sum["r_rev"] + role_sum["r_joint"]
    b_mu = role_sum["bound"]
    totals: dict[str, Any] = dict(role_sum)
    totals.update(
        {
            "r_mu": r_mu,
            "b_mu": b_mu,
            "sigma_mu": role_sum["elusive"],
            "rho_mu": role_sum["predictive"] + role_sum["shared"],
            "h_mu": r_mu + b_mu,
            "H[X0]": r_mu + b_mu + role_sum["predictive"] + role_sum["shared"],
            # Internal (causal-state) Markov-chain entropy rates: H[S⁺₁|S⁺₀] and
            # H[S⁻₀|S⁻₁]. Equal iff r_fwd == r_rev (arrow-of-time symmetry).
            "h_imc": b_mu + role_sum["r_fwd"] + role_sum["r_joint"],
            "h_imc_reverse": b_mu + role_sum["r_rev"] + role_sum["r_joint"],
        }
    )
    return totals


def information_diagram(
    source: Any,
    *,
    show_zero: bool = False,
    tol: float = 1e-9,
) -> InformationDiagram:
    """Compute the five-variable information-anatomy I-diagram.

    Args:
        source: A :class:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine`,
            an :class:`~sofic.generators.epsilon_machine.EpsilonMachine` (converted
            via ``to_bidirectional``), or a five-variable ``dit`` distribution
            whose random variables are ordered ``(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)``.
        show_zero: Keep atoms whose value is (numerically) zero.
        tol: Magnitude below which a numeric atom is treated as zero.

    Returns:
        An :class:`InformationDiagram` with the 31 atoms (filtered to the nonzero
        ones by default) in fixed :data:`ROLE_ORDER` layout and the named anatomy
        ``totals``.

    The atoms are the conditional co-informations of every nonempty subset of the
    five variables (:cite:`yeung1991new`); the anatomy totals are the atom sums
    identified in the module docstring (:cite:`James2011`; :cite:`ara2016elusive`).
    """
    _require_dit()
    from dit.multivariate import coinformation

    dist = _resolve_step_distribution(source)
    length = dist.outcome_length() if hasattr(dist, "outcome_length") else None
    if length is not None and length != len(_ALL_INDICES):
        raise ValueError(
            f"information_diagram expects a 5-variable step distribution "
            f"(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁); got outcome length {length}."
        )

    records: list[tuple[tuple[int, ...], Any]] = []
    for size in range(1, len(_ALL_INDICES) + 1):
        for subset in combinations(_ALL_INDICES, size):
            crvs = [i for i in _ALL_INDICES if i not in subset]
            value = coinformation(dist, rvs=[[i] for i in subset], crvs=crvs)
            records.append((subset, _coerce(dist, value)))

    records.sort(key=lambda record: _ordering_key(record[0]))

    atoms: list[IDiagramAtom] = []
    for subset, value in records:
        if not show_zero and _is_zero(value, tol):
            continue
        atoms.append(
            IDiagramAtom(
                indices=subset,
                variables=tuple(VARIABLE_NAMES[i] for i in subset),
                value=value,
                role=_classify(subset),
            )
        )

    return InformationDiagram(
        atoms=atoms,
        variable_names=VARIABLE_NAMES,
        totals=_compute_totals(records),
    )
