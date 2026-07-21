"""Five-variable information-anatomy I-diagram over ``Pr(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)``.

The signed I-measure (:cite:`yeung1991new`) of the five bidirectional random
variables has ``2⁵ − 1 = 31`` atoms.  Each atom is the *conditional
co-information* of the variables that are "inside" the region given the ones
that are "outside".  For a nonempty subset ``S`` of the five random variables,

.. math::

    a_S = I\\!\\left(X_i : i \\in S \\;\\middle|\\; X_j : j \\notin S\\right),

which reduces to a conditional entropy when ``|S| = 1`` and to a (possibly
negative) conditional co-information otherwise.

Unifilarity forces ten of the 31 atoms to vanish identically.  Of the remaining
twenty-one *generically nonzero* atoms, fourteen are the named taxonomy atoms of
:cite:`jurgens2026taxonomy` Table II; the other seven are cancelling partners of
shielded four-variable zeros (Theorem A / A′ and the classical ``q_μ`` refinement)
that Table II omits.

:func:`information_diagram` computes every atom, classifies it into an anatomy
role, attaches the Jurgens taxonomy label when one exists, and reports named
totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

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
    "b_plus",  # b⁺μ: present ∩ future ∖ past (forward binding zone)
    "b_minus",  # b⁻μ: present ∩ past ∖ future (reverse binding zone)
    "q_mu",  # classical qμ = I[S⁺₀:X₀:S⁻₁] (present ∩ past ∩ future)
    "sigma_mu",  # σμ: past↔future information bypassing the present
    "chi_plus",  # χ⁺ crypticity atoms (t.χ⁺ / p.χ⁺)
    "chi_minus",  # χ⁻ crypticity atoms (t.χ⁻ / p.χ⁻)
    "structure",  # leftover state-structure (always zero under unifilarity)
)

#: Colour / legend group for each fine role (several roles share a colour).
COLOR_GROUP: dict[str, str] = {
    "r_gauge": "r_mu",
    "r_fwd": "r_mu",
    "r_rev": "r_mu",
    "r_joint": "r_mu",
    "b_plus": "b_plus",
    "b_minus": "b_minus",
    "q_mu": "q_mu",
    "sigma_mu": "sigma_mu",
    "chi_plus": "chi_plus",
    "chi_minus": "chi_minus",
    "structure": "structure",
}

#: Jurgens & Crutchfield (2026) Table II labels by Yeung-atom membership.
JURGENS_LABELS: dict[tuple[int, ...], str] = {
    (2,): "t.rμ",
    (1, 2): "p.r⁻μ",
    (2, 3): "p.r⁺μ",
    (1, 2, 3): "p.r±μ",
    (2, 3, 4): "t.b⁺μ",
    (1, 2, 3, 4): "p.b⁺μ",
    (0, 1, 2): "t.b⁻μ",
    (0, 1, 2, 3): "p.b⁻μ",
    (0, 1, 2, 3, 4): "qμ",
    (0, 1, 3, 4): "σμ",
    (0,): "t.χ⁺",
    (0, 3): "p.χ⁺",
    (4,): "t.χ⁻",
    (1, 4): "p.χ⁻",
}

#: Labels for the seven generically-nonzero atoms omitted from Table II.
#: These are the cancelling partners of shielded four-variable zeros.
EXTRA_LABELS: dict[tuple[int, ...], str] = {
    (2, 4): "†b⁺μ gauge",
    (1, 2, 4): "†b⁺μ rev",
    (0, 2): "†b⁻μ gauge",
    (0, 2, 3): "†b⁻μ fwd",
    (0, 2, 4): "†qμ gauge",
    (0, 1, 2, 4): "†qμ rev",
    (0, 2, 3, 4): "†qμ fwd",
}

#: The 21 Yeung atoms that are not forced to zero by unifilarity / shielding
#: (Table II's fourteen plus the seven cancelling extras).  Verified across the
#: structural-ephemeral zoo.
GENERICALLY_NONZERO: frozenset[tuple[int, ...]] = frozenset(
    {
        *JURGENS_LABELS,
        *EXTRA_LABELS,
    }
)

#: Zone symbol for plot / legend naming.
_ZONE_SYMBOL: dict[str, str | None] = {
    "r_gauge": "rμ",
    "r_fwd": "rμ",
    "r_rev": "rμ",
    "r_joint": "rμ",
    "b_plus": "b⁺μ",
    "b_minus": "b⁻μ",
    "q_mu": "qμ",
    "sigma_mu": "σμ",
    "chi_plus": "χ⁺",
    "chi_minus": "χ⁻",
    "structure": None,
}

#: Maps each role to the ``totals`` key holding its aggregate value.
ROLE_TOTAL_KEY: dict[str, str] = {
    "r_gauge": "r_gauge",
    "r_fwd": "r_fwd",
    "r_rev": "r_rev",
    "r_joint": "r_joint",
    "b_plus": "b_plus",
    "b_minus": "b_minus",
    "q_mu": "q_mu",
    "sigma_mu": "sigma_mu",
    "chi_plus": "chi_plus",
    "chi_minus": "chi_minus",
    "structure": "structure",
}


def _require_dit() -> Any:
    from sofic.generators.measures import require_dit

    return require_dit("information diagrams")


def _branch_name(indices: tuple[int, ...]) -> str:
    next_fwd = _S_PLUS_1 in indices
    prev_rev = _S_MINUS_0 in indices
    if next_fwd and prev_rev:
        return "joint"
    if next_fwd:
        return "fwd"
    if prev_rev:
        return "rev"
    return "gauge"


def _classify(indices: tuple[int, ...]) -> str:
    """Assign an I-diagram atom to exactly one anatomy role by its membership."""
    inside = set(indices)
    present = _X_0 in inside
    past = _S_PLUS_0 in inside
    future = _S_MINUS_1 in inside
    if present:
        if past and future:
            return "q_mu"
        if future:
            return "b_plus"
        if past:
            return "b_minus"
        branch = _branch_name(indices)
        return {
            "gauge": "r_gauge",
            "fwd": "r_fwd",
            "rev": "r_rev",
            "joint": "r_joint",
        }[branch]
    if past and future:
        return "sigma_mu"
    if past:
        return "chi_plus"
    if future:
        return "chi_minus"
    return "structure"


def _atom_symbol(indices: tuple[int, ...]) -> str | None:
    """A ``{zone} {branch}`` anatomy name, e.g. ``rμ gauge`` or ``χ⁺ transient``."""
    role = _classify(indices)
    zone = _ZONE_SYMBOL[role]
    if zone is None:
        return None
    if role in ("chi_plus", "chi_minus"):
        # Crypticity: transient = singleton state, persistent = correlated with next.
        persistent = (_S_PLUS_1 in indices) if role == "chi_plus" else (_S_MINUS_0 in indices)
        return f"{zone} {'persistent' if persistent else 'transient'}"
    if role == "sigma_mu":
        return f"{zone} {_branch_name(indices)}"
    return f"{zone} {_branch_name(indices)}"


def _taxonomy_label(indices: tuple[int, ...]) -> str | None:
    """Jurgens Table II label, or a †-marked extra label; ``None`` if always zero."""
    if indices in JURGENS_LABELS:
        return JURGENS_LABELS[indices]
    if indices in EXTRA_LABELS:
        return EXTRA_LABELS[indices]
    return None


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
        """A ``{zone} {branch}`` anatomy name (e.g. ``rμ gauge``, ``χ⁺ transient``)."""
        return _atom_symbol(self.indices)

    @property
    def jurgens_label(self) -> str | None:
        """Taxonomy label from :cite:`jurgens2026taxonomy` Table II, or a
        ``†``-marked extra for the seven cancelling partners omitted there."""
        return _taxonomy_label(self.indices)

    @property
    def color_group(self) -> str:
        """Coarse colour group (``r_mu``, ``b_plus``, ``b_minus``, …)."""
        return COLOR_GROUP[self.role]

    def value_float(self) -> float:
        """The atom value as a float (raises for symbolic values)."""
        return float(self.value)


@dataclass
class InformationDiagram:
    """The full five-variable information diagram with anatomy roles and totals.

    Attributes:
        atoms: Retained atoms in fixed role order (see ``atoms`` argument of
            :func:`information_diagram`).
        variable_names: The five variable display names.
        totals: Named anatomy quantities and per-role sums.
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
    b_plus = role_sum["b_plus"]
    b_minus = role_sum["b_minus"]
    q_mu = role_sum["q_mu"]
    totals: dict[str, Any] = dict(role_sum)
    totals.update(
        {
            "r_mu": r_mu,
            "b_mu": b_plus,  # stationary: b⁺μ = b⁻μ = b_μ
            "b_plus": b_plus,
            "b_minus": b_minus,
            "q_mu": q_mu,
            "sigma_mu": role_sum["sigma_mu"],
            "chi_plus": role_sum["chi_plus"],
            "chi_minus": role_sum["chi_minus"],
            # Classical predicted information I[X₀:S⁺₀] = b⁻ zone + qμ zone.
            "rho_mu": b_minus + q_mu,
            "h_mu": r_mu + b_plus,
            "H[X0]": r_mu + b_plus + b_minus + q_mu,
            "h_imc": b_plus + role_sum["r_fwd"] + role_sum["r_joint"],
            "h_imc_reverse": b_plus + role_sum["r_rev"] + role_sum["r_joint"],
        }
    )
    return totals


def information_diagram(
    source: Any,
    *,
    show_zero: bool = False,
    atoms: Literal["process", "generic", "all"] | None = None,
    tol: float = 1e-9,
) -> InformationDiagram:
    """Compute the five-variable information-anatomy I-diagram.

    Args:
        source: A :class:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine`,
            an :class:`~sofic.generators.epsilon_machine.EpsilonMachine` (converted
            via ``to_bidirectional``), or a five-variable ``dit`` distribution
            whose random variables are ordered ``(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)``.
        show_zero: Deprecated alias — if ``True`` and ``atoms`` is omitted,
            keep every atom (``atoms="all"``). Prefer the ``atoms`` argument.
        atoms: Which atoms to retain:

            * ``"process"`` (default) — only numerically nonzero atoms for this
              process;
            * ``"generic"`` — the 21 generically nonzero membership sets
              (Table II's 14 plus the 7 cancelling extras), including zeros for
              this process;
            * ``"all"`` — all 31 nonempty Yeung atoms.
        tol: Magnitude below which a numeric atom is treated as zero.

    Returns:
        An :class:`InformationDiagram` with the retained atoms in fixed
        :data:`ROLE_ORDER` layout and the named anatomy ``totals``.
    """
    _require_dit()
    from dit.multivariate import coinformation

    if atoms is None:
        atoms = "all" if show_zero else "process"
    if atoms not in ("process", "generic", "all"):
        raise ValueError(f"atoms must be 'process', 'generic', or 'all'; got {atoms!r}")

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

    retained: list[IDiagramAtom] = []
    for subset, value in records:
        if atoms == "process" and _is_zero(value, tol):
            continue
        if atoms == "generic" and subset not in GENERICALLY_NONZERO:
            continue
        retained.append(
            IDiagramAtom(
                indices=subset,
                variables=tuple(VARIABLE_NAMES[i] for i in subset),
                value=value,
                role=_classify(subset),
            )
        )

    return InformationDiagram(
        atoms=retained,
        variable_names=VARIABLE_NAMES,
        totals=_compute_totals(records),
    )
