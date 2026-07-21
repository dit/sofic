"""Tests for the five-variable information-anatomy I-diagram and its UpSet plot."""

from __future__ import annotations

import pytest

from sofic.examples import (
    NRPS,
    bernoulli,
    butterfly_process,
    even_process,
    golden_mean_forward,
    golden_mean_reverse,
    nemo_process,
)
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.information_diagram import (
    GENERICALLY_NONZERO,
    JURGENS_LABELS,
    ROLE_ORDER,
    IDiagramAtom,
    InformationDiagram,
    information_diagram,
)


def _processes() -> dict[str, BidirectionalEpsilonMachine]:
    """Bidirectional presentations spanning the ephemeral-motif zoo."""
    return {
        "bernoulli_half": bernoulli(0.5).to_bidirectional(),
        "golden_mean": BidirectionalEpsilonMachine.from_pair(
            golden_mean_forward(0.5), golden_mean_reverse(0.5)
        ),
        "even": even_process(0.5).to_bidirectional(),
        "butterfly": butterfly_process().to_bidirectional(),
        "nemo": nemo_process().to_bidirectional(),
        "nrps": NRPS().to_bidirectional(),
    }


_NAMES = ["bernoulli_half", "golden_mean", "even", "butterfly", "nemo", "nrps"]


@pytest.mark.parametrize("name", _NAMES)
def test_ephemeral_atoms_match_anatomy_methods(name: str):
    """The four ephemeral roles equal the anatomy's four-atom partition exactly."""
    pytest.importorskip("dit")
    bidir = _processes()[name]
    totals = information_diagram(bidir).totals

    assert totals["r_gauge"] == pytest.approx(bidir.pure_gauge_information(), abs=1e-9)
    assert totals["r_fwd"] == pytest.approx(bidir.forward_only_structural_ephemeral(), abs=1e-9)
    assert totals["r_rev"] == pytest.approx(bidir.reverse_only_structural_ephemeral(), abs=1e-9)
    assert totals["r_joint"] == pytest.approx(bidir.joint_structural_ephemeral(), abs=1e-9)


@pytest.mark.parametrize("name", _NAMES)
def test_named_totals_match_anatomy(name: str):
    """r_μ, b_μ, ρ_μ and h_μ read off the diagram match the direct anatomy methods."""
    pytest.importorskip("dit")
    bidir = _processes()[name]
    totals = information_diagram(bidir).totals

    assert totals["r_mu"] == pytest.approx(bidir.ephemeral_information(), abs=1e-9)
    assert totals["b_mu"] == pytest.approx(bidir.bound_information(), abs=1e-9)
    assert totals["rho_mu"] == pytest.approx(bidir.predicted_information(), abs=1e-9)
    assert totals["h_mu"] == pytest.approx(bidir.entropy_rate(), abs=1e-9)
    assert totals["h_mu"] == pytest.approx(totals["r_mu"] + totals["b_mu"], abs=1e-12)
    assert totals["h_imc"] == pytest.approx(bidir.internal_markov_entropy_rate(), abs=1e-9)
    assert totals["h_imc_reverse"] == pytest.approx(
        bidir.reverse_internal_markov_entropy_rate(), abs=1e-9
    )


@pytest.mark.parametrize("name", _NAMES)
def test_sigma_mu_is_elusive_information(name: str):
    """The elusive role sums to σ_μ = I[S⁺₀ : S⁻₁ | X₀]."""
    pytest.importorskip("dit")
    from dit.multivariate import coinformation

    bidir = _processes()[name]
    dist = bidir.step_distribution()
    sigma_direct = float(coinformation(dist, rvs=[[0], [4]], crvs=[2]))
    assert information_diagram(bidir).totals["sigma_mu"] == pytest.approx(sigma_direct, abs=1e-9)


@pytest.mark.parametrize("name", _NAMES)
def test_all_atoms_sum_to_joint_entropy(name: str):
    """The 31 signed I-measure atoms partition the joint entropy H[S⁺₀,S⁻₀,X₀,S⁺₁,S⁻₁]."""
    dit = pytest.importorskip("dit")
    bidir = _processes()[name]
    diagram = information_diagram(bidir, show_zero=True)

    assert len(diagram.atoms) == 31  # 2**5 - 1
    total = sum(atom.value_float() for atom in diagram.atoms)
    h_joint = float(dit.shannon.entropy(bidir.step_distribution()))
    assert total == pytest.approx(h_joint, abs=1e-9)


@pytest.mark.parametrize("name", _NAMES)
def test_every_atom_has_exactly_one_valid_role(name: str):
    """Roles partition the diagram: every atom carries one role from ROLE_ORDER."""
    pytest.importorskip("dit")
    diagram = information_diagram(_processes()[name], show_zero=True)
    for atom in diagram.atoms:
        assert atom.role in ROLE_ORDER
        assert atom.variables == tuple(diagram.variable_names[i] for i in atom.indices)
    grouped = sum(len(v) for v in diagram.by_role().values())
    assert grouped == len(diagram.atoms)


def test_fixed_order_is_deterministic_and_role_grouped():
    """Atoms are laid out in a stable, role-then-cardinality order (never by value)."""
    pytest.importorskip("dit")
    bidir = _processes()["nemo"]
    first = [atom.indices for atom in information_diagram(bidir, show_zero=True).atoms]
    second = [atom.indices for atom in information_diagram(bidir, show_zero=True).atoms]
    assert first == second

    roles = [atom.role for atom in information_diagram(bidir, show_zero=True).atoms]
    rank = [ROLE_ORDER.index(role) for role in roles]
    assert rank == sorted(rank)  # roles appear as contiguous blocks in ROLE_ORDER


def test_ephemeral_role_atoms_are_the_expected_single_atoms():
    """Under unifilarity the four ephemeral atoms are single I-diagram atoms."""
    pytest.importorskip("dit")
    diagram = information_diagram(_processes()["nemo"], show_zero=True)
    by_index = {atom.indices: atom.role for atom in diagram.atoms}
    assert by_index[(2,)] == "r_gauge"  # {X₀}
    assert by_index[(2, 3)] == "r_fwd"  # {X₀, S⁺₁}
    assert by_index[(1, 2)] == "r_rev"  # {S⁻₀, X₀}
    assert by_index[(1, 2, 3)] == "r_joint"  # {S⁻₀, X₀, S⁺₁}


def test_bernoulli_ephemeral_is_pure_gauge_atom():
    """A fair coin's whole entropy rate is the single X₀-only gauge atom."""
    pytest.importorskip("dit")
    diagram = information_diagram(bernoulli(0.5).to_bidirectional())
    assert diagram.totals["r_gauge"] == pytest.approx(1.0, abs=1e-9)
    assert diagram.totals["r_mu"] == pytest.approx(1.0, abs=1e-9)
    assert diagram.totals["b_mu"] == pytest.approx(0.0, abs=1e-9)
    gauge = [a for a in diagram.atoms if a.role == "r_gauge"]
    assert len(gauge) == 1 and gauge[0].indices == (2,)
    assert gauge[0].label == "{X₀}"


def test_epsilon_machine_delegates_to_bidirectional():
    """EpsilonMachine.information_diagram matches the bidirectional computation."""
    pytest.importorskip("dit")
    forward = golden_mean_forward(0.5)
    from_forward = forward.information_diagram()
    from_bidir = forward.to_bidirectional().information_diagram()
    assert isinstance(from_forward, InformationDiagram)
    assert [a.indices for a in from_forward.atoms] == [a.indices for a in from_bidir.atoms]
    assert from_forward.totals["r_mu"] == pytest.approx(from_bidir.totals["r_mu"], abs=1e-12)


def test_information_diagram_rejects_wrong_arity():
    """A non-five-variable distribution is rejected with a clear error."""
    dit = pytest.importorskip("dit")
    two_var = dit.Distribution(["00", "11"], [0.5, 0.5])
    with pytest.raises(ValueError, match="5-variable"):
        information_diagram(two_var)


def test_atom_value_float_and_label():
    pytest.importorskip("dit")
    atom = IDiagramAtom(indices=(1, 2), variables=("S⁻₀", "X₀"), value=0.5, role="r_rev")
    assert atom.value_float() == pytest.approx(0.5)
    assert atom.label == "{S⁻₀,X₀}"


def test_atom_conditional_expression_and_symbol():
    """Every atom is named by its conditional co-information; anatomy atoms also
    carry a ``{zone} {branch}`` symbol."""
    gauge = IDiagramAtom(indices=(2,), variables=("X₀",), value=0.1, role="r_gauge")
    assert gauge.conditional_expression == "H[X₀|S⁺₀,S⁻₀,S⁺₁,S⁻₁]"
    assert gauge.symbol == "rμ gauge"

    joint = IDiagramAtom(indices=(1, 2, 3), variables=("S⁻₀", "X₀", "S⁺₁"), value=0.2, role="r_joint")
    assert joint.conditional_expression == "I[S⁻₀:X₀:S⁺₁|S⁺₀,S⁻₁]"
    assert joint.symbol == "rμ joint"

    bound = IDiagramAtom(indices=(2, 4), variables=("X₀", "S⁻₁"), value=0.1, role="b_plus")
    assert bound.conditional_expression == "I[X₀:S⁻₁|S⁺₀,S⁻₀,S⁺₁]"
    assert bound.symbol == "b⁺μ gauge"
    assert bound.jurgens_label == "†b⁺μ gauge"

    full = IDiagramAtom(
        indices=(0, 1, 2, 3, 4),
        variables=("S⁺₀", "S⁻₀", "X₀", "S⁺₁", "S⁻₁"),
        value=0.0,
        role="q_mu",
    )
    assert full.conditional_expression == "I[S⁺₀:S⁻₀:X₀:S⁺₁:S⁻₁]"
    assert full.symbol == "qμ joint"
    assert full.jurgens_label == "qμ"

    chi = IDiagramAtom(indices=(0,), variables=("S⁺₀",), value=0.3, role="chi_plus")
    assert chi.symbol == "χ⁺ transient"
    assert chi.jurgens_label == "t.χ⁺"


def test_zone_branch_symbols_cover_every_anatomy_atom():
    """Each anatomy atom gets a ``{zone} {branch}`` name; leftover structure none."""
    cases = {
        (2,): "rμ gauge",
        (2, 3): "rμ fwd",
        (1, 2): "rμ rev",
        (1, 2, 3): "rμ joint",
        (2, 4): "b⁺μ gauge",
        (2, 3, 4): "b⁺μ fwd",
        (1, 2, 4): "b⁺μ rev",
        (1, 2, 3, 4): "b⁺μ joint",
        (0, 2): "b⁻μ gauge",
        (0, 2, 3): "b⁻μ fwd",
        (0, 1, 2): "b⁻μ rev",
        (0, 1, 2, 3): "b⁻μ joint",
        (0, 2, 4): "qμ gauge",
        (0, 2, 3, 4): "qμ fwd",
        (0, 1, 2, 4): "qμ rev",
        (0, 1, 2, 3, 4): "qμ joint",
        (0, 4): "σμ gauge",
        (0, 3, 4): "σμ fwd",
        (0, 1, 4): "σμ rev",
        (0, 1, 3, 4): "σμ joint",
        (0,): "χ⁺ transient",
        (0, 3): "χ⁺ persistent",
        (4,): "χ⁻ transient",
        (1, 4): "χ⁻ persistent",
    }
    for indices, expected in cases.items():
        atom = IDiagramAtom(indices=indices, variables=(), value=0.0, role="")
        assert atom.symbol == expected, indices

    # Leftover structure (unifilarity zeros with no past/future anatomy role).
    for indices in [(3,), (1,)]:
        atom = IDiagramAtom(indices=indices, variables=(), value=0.0, role="")
        assert atom.symbol is None, indices


def test_jurgens_labels_match_table_ii():
    """Every Table II membership set carries its taxonomy label."""
    for indices, label in JURGENS_LABELS.items():
        atom = IDiagramAtom(indices=indices, variables=(), value=0.0, role="")
        assert atom.jurgens_label == label


def test_theorem_a_visible_in_bound_zone_symbols():
    """bμ = b⁺μ fwd + b⁺μ joint; b⁺μ gauge + b⁺μ rev = 0 (Theorem A)."""
    pytest.importorskip("dit")
    diagram = information_diagram(_processes()["nemo"], show_zero=True)
    by_symbol = {a.symbol: a.value_float() for a in diagram.atoms}
    b_mu = diagram.totals["b_mu"]
    assert by_symbol["b⁺μ fwd"] + by_symbol["b⁺μ joint"] == pytest.approx(float(b_mu), abs=1e-9)
    assert by_symbol["b⁺μ gauge"] + by_symbol["b⁺μ rev"] == pytest.approx(0.0, abs=1e-9)


def test_nemo_has_twenty_nonzero_atoms():
    """Independent check: nemo realises 13 Table II atoms + 7 cancelling extras."""
    pytest.importorskip("dit")
    diagram = information_diagram(_processes()["nemo"], atoms="process")
    assert len(diagram.atoms) == 20
    table_ii = {a.indices for a in diagram.atoms if a.indices in JURGENS_LABELS}
    extras = {a.indices for a in diagram.atoms if a.indices not in JURGENS_LABELS}
    assert len(table_ii) == 13  # p.r±μ is zero for nemo
    assert len(extras) == 7


def test_generic_mode_retains_twenty_one_atoms():
    """atoms='generic' keeps the 21 generically nonzero membership sets."""
    pytest.importorskip("dit")
    diagram = information_diagram(bernoulli(0.5), atoms="generic")
    assert len(diagram.atoms) == len(GENERICALLY_NONZERO) == 21
    assert {a.indices for a in diagram.atoms} == set(GENERICALLY_NONZERO)


def test_every_plotted_atom_has_a_name():
    pytest.importorskip("dit")
    diagram = information_diagram(_processes()["nemo"], show_zero=True)
    for atom in diagram.atoms:
        assert atom.conditional_expression.startswith(("H[", "I["))


def test_plot_information_diagram_smoke():
    """The UpSet plot builds a two-panel figure with one bar per atom."""
    pytest.importorskip("dit")
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from sofic.viz import plot_information_diagram

    bidir = _processes()["nemo"]
    n_atoms = len(information_diagram(bidir).atoms)
    fig = plot_information_diagram(bidir, title="nemo")
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        bars = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
        assert len(bars) == n_atoms
    finally:
        plt.close(fig)


def test_plot_information_diagram_empty_raises():
    """Filtering out every atom should be reported rather than drawing nothing."""
    pytest.importorskip("dit")
    pytest.importorskip("matplotlib")
    from sofic.viz import plot_information_diagram

    empty = InformationDiagram(atoms=[], variable_names=("S⁺₀", "S⁻₀", "X₀", "S⁺₁", "S⁻₁"), totals={})
    with pytest.raises(ValueError, match="no atoms"):
        plot_information_diagram(empty)
