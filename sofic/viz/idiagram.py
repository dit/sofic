"""UpSet-style plot of the five-variable information-anatomy I-diagram.

Renders the atoms of :func:`sofic.generators.information_diagram.information_diagram`
as an UpSet plot (:cite:`lex2014upset`): a signed bar per atom on top (so
negative co-information atoms dip below zero), a dot-matrix below showing which
of the five random variables ``(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)`` are inside each atom,
and a colour per anatomy role so the components of ``r_μ``, ``b_μ``, ``σ_μ`` and
``ρ_μ`` are immediately legible.  Atoms are laid out in the fixed
:data:`~sofic.generators.information_diagram.ROLE_ORDER`, never sorted by value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sofic.generators.information_diagram import (
    ROLE_ORDER,
    ROLE_TOTAL_KEY,
    InformationDiagram,
    information_diagram,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Default colour per anatomy role.
DEFAULT_ROLE_COLORS: dict[str, str] = {
    "r_gauge": "#ffd166",  # gold — transient / gauge ephemeral
    "r_fwd": "#ef476f",  # pink-red — forward ephemeral branch
    "r_rev": "#118ab2",  # blue — reverse ephemeral branch
    "r_joint": "#8338ec",  # purple — joint ephemeral branch
    "bound": "#06d6a0",  # green — bound information b_μ
    "predictive": "#f78c6b",  # salmon — present shared with past only
    "shared": "#7f5539",  # brown — present · past · future co-information
    "elusive": "#264653",  # dark teal — elusive σ_μ (past↔future, no present)
    "structure": "#adb5bd",  # grey — causal-state structure
}

#: Human-readable legend labels per role.
ROLE_LABELS: dict[str, str] = {
    "r_gauge": "rμ gauge (transient)",
    "r_fwd": "rμ forward",
    "r_rev": "rμ reverse",
    "r_joint": "rμ joint",
    "bound": "bμ (bound)",
    "predictive": "ρμ (present · past)",
    "shared": "cμ (present · past · future)",
    "elusive": "σμ (elusive)",
    "structure": "state structure",
}


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Information-diagram plotting requires the optional sofic[viz] extra "
            "(pip install 'sofic[viz]')."
        ) from exc
    return plt


def plot_information_diagram(
    source: Any,
    *,
    show_zero: bool = False,
    role_colors: dict[str, str] | None = None,
    annotate: bool = True,
    label_atoms: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw the five-variable information anatomy as a colour-coded UpSet plot.

    Args:
        source: A bidirectional/forward ε-machine or a pre-computed
            :class:`~sofic.generators.information_diagram.InformationDiagram`.
        show_zero: Include atoms whose value is numerically zero.
        role_colors: Overrides for :data:`DEFAULT_ROLE_COLORS` (per-role).
        annotate: Print each atom's value above/below its bar.
        label_atoms: Name each atom below the matrix by its conditional
            co-information (and its short anatomy symbol, where one exists).
        title: Plot title; a default anatomy title is used when ``None``.
        figsize: Figure size; auto-sized from the atom count when ``None``.

    Returns:
        The :class:`matplotlib.figure.Figure`.  The bars are coloured by anatomy
        role (see :data:`ROLE_LABELS`), laid out in the fixed
        :data:`~sofic.generators.information_diagram.ROLE_ORDER`.
    """
    plt = _require_matplotlib()
    import matplotlib.patches as mpatches

    diagram = (
        source
        if isinstance(source, InformationDiagram)
        else information_diagram(source, show_zero=show_zero)
    )
    colors = {**DEFAULT_ROLE_COLORS, **(role_colors or {})}
    atoms = diagram.atoms
    if not atoms:
        raise ValueError("information diagram has no atoms to plot")

    var_names = diagram.variable_names
    n_vars = len(var_names)
    n = len(atoms)
    xs = list(range(n))
    values = [atom.value_float() for atom in atoms]
    bar_colors = [colors[atom.role] for atom in atoms]

    vmax = max(values + [0.0])
    vmin = min(values + [0.0])
    span = (vmax - vmin) or 1.0

    # Per-atom names go under the matrix as rotated labels; size the figure and
    # bottom margin to the longest label so nothing is clipped.
    atom_labels = []
    for atom in atoms:
        atom_labels.append(f"{atom.symbol}\n{atom.conditional_expression}" if atom.symbol else atom.conditional_expression)
    label_chars = max((len(atom.conditional_expression) for atom in atoms), default=0) if label_atoms else 0
    label_in = 0.062 * label_chars  # rotated-text height allowance (inches)

    if figsize is None:
        figsize = (max(8.5, 0.6 * n + 3.4), 4.9 + label_in)
    fig, (ax_bar, ax_mat) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.06},
    )

    # ── top: signed bars, coloured by anatomy role ──────────────────────────
    ax_bar.axhline(0.0, color="0.4", lw=0.8, zorder=1)
    ax_bar.bar(xs, values, width=0.72, color=bar_colors, edgecolor="0.25", lw=0.5, zorder=2)
    if annotate:
        pad = 0.02 * span
        for x, v in zip(xs, values, strict=True):
            ax_bar.text(
                x,
                v + (pad if v >= 0 else -pad),
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
            )
    ax_bar.set_ylabel("bits")
    ax_bar.set_ylim(vmin - 0.14 * span, vmax + 0.16 * span)
    ax_bar.set_xlim(-1.4, n - 0.5)
    ax_bar.set_xticks([])
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)

    summary = diagram.totals
    ax_bar.set_title(title or "Five-variable information anatomy I-diagram", fontsize=11)
    ax_bar.text(
        0.005,
        0.98,
        (
            f"hμ={summary['h_mu']:.3f}   rμ={summary['r_mu']:.3f}   "
            f"bμ={summary['b_mu']:.3f}   σμ={summary['sigma_mu']:.3f}   "
            f"ρμ={summary['rho_mu']:.3f}"
        ),
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )

    # ── bottom: UpSet membership dot-matrix ─────────────────────────────────
    for var_index in range(n_vars):
        y = n_vars - 1 - var_index
        if var_index % 2 == 0:
            ax_mat.axhspan(y - 0.5, y + 0.5, color="0.95", zorder=0)
        ax_mat.text(-1.2, y, var_names[var_index], ha="right", va="center", fontsize=9)

    for x, atom in zip(xs, atoms, strict=True):
        inside = set(atom.indices)
        color = colors[atom.role]
        ys_inside = []
        for var_index in range(n_vars):
            y = n_vars - 1 - var_index
            if var_index in inside:
                ax_mat.plot(x, y, "o", color=color, ms=8, zorder=3)
                ys_inside.append(y)
            else:
                ax_mat.plot(x, y, "o", color="0.86", ms=8, zorder=2)
        if len(ys_inside) > 1:
            ax_mat.plot([x, x], [min(ys_inside), max(ys_inside)], color=color, lw=2.0, zorder=2)

    ax_mat.set_xlim(-1.4, n - 0.5)
    ax_mat.set_ylim(-0.6, n_vars - 0.4)
    ax_mat.set_yticks([])
    if label_atoms:
        ax_mat.set_xticks(xs)
        ax_mat.set_xticklabels(atom_labels, rotation=90, fontsize=6, va="top")
        ax_mat.tick_params(axis="x", length=0, pad=3)
        for tick, atom in zip(ax_mat.get_xticklabels(), atoms, strict=True):
            tick.set_color(colors[atom.role])
    else:
        ax_mat.set_xticks([])
    for spine in ax_mat.spines.values():
        spine.set_visible(False)

    # ── legend: one entry per role present, with its aggregate value ────────
    present_roles = [role for role in ROLE_ORDER if any(a.role == role for a in atoms)]
    handles = []
    for role in present_roles:
        total = summary.get(ROLE_TOTAL_KEY[role])
        label = ROLE_LABELS[role]
        if total is not None:
            label = f"{label}  ({float(total):+.3f})"
        handles.append(mpatches.Patch(facecolor=colors[role], edgecolor="0.25", lw=0.5, label=label))
    ax_bar.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        framealpha=0.92,
        borderaxespad=0.0,
        title="anatomy role",
        title_fontsize=8,
    )

    # Explicit margins (tight_layout is incompatible with the dot-matrix axes);
    # reserve the right band for the outside legend and the bottom for labels.
    height = figsize[1]
    bottom = (label_in + 0.35) / height if label_atoms else 0.05
    fig.subplots_adjust(left=0.075, right=0.79, top=0.91, bottom=min(bottom, 0.5), hspace=0.06)
    return fig
