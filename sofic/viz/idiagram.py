"""UpSet-style plot of the five-variable information-anatomy I-diagram.

Renders the atoms of :func:`sofic.generators.information_diagram.information_diagram`
as an UpSet plot (:cite:`lex2014upset`): a signed bar per atom on top (so
negative co-information atoms dip below zero), a dot-matrix below showing which
of the five random variables ``(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)`` are inside each atom,
and a colour per anatomy aggregate (``r_μ``, ``b⁺_μ``, ``b⁻_μ``, ``q_μ``,
``σ_μ``, ``χ⁺``, ``χ⁻``).  Atoms are laid out in the fixed
:data:`~sofic.generators.information_diagram.ROLE_ORDER`, never sorted by value,
and labeled with the Jurgens taxonomy name when one exists
(:cite:`jurgens2026taxonomy` Table II).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from sofic.generators.information_diagram import (
    COLOR_GROUP,
    ROLE_ORDER,
    ROLE_TOTAL_KEY,
    InformationDiagram,
    information_diagram,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Default colour per fine role.  Roles that contribute to the same aggregate
#: share a colour (all ``r_*`` red, both ``b_plus`` atoms dark green, …).
DEFAULT_ROLE_COLORS: dict[str, str] = {
    # rμ — red
    "r_gauge": "#c1121f",
    "r_fwd": "#c1121f",
    "r_rev": "#c1121f",
    "r_joint": "#c1121f",
    # b⁺μ — darkish green
    "b_plus": "#1b4332",
    # b⁻μ — lighter green
    "b_minus": "#74c69d",
    # qμ — purple
    "q_mu": "#7b2cbf",
    # σμ — blue
    "sigma_mu": "#1d4e89",
    # χ⁺ / χ⁻ — contrasting warm accents (orange / amber)
    "chi_plus": "#e85d04",
    "chi_minus": "#ffba08",
    # leftover structure (always zero under unifilarity)
    "structure": "#adb5bd",
}

#: Legend labels keyed by colour group (one legend entry per aggregate).
GROUP_LABELS: dict[str, str] = {
    "r_mu": "rμ (ephemeral)",
    "b_plus": "b⁺μ (forward binding)",
    "b_minus": "b⁻μ (reverse binding)",
    "q_mu": "qμ (enigmatic)",
    "sigma_mu": "σμ (elusive)",
    "chi_plus": "χ⁺ (forward crypticity)",
    "chi_minus": "χ⁻ (reverse crypticity)",
    "structure": "state structure",
}

#: Totals key for each colour-group legend entry.
GROUP_TOTAL_KEY: dict[str, str] = {
    "r_mu": "r_mu",
    "b_plus": "b_plus",
    "b_minus": "b_minus",
    "q_mu": "q_mu",
    "sigma_mu": "sigma_mu",
    "chi_plus": "chi_plus",
    "chi_minus": "chi_minus",
    "structure": "structure",
}

#: Ordered colour groups for the legend.
GROUP_ORDER: tuple[str, ...] = (
    "r_mu",
    "b_plus",
    "b_minus",
    "q_mu",
    "sigma_mu",
    "chi_plus",
    "chi_minus",
    "structure",
)

# Back-compat aliases used by older docs / callers.
ROLE_LABELS: dict[str, str] = {
    role: GROUP_LABELS[COLOR_GROUP[role]] for role in ROLE_ORDER
}
ROLE_TOTAL_KEY_LEGACY = ROLE_TOTAL_KEY


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Information-diagram plotting requires the optional sofic[viz] extra "
            "(pip install 'sofic[viz]')."
        ) from exc
    return plt


def _atom_tick_label(atom: Any) -> str:
    """Primary label is the Jurgens / †-extra name; fall back to zone×branch."""
    primary = atom.jurgens_label or atom.symbol or atom.conditional_expression
    secondary = atom.conditional_expression
    if primary == secondary:
        return primary
    return f"{primary}\n{secondary}"


def plot_information_diagram(
    source: Any,
    *,
    show_zero: bool = False,
    atoms: Literal["process", "generic", "all"] | None = None,
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
        show_zero: Deprecated; prefer ``atoms``.  If ``True`` and ``atoms`` is
            omitted, keeps every atom (``atoms="all"``).
        atoms: ``"process"`` (default) — nonzero for this process;
            ``"generic"`` — the 21 generically nonzero membership sets;
            ``"all"`` — all 31 Yeung atoms.  Ignored when ``source`` is already
            an :class:`InformationDiagram`.
        role_colors: Overrides for :data:`DEFAULT_ROLE_COLORS` (per-role).
        annotate: Print each atom's value above/below its bar.
        label_atoms: Name each atom below the matrix by its Jurgens taxonomy
            label (Table II / †-extra) and conditional co-information.
        title: Plot title; a default anatomy title is used when ``None``.
        figsize: Figure size; auto-sized from the atom count when ``None``.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    plt = _require_matplotlib()
    import matplotlib.patches as mpatches

    if isinstance(source, InformationDiagram):
        diagram = source
    else:
        diagram = information_diagram(source, show_zero=show_zero, atoms=atoms)

    colors = {**DEFAULT_ROLE_COLORS, **(role_colors or {})}
    plotted = diagram.atoms
    if not plotted:
        raise ValueError("information diagram has no atoms to plot")

    var_names = diagram.variable_names
    n_vars = len(var_names)
    n = len(plotted)
    xs = list(range(n))
    values = [atom.value_float() for atom in plotted]
    bar_colors = [colors[atom.role] for atom in plotted]

    vmax = max(values + [0.0])
    vmin = min(values + [0.0])
    span = (vmax - vmin) or 1.0

    atom_labels = [_atom_tick_label(atom) for atom in plotted]
    label_chars = (
        max((max(len(line) for line in label.split("\n")) for label in atom_labels), default=0)
        if label_atoms
        else 0
    )
    label_in = 0.062 * label_chars

    if figsize is None:
        figsize = (max(8.5, 0.55 * n + 3.6), 5.0 + label_in)
    fig, (ax_bar, ax_mat) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.06},
    )

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
            f"b⁺μ={summary['b_plus']:.3f}   b⁻μ={summary['b_minus']:.3f}   "
            f"qμ={summary['q_mu']:.3f}   σμ={summary['sigma_mu']:.3f}"
        ),
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )

    for var_index in range(n_vars):
        y = n_vars - 1 - var_index
        if var_index % 2 == 0:
            ax_mat.axhspan(y - 0.5, y + 0.5, color="0.95", zorder=0)
        ax_mat.text(-1.2, y, var_names[var_index], ha="right", va="center", fontsize=9)

    for x, atom in zip(xs, plotted, strict=True):
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
        for tick, atom in zip(ax_mat.get_xticklabels(), plotted, strict=True):
            tick.set_color(colors[atom.role])
    else:
        ax_mat.set_xticks([])
    for spine in ax_mat.spines.values():
        spine.set_visible(False)

    present_groups = []
    for group in GROUP_ORDER:
        if any(COLOR_GROUP[a.role] == group for a in plotted):
            present_groups.append(group)
    handles = []
    for group in present_groups:
        # Pick a representative role colour for the group.
        role_for_color = next(role for role, g in COLOR_GROUP.items() if g == group)
        total = summary.get(GROUP_TOTAL_KEY[group])
        label = GROUP_LABELS[group]
        if total is not None:
            label = f"{label}  ({float(total):+.3f})"
        handles.append(
            mpatches.Patch(
                facecolor=colors[role_for_color],
                edgecolor="0.25",
                lw=0.5,
                label=label,
            )
        )
    ax_bar.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        framealpha=0.92,
        borderaxespad=0.0,
        title="anatomy aggregate",
        title_fontsize=8,
    )

    height = figsize[1]
    bottom = (label_in + 0.35) / height if label_atoms else 0.05
    fig.subplots_adjust(left=0.075, right=0.78, top=0.91, bottom=min(bottom, 0.5), hspace=0.06)
    return fig
