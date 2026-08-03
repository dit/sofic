"""UpSet-style plot of the five-variable information-anatomy I-diagram.

Renders the atoms of :func:`sofic.generators.information_diagram.information_diagram`
as an UpSet plot (:cite:`lex2014upset`): a signed bar per atom (so negative
co-information atoms dip below zero), a width-scaled anatomy-aggregate legend
above the bars, a dot-matrix below showing which of the five random variables
``(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)`` are inside each atom, and a colour per anatomy
aggregate (``r_μ``, ``b⁺_μ``, ``b⁻_μ``, ``q_μ``, ``σ_μ``, ``χ⁺``, ``χ⁻``).
Atoms are laid out in the fixed
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

#: Legend symbols keyed by colour group (one legend entry per aggregate).
GROUP_LABELS: dict[str, str] = {
    "r_mu": "rμ",
    "b_plus": "b⁺μ",
    "b_minus": "b⁻μ",
    "q_mu": "qμ",
    "sigma_mu": "σμ",
    "chi_plus": "χ⁺",
    "chi_minus": "χ⁻",
    "structure": "struct",
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


#: Approximate horizontal inches per compact ``symbol = value`` legend entry.
_LEGEND_ENTRY_WIDTH_IN = 1.1
#: Vertical gridspec share for the legend title row + each entry row.
_LEGEND_TITLE_RATIO = 0.28
_LEGEND_ROW_RATIO = 0.22

#: Semantic columns used when the legend is laid out in four columns:
#: rμ | (b⁺μ, b⁻μ) | (qμ, σμ) | (χ⁺, χ⁻).  Rare once entries are compact
#: enough to fit one aggregate per column on a typical figure width.
_LEGEND_COLUMN_GROUPS_4: tuple[tuple[str, ...], ...] = (
    ("r_mu",),
    ("b_plus", "b_minus"),
    ("q_mu", "sigma_mu"),
    ("chi_plus", "chi_minus"),
)


def _legend_ncols(fig_width: float, n_items: int) -> int:
    """How many legend columns fit in ``fig_width`` without crowding.

    Driven by figure width (itself a function of the number of bars), so machines
    with similar atom counts share the same legend geometry.
    """
    if n_items <= 0:
        return 1
    usable = max(fig_width - 0.9, _LEGEND_ENTRY_WIDTH_IN)
    fit = max(1, int(usable // _LEGEND_ENTRY_WIDTH_IN))
    return max(1, min(n_items, fit))


def _pack_legend_handles(
    handles_by_group: dict[str, Any],
    ncol: int,
    *,
    empty_patch: Any,
) -> tuple[list[Any], int]:
    """Order legend handles for matplotlib's column-major ``ncol`` packing.

    When ``ncol == 4``, pack into the semantic columns of
    :data:`_LEGEND_COLUMN_GROUPS_4` (padding shorter columns so ``rμ`` sits alone
    in the first column).  Otherwise keep :data:`GROUP_ORDER`.
    """
    if ncol == 4:
        columns: list[list[Any]] = []
        for keys in _LEGEND_COLUMN_GROUPS_4:
            col = [handles_by_group[g] for g in keys if g in handles_by_group]
            if col:
                columns.append(col)
        # Leftover groups (e.g. structure) append to the last column.
        placed = {g for keys in _LEGEND_COLUMN_GROUPS_4 for g in keys}
        leftovers = [handles_by_group[g] for g in GROUP_ORDER if g in handles_by_group and g not in placed]
        if leftovers:
            if columns:
                columns[-1].extend(leftovers)
            else:
                columns.append(leftovers)
        if not columns:
            return [], 1
        ncol_eff = len(columns)
        height = max(len(col) for col in columns)
        packed: list[Any] = []
        for col in columns:
            packed.extend(col)
            packed.extend([empty_patch] * (height - len(col)))
        return packed, ncol_eff

    ordered = [handles_by_group[g] for g in GROUP_ORDER if g in handles_by_group]
    return ordered, max(1, min(ncol, len(ordered) or 1))

def plot_information_diagram(
    source: Any,
    *,
    show_zero: bool = False,
    atoms: Literal["process", "generic", "all"] | None = None,
    role_colors: dict[str, str] | None = None,
    annotate: bool = True,
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

    summary = diagram.totals
    present_groups = [
        group for group in GROUP_ORDER if any(COLOR_GROUP[a.role] == group for a in plotted)
    ]
    handles_by_group: dict[str, Any] = {}
    for group in present_groups:
        role_for_color = next(role for role, g in COLOR_GROUP.items() if g == group)
        total = summary.get(GROUP_TOTAL_KEY[group])
        label = GROUP_LABELS[group]
        if total is not None:
            label = f"{label} = {float(total):+.3f}"
        handles_by_group[group] = mpatches.Patch(
            facecolor=colors[role_for_color],
            edgecolor="0.25",
            lw=0.5,
            label=label,
        )

    if figsize is None:
        figsize = (max(9.0, 0.55 * n + 1.2), 5.6)
    n_legend = len(handles_by_group)
    ncol = _legend_ncols(figsize[0], n_legend)
    empty_patch = mpatches.Patch(facecolor="none", edgecolor="none", label=" ")
    handles, ncol = _pack_legend_handles(handles_by_group, ncol, empty_patch=empty_patch)
    # Column-major packing: rows = max column height (including rμ's spacer).
    n_rows = (len(handles) // ncol) if ncol and handles else 1
    legend_ratio = _LEGEND_TITLE_RATIO + _LEGEND_ROW_RATIO * max(n_rows, 1)

    fig = plt.figure(figsize=figsize)
    # Outer split keeps a little air under the legend; the bar + UpSet matrix
    # share a nested gridspec with almost no gap so the dots sit under the bars.
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[legend_ratio, 4.3],
        hspace=0.08,
    )
    gs_plot = gs[1].subgridspec(2, 1, height_ratios=[3.0, 1.3], hspace=0.0)
    ax_leg = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs_plot[0])
    ax_mat = fig.add_subplot(gs_plot[1], sharex=ax_bar)
    ax_leg.set_axis_off()

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
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    for side in ("top", "right", "bottom"):
        ax_bar.spines[side].set_visible(False)

    fig.suptitle(title or "Five-variable information anatomy I-diagram", fontsize=11, y=0.995)

    # Top row is a gray strip (var_index 0); match the axes face so the bar
    # x-axis borders gray rather than a white fringe at the join.
    ax_mat.set_facecolor("0.95")
    for var_index in range(n_vars):
        y = n_vars - 1 - var_index
        if var_index % 2 == 1:
            ax_mat.axhspan(y - 0.5, y + 0.5, color="1.0", zorder=0)
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
    # First gray strip is the top row at y = n_vars - 1, spanning
    # [n_vars - 1.5, n_vars - 0.5]. Flush the axes top to that edge and draw
    # the bar x-axis there so the black line borders the gray strip.
    top = n_vars - 0.5
    ax_mat.set_ylim(-0.5, top)
    ax_mat.axhline(top, color="0.15", lw=1.0, solid_capstyle="butt", zorder=10)
    ax_mat.set_yticks([])
    ax_mat.set_xticks([])
    ax_mat.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_mat.spines.values():
        spine.set_visible(False)

    # Width-scaled columns; at ncol==4 use semantic grouping
    # rμ | (b⁺, b⁻) | (q, σ) | (χ⁺, χ⁻).
    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        fontsize=7,
        framealpha=0.92,
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.5,
        title="anatomy aggregate",
        title_fontsize=8,
    )

    fig.subplots_adjust(left=0.075, right=0.98, top=0.93, bottom=0.05)
    # After layout, seat the matrix flush under the bar so the bar x-axis
    # (bottom spine) borders the top of the first gray strip.
    bar_pos = ax_bar.get_position()
    mat_pos = ax_mat.get_position()
    lift = bar_pos.y0 - mat_pos.y1
    if abs(lift) > 1e-6:
        ax_mat.set_position([mat_pos.x0, mat_pos.y0 + lift, mat_pos.width, mat_pos.height])
    return fig
