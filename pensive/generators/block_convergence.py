"""James et al. (2011) block-convergence / anatomy-of-a-bit curves for epsilon-machines."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pensive.generators.block_entropy import (
    _block_entropy_curves,
    _entropy,
    _entropy_rate,
    _entropy_rate_estimates,
    _estimated_excess_entropy,
    _predictability_gain,
)

if TYPE_CHECKING:
    from pensive.generators.epsilon_machine import EpsilonMachine

_TOL = 1e-15
_CAEKL_RATE_STABLE_STEPS = 2
_CAEKL_RATE_TOL = 1e-9
FigureName = Literal["all", "fig4", "fig5", "fig6", "caekl", "cm"]


def _require_dit():
    try:
        import dit
    except ImportError as exc:
        raise ImportError(
            "dit is required for block convergence measures; install with `pip install pensive[measures]`"
        ) from exc
    return dit


def _positional_rvs(length: int) -> list[list[int]]:
    return [[index] for index in range(length)]


def _word_distribution(
    distribution: dict[tuple[Any, ...], float],
) -> Any:
    dit = _require_dit()
    if not distribution:
        return dit.Distribution([()], [1.0])
    total = sum(float(prob) for prob in distribution.values())
    if total <= _TOL:
        return dit.Distribution([()], [1.0])
    outcomes = list(distribution.keys())
    probs = [float(distribution[outcome]) / total for outcome in outcomes]
    dist = dit.Distribution(outcomes, probs)
    dist.set_rv_names([f"X_{index}" for index in range(len(outcomes[0]))])
    return dist


def _block_word_distribution(machine: EpsilonMachine, length: int) -> Any:
    if length == 0:
        dit = _require_dit()
        return dit.Distribution([()], [1.0])
    return _word_distribution(machine.word_probabilities(length))


def _block_total_correlation(dist: Any, length: int, block_entropy: float, h1: float) -> float:
    if length <= 1:
        return 0.0
    dit = _require_dit()
    from dit.multivariate import total_correlation

    return float(total_correlation(dist, rvs=_positional_rvs(length)))


def _block_residual_entropy(dist: Any, length: int) -> float:
    if length == 0:
        return 0.0
    dit = _require_dit()
    from dit.multivariate import residual_entropy

    return float(residual_entropy(dist, rvs=_positional_rvs(length)))


def _block_binding_information(dist: Any, length: int, block_entropy: float) -> float:
    if length == 0:
        return 0.0
    return block_entropy - _block_residual_entropy(dist, length)


def _block_coinformation(dist: Any, length: int, block_entropy: float) -> float:
    if length <= 1:
        return 0.0
    dit = _require_dit()
    from dit.multivariate import coinformation

    return float(coinformation(dist, rvs=_positional_rvs(length)))


def _block_caekl(dist: Any, length: int) -> float:
    if length <= 1:
        return 0.0
    dit = _require_dit()
    from dit.multivariate import caekl_mutual_information

    return float(caekl_mutual_information(dist, rvs=_positional_rvs(length)))


def block_caekl(machine: EpsilonMachine, length: int) -> float:
    """Exact block CAEKL mutual information ``J(ℓ)`` for block length ``length``.

    Uses the ε-machine word distribution ``P(X_{0:ℓ-1})`` and
    ``dit.multivariate.caekl_mutual_information``.  Requires ``dit``
    (``pip install pensive[measures]``).  Returns ``0`` for ``length <= 1``.
    """
    if length < 0:
        raise ValueError("length must be nonnegative")
    if length <= 1:
        return 0.0
    dist = _block_word_distribution(machine, length)
    return _block_caekl(dist, length)


def _block_caekl_curve(
    machine: EpsilonMachine,
    max_length: int,
    *,
    max_caekl_length: int | None = None,
) -> np.ndarray:
    """Build ``J(ℓ)`` for ``ℓ = 0 … max_length`` (zeros where not computed)."""
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")
    caekl_limit = max_length if max_caekl_length is None else min(max_length, max_caekl_length)
    block_caekl = np.zeros(max_length + 1, dtype=float)
    for length in range(2, caekl_limit + 1):
        block_caekl[length] = block_caekl(machine, length)
    return block_caekl


@dataclass(frozen=True)
class CurveConvergence:
    """Rate/intercept/asymptote arrays for one block curve."""

    rate_estimate: np.ndarray
    intercept: np.ndarray
    asymptote: np.ndarray
    rate: float
    intercept_scalar: float


def _promoted_rate(curve: np.ndarray, *, exact: float | None = None) -> float:
    if exact is not None:
        return float(exact)
    if len(curve) < 2:
        return math.nan
    estimates = _entropy_rate_estimates(curve)
    return float(estimates[-1])


def _convergence_scalars(
    lengths: np.ndarray,
    curve: np.ndarray,
    *,
    rate: float | None = None,
) -> CurveConvergence:
    rate_estimate = _entropy_rate_estimates(curve)
    promoted_rate = _promoted_rate(curve, exact=rate)
    intercept = np.array(curve, dtype=float, copy=True)
    if math.isfinite(promoted_rate):
        intercept = curve - promoted_rate * lengths
    intercept_scalar = float(intercept[-1]) if intercept.size else math.nan
    asymptote = curve.copy()
    if math.isfinite(promoted_rate):
        asymptote = intercept_scalar + promoted_rate * lengths
    return CurveConvergence(
        rate_estimate=rate_estimate,
        intercept=intercept,
        asymptote=asymptote,
        rate=promoted_rate,
        intercept_scalar=intercept_scalar,
    )


@dataclass(frozen=True)
class CaeklConvergence(CurveConvergence):
    """CAEKL block-curve convergence with an affine-tail certification flag."""

    converged: bool


def _caekl_convergence_scalars(
    lengths: np.ndarray,
    block_caekl: np.ndarray,
    *,
    stable_steps: int = _CAEKL_RATE_STABLE_STEPS,
    tol: float = _CAEKL_RATE_TOL,
) -> CaeklConvergence:
    """Promote ``j_μ`` from ``J(ℓ)``, certifying exact rate when ``ΔJ`` stabilizes."""
    rate_estimate = _entropy_rate_estimates(block_caekl)
    converged = False
    promoted_rate = math.nan

    if len(block_caekl) > 2 and stable_steps >= 1:
        diffs = rate_estimate[2:]
        for start in range(len(diffs) - stable_steps + 1):
            window = diffs[start : start + stable_steps]
            if not all(math.isfinite(value) for value in window):
                continue
            reference = float(window[0])
            if all(abs(float(value) - reference) <= tol for value in window):
                promoted_rate = reference
                converged = True
                break

    if not converged:
        promoted_rate = _promoted_rate(block_caekl)

    intercept = np.array(block_caekl, dtype=float, copy=True)
    if math.isfinite(promoted_rate):
        intercept = block_caekl - promoted_rate * lengths
    intercept_scalar = float(intercept[-1]) if intercept.size else math.nan
    asymptote = block_caekl.copy()
    if math.isfinite(promoted_rate):
        asymptote = intercept_scalar + promoted_rate * lengths

    return CaeklConvergence(
        rate_estimate=rate_estimate,
        intercept=intercept,
        asymptote=asymptote,
        rate=promoted_rate,
        intercept_scalar=intercept_scalar,
        converged=converged,
    )


def _anatomy_curves(
    machine: EpsilonMachine,
    max_length: int,
    *,
    block_entropy: np.ndarray,
    h1: float,
    max_caekl_length: int | None = None,
) -> dict[str, np.ndarray]:
    block_total_correlation = np.zeros(max_length + 1, dtype=float)
    block_residual_entropy = np.zeros(max_length + 1, dtype=float)
    block_binding_information = np.zeros(max_length + 1, dtype=float)
    block_enigmatic_information = np.zeros(max_length + 1, dtype=float)
    block_local_exogenous_information = np.zeros(max_length + 1, dtype=float)
    block_coinformation = np.zeros(max_length + 1, dtype=float)
    caekl_curve = np.zeros(max_length + 1, dtype=float)

    for length in range(max_length + 1):
        h_l = float(block_entropy[length])
        if length == 0:
            continue
        if length == 1:
            block_total_correlation[length] = 0.0
            block_residual_entropy[length] = h_l
            block_binding_information[length] = 0.0
            block_enigmatic_information[length] = 0.0
            block_local_exogenous_information[length] = 0.0
            block_coinformation[length] = 0.0
            caekl_curve[length] = 0.0
            continue

        dist = _block_word_distribution(machine, length)
        t_l = _block_total_correlation(dist, length, h_l, h1)
        r_l = _block_residual_entropy(dist, length)
        b_l = h_l - r_l
        block_total_correlation[length] = t_l
        block_residual_entropy[length] = r_l
        block_binding_information[length] = b_l
        block_enigmatic_information[length] = t_l - b_l
        block_local_exogenous_information[length] = b_l + t_l
        block_coinformation[length] = _block_coinformation(dist, length, h_l)

    caekl_limit = max_length if max_caekl_length is None else min(max_length, max_caekl_length)
    for length in range(2, caekl_limit + 1):
        caekl_curve[length] = block_caekl(machine, length)

    return {
        "block_total_correlation": block_total_correlation,
        "block_residual_entropy": block_residual_entropy,
        "block_binding_information": block_binding_information,
        "block_enigmatic_information": block_enigmatic_information,
        "block_local_exogenous_information": block_local_exogenous_information,
        "block_coinformation": block_coinformation,
        "block_caekl": caekl_curve,
    }


def _exact_anatomy_scalars(machine: EpsilonMachine) -> dict[str, float] | None:
    try:
        bidir = machine.to_bidirectional()
    except Exception:
        return None
    h_mu = float(bidir.entropy_rate())
    rho_mu = float(bidir.predicted_information())
    b_mu = float(bidir.bound_information())
    r_mu = float(bidir.ephemeral_information())
    if abs(b_mu + r_mu - h_mu) > 1e-9:
        return None
    q_mu = rho_mu - b_mu
    w_mu = h_mu + rho_mu
    excess = float(bidir.excess_entropy())
    return {
        "entropy_rate": h_mu,
        "predicted_information": rho_mu,
        "bound_information": b_mu,
        "ephemeral_information": r_mu,
        "q_mu": q_mu,
        "w_mu": w_mu,
        "excess_entropy": excess,
    }


def _attach_convergence(
    lengths: np.ndarray,
    curve: np.ndarray,
    *,
    rate: float | None = None,
) -> dict[str, Any]:
    conv = _convergence_scalars(lengths, curve, rate=rate)
    return {
        "curve": curve,
        "rate_estimate": conv.rate_estimate,
        "intercept": conv.intercept,
        "asymptote": conv.asymptote,
        "rate": conv.rate,
        "intercept_scalar": conv.intercept_scalar,
    }


@dataclass(frozen=True)
class BlockConvergenceDiagram:
    """James (2011) block convergence curves for an epsilon-machine."""

    lengths: np.ndarray
    block_entropy: np.ndarray
    block_total_correlation: np.ndarray
    block_residual_entropy: np.ndarray
    block_binding_information: np.ndarray
    block_enigmatic_information: np.ndarray
    block_local_exogenous_information: np.ndarray
    block_coinformation: np.ndarray
    block_caekl: np.ndarray
    state_block_entropy: np.ndarray
    block_state_entropy: np.ndarray
    crypticity_estimate: np.ndarray
    entropy_asymptote: np.ndarray
    entropy_rate_estimate: np.ndarray
    total_correlation_rate_estimate: np.ndarray
    total_correlation_intercept: np.ndarray
    total_correlation_asymptote: np.ndarray
    residual_entropy_rate_estimate: np.ndarray
    residual_entropy_intercept: np.ndarray
    residual_entropy_asymptote: np.ndarray
    binding_information_rate_estimate: np.ndarray
    binding_information_intercept: np.ndarray
    binding_information_asymptote: np.ndarray
    enigmatic_information_rate_estimate: np.ndarray
    enigmatic_information_intercept: np.ndarray
    enigmatic_information_asymptote: np.ndarray
    local_exogenous_information_rate_estimate: np.ndarray
    local_exogenous_information_intercept: np.ndarray
    local_exogenous_information_asymptote: np.ndarray
    coinformation_rate_estimate: np.ndarray
    coinformation_intercept: np.ndarray
    coinformation_asymptote: np.ndarray
    caekl_rate_estimate: np.ndarray
    caekl_intercept: np.ndarray
    caekl_asymptote: np.ndarray
    entropy_rate: float
    predicted_information: float
    ephemeral_information: float
    bound_information: float
    q_mu: float
    w_mu: float
    excess_entropy: float
    E_B: float
    E_R: float
    E_Q: float
    E_W: float
    caekl_rate: float
    caekl_intercept_scalar: float
    caekl_rate_converged: bool
    statistical_complexity: float
    crypticity: float
    markov_order: int | float
    cryptic_order: int | float

    @property
    def h_mu(self) -> float:
        return self.entropy_rate

    @property
    def rho_mu(self) -> float:
        return self.predicted_information

    @property
    def r_mu(self) -> float:
        return self.ephemeral_information

    @property
    def b_mu(self) -> float:
        return self.bound_information

    @property
    def E(self) -> float:
        return self.excess_entropy

    @property
    def j_mu(self) -> float:
        return self.caekl_rate

    @property
    def J_inf(self) -> float:
        return self.caekl_intercept_scalar

    def validate_identities(self, *, tol: float = 1e-9) -> None:
        """Assert James (2011) block identities at each finite length."""
        h1 = float(self.block_entropy[1]) if len(self.block_entropy) > 1 else math.nan
        for length in self.lengths:
            index = int(length)
            h_l = float(self.block_entropy[index])
            t_l = float(self.block_total_correlation[index])
            r_l = float(self.block_residual_entropy[index])
            b_l = float(self.block_binding_information[index])
            q_l = float(self.block_enigmatic_information[index])
            w_l = float(self.block_local_exogenous_information[index])
            if index == 0:
                continue
            assert abs(h_l - (b_l + r_l)) <= tol, (index, h_l, b_l, r_l)
            assert abs(t_l - (b_l + q_l)) <= tol, (index, t_l, b_l, q_l)
            assert abs(w_l - (b_l + t_l)) <= tol, (index, w_l, b_l, t_l)
            if index >= 1 and math.isfinite(h1):
                assert abs(h_l + t_l - index * h1) <= tol, (index, h_l, t_l, h1)
                assert abs(r_l + w_l - index * h1) <= tol, (index, r_l, w_l, h1)

    def plot(
        self,
        ax: Any | None = None,
        *,
        figure: FigureName = "all",
        show_asymptotes: bool = True,
        show_grid: bool = True,
        show_legend: bool = True,
        title: str | None = None,
        marker: str = "o",
    ) -> Any:
        if ax is None:
            import matplotlib.pyplot as plt

            if figure == "all":
                _, axes = plt.subplots(2, 3, figsize=(12, 7))
                axes_list = axes.ravel()
                for panel, name in zip(
                    axes_list,
                    ("fig4", "fig5", "fig6", "caekl", "cm", "fig4"),
                    strict=False,
                ):
                    if name == "cm":
                        self._plot_cm(panel, marker=marker, show_grid=show_grid, show_legend=show_legend)
                    else:
                        self.plot(ax=panel, figure=name, show_asymptotes=show_asymptotes, show_grid=show_grid, show_legend=show_legend)
                if title is not None:
                    axes_list[0].figure.suptitle(title)
                return axes_list
            _, ax = plt.subplots()

        if figure == "fig4":
            ax.plot(self.lengths, self.block_entropy, marker=marker, label=r"$H(\ell)$")
            ax.plot(self.lengths, self.block_total_correlation, marker=marker, label=r"$T(\ell)$")
            if show_asymptotes:
                ax.plot(
                    self.lengths,
                    self.entropy_asymptote,
                    linestyle="--",
                    color="black",
                    label=r"$E + h_\mu \ell$",
                )
                ax.plot(
                    self.lengths,
                    self.total_correlation_asymptote,
                    linestyle="--",
                    color="tab:orange",
                    label=r"$-E + \rho_\mu \ell$",
                )
        elif figure == "fig5":
            ax.plot(self.lengths, self.block_residual_entropy, marker=marker, label=r"$R(\ell)$")
            ax.plot(self.lengths, self.block_binding_information, marker=marker, label=r"$B(\ell)$")
            ax.plot(self.lengths, self.block_enigmatic_information, marker=marker, label=r"$Q(\ell)$")
            ax.plot(self.lengths, self.block_local_exogenous_information, marker=marker, label=r"$W(\ell)$")
            if show_asymptotes:
                ax.plot(self.lengths, self.residual_entropy_asymptote, linestyle="--", label=r"$E_R + r_\mu \ell$")
                ax.plot(self.lengths, self.binding_information_asymptote, linestyle="--", label=r"$E_B + b_\mu \ell$")
                ax.plot(self.lengths, self.enigmatic_information_asymptote, linestyle="--", label=r"$E_Q + q_\mu \ell$")
                ax.plot(self.lengths, self.local_exogenous_information_asymptote, linestyle="--", label=r"$E_W + w_\mu \ell$")
        elif figure == "fig6":
            ax.plot(self.lengths, self.block_coinformation, marker=marker, label=r"$I(\ell)$")
            if show_asymptotes:
                ax.plot(self.lengths, self.coinformation_asymptote, linestyle="--", label=r"$I + i_\mu \ell$")
        elif figure == "caekl":
            ax.plot(self.lengths, self.block_caekl, marker=marker, label=r"$J(\ell)$")
            if show_asymptotes:
                ax.plot(self.lengths, self.caekl_asymptote, linestyle="--", label=r"$J_\infty + j_\mu \ell$")
        elif figure == "cm":
            self._plot_cm(ax, marker=marker, show_grid=False, show_legend=show_legend, show_asymptotes=show_asymptotes)
        else:
            raise ValueError(f"unknown figure {figure!r}")

        if show_grid:
            ax.set_axisbelow(True)
            ax.grid(True, color="0.85", linewidth=0.8)
        ax.set_xlabel(r"Block length $\ell$")
        ax.set_ylabel("Information (bits)")
        if title is not None:
            ax.set_title(title)
        if show_legend:
            ax.legend()
        return ax

    def _plot_cm(
        self,
        ax: Any,
        *,
        marker: str,
        show_grid: bool,
        show_legend: bool,
        show_asymptotes: bool = True,
    ) -> None:
        ax.plot(self.lengths, self.block_entropy, marker=marker, label=r"$H[X_{0:\ell}]$")
        ax.plot(self.lengths, self.state_block_entropy, marker=marker, label=r"$H[S_0, X_{0:\ell}]$")
        ax.plot(self.lengths, self.block_state_entropy, marker=marker, label=r"$H[X_{0:\ell}, S_\ell]$")
        ax.plot(self.lengths, self.crypticity_estimate, marker=marker, linestyle="-.", label=r"$\chi(\ell)$")
        if show_asymptotes:
            ax.plot(self.lengths, self.entropy_asymptote, linestyle="--", color="black", label=r"$E + h_\mu \ell$")
        if show_grid:
            ax.set_axisbelow(True)
            ax.grid(True, color="0.85", linewidth=0.8)
        ax.set_xlabel(r"Block length $\ell$")
        ax.set_ylabel("Information (bits)")
        if show_legend:
            ax.legend()


@dataclass(frozen=True)
class BlockConvergenceEstimates(BlockConvergenceDiagram):
    """Block convergence diagram plus Ellison/Mahoney CM extension curves."""

    excess_entropy_lower: np.ndarray
    excess_entropy_upper: np.ndarray
    excess_entropy_estimate: np.ndarray
    synchronization_estimate: np.ndarray
    reverse_synchronization_estimate: np.ndarray
    transient_information_estimate: np.ndarray
    predictability_gain_estimate: np.ndarray
    oracular_information_estimate: np.ndarray
    gauge_information_estimate: np.ndarray

    def information_anatomy(self) -> dict[str, float]:
        """Return promoted anatomy scalars using James (2011) key names."""
        return {
            "rho_mu": self.predicted_information,
            "bound_mu": self.bound_information,
            "ephemeral_mu": self.ephemeral_information,
            "entropy_rate": self.entropy_rate,
            "excess_entropy": self.excess_entropy,
            "crypticity": self.crypticity,
            "q_mu": self.q_mu,
            "w_mu": self.w_mu,
            "E_B": self.E_B,
            "E_R": self.E_R,
            "E_Q": self.E_Q,
            "E_W": self.E_W,
            "j_mu": self.caekl_rate,
            "caekl_intercept": self.caekl_intercept_scalar,
            "caekl_rate_converged": self.caekl_rate_converged,
        }


def block_convergence_diagram(
    machine: EpsilonMachine,
    max_length: int,
    *,
    max_caekl_length: int | None = None,
) -> BlockConvergenceDiagram:
    estimates = block_convergence_estimates(
        machine,
        max_length,
        max_caekl_length=max_caekl_length,
    )
    return BlockConvergenceDiagram(
        **{
            field: getattr(estimates, field)
            for field in BlockConvergenceDiagram.__dataclass_fields__
        }
    )


def block_convergence_estimates(
    machine: EpsilonMachine,
    max_length: int,
    *,
    entropy_rate: float | None = None,
    use_exact: bool = True,
    max_caekl_length: int | None = None,
) -> BlockConvergenceEstimates:
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")
    if max_caekl_length is not None and max_caekl_length < 0:
        raise ValueError("max_caekl_length must be nonnegative")

    lengths, block_entropy, state_block_entropy, block_state_entropy, pi, symbol_matrices = _block_entropy_curves(
        machine,
        max_length,
    )
    h1 = float(block_entropy[1]) if max_length >= 1 else math.nan
    anatomy = _anatomy_curves(
        machine,
        max_length,
        block_entropy=block_entropy,
        h1=h1,
        max_caekl_length=max_caekl_length,
    )

    exact = _exact_anatomy_scalars(machine) if use_exact else None
    h_mu_exact = exact["entropy_rate"] if exact else None
    h_mu = float(entropy_rate if entropy_rate is not None else (_entropy_rate(pi, symbol_matrices) if h_mu_exact is None else h_mu_exact))

    statistical_complexity = _entropy(pi)
    excess_entropy = _estimated_excess_entropy(
        machine,
        0.5 * (block_entropy - h_mu * lengths + block_state_entropy - h_mu * lengths),
        use_exact=use_exact,
    )
    if exact is not None:
        excess_entropy = exact["excess_entropy"]
    crypticity = statistical_complexity - excess_entropy
    entropy_asymptote = excess_entropy + h_mu * lengths
    crypticity_estimate = state_block_entropy - block_state_entropy
    entropy_rate_estimate = _entropy_rate_estimates(block_entropy)

    h_conv = _convergence_scalars(lengths, block_entropy, rate=h_mu if exact else None)
    t_conv = _convergence_scalars(
        lengths,
        anatomy["block_total_correlation"],
        rate=exact["predicted_information"] if exact else None,
    )
    r_conv = _convergence_scalars(
        lengths,
        anatomy["block_residual_entropy"],
        rate=exact["ephemeral_information"] if exact else None,
    )
    b_conv = _convergence_scalars(
        lengths,
        anatomy["block_binding_information"],
        rate=exact["bound_information"] if exact else None,
    )
    q_conv = _convergence_scalars(lengths, anatomy["block_enigmatic_information"])
    w_conv = _convergence_scalars(lengths, anatomy["block_local_exogenous_information"])
    i_conv = _convergence_scalars(lengths, anatomy["block_coinformation"])
    j_conv = _caekl_convergence_scalars(lengths, anatomy["block_caekl"])

    rho_mu = exact["predicted_information"] if exact else t_conv.rate
    # James block residual R(ℓ) → b_μ; binding B(ℓ) = H(ℓ) − R(ℓ) → r_μ.
    b_mu = exact["bound_information"] if exact else r_conv.rate
    r_mu = exact["ephemeral_information"] if exact else b_conv.rate
    q_mu = exact["q_mu"] if exact else q_conv.rate
    w_mu = exact["w_mu"] if exact else w_conv.rate

    excess_entropy_lower = block_entropy - h_mu * lengths
    excess_entropy_upper = block_state_entropy - h_mu * lengths
    excess_entropy_estimate = 0.5 * (excess_entropy_lower + excess_entropy_upper)
    synchronization = block_state_entropy - block_entropy
    reverse_synchronization = state_block_entropy - block_entropy
    transient_information = np.cumsum(entropy_asymptote - block_entropy)
    predictability_gain = _predictability_gain(block_entropy)
    oracular_information = statistical_complexity + h_mu * lengths - state_block_entropy
    gauge_information = statistical_complexity - excess_entropy - crypticity_estimate - oracular_information

    return BlockConvergenceEstimates(
        lengths=lengths,
        block_entropy=block_entropy,
        block_total_correlation=anatomy["block_total_correlation"],
        block_residual_entropy=anatomy["block_residual_entropy"],
        block_binding_information=anatomy["block_binding_information"],
        block_enigmatic_information=anatomy["block_enigmatic_information"],
        block_local_exogenous_information=anatomy["block_local_exogenous_information"],
        block_coinformation=anatomy["block_coinformation"],
        block_caekl=anatomy["block_caekl"],
        state_block_entropy=state_block_entropy,
        block_state_entropy=block_state_entropy,
        crypticity_estimate=crypticity_estimate,
        entropy_asymptote=entropy_asymptote,
        entropy_rate_estimate=entropy_rate_estimate,
        total_correlation_rate_estimate=t_conv.rate_estimate,
        total_correlation_intercept=t_conv.intercept,
        total_correlation_asymptote=t_conv.asymptote,
        residual_entropy_rate_estimate=r_conv.rate_estimate,
        residual_entropy_intercept=r_conv.intercept,
        residual_entropy_asymptote=r_conv.asymptote,
        binding_information_rate_estimate=b_conv.rate_estimate,
        binding_information_intercept=b_conv.intercept,
        binding_information_asymptote=b_conv.asymptote,
        enigmatic_information_rate_estimate=q_conv.rate_estimate,
        enigmatic_information_intercept=q_conv.intercept,
        enigmatic_information_asymptote=q_conv.asymptote,
        local_exogenous_information_rate_estimate=w_conv.rate_estimate,
        local_exogenous_information_intercept=w_conv.intercept,
        local_exogenous_information_asymptote=w_conv.asymptote,
        coinformation_rate_estimate=i_conv.rate_estimate,
        coinformation_intercept=i_conv.intercept,
        coinformation_asymptote=i_conv.asymptote,
        caekl_rate_estimate=j_conv.rate_estimate,
        caekl_intercept=j_conv.intercept,
        caekl_asymptote=j_conv.asymptote,
        entropy_rate=h_mu,
        predicted_information=float(rho_mu),
        ephemeral_information=float(r_mu),
        bound_information=float(b_mu),
        q_mu=float(q_mu),
        w_mu=float(w_mu),
        excess_entropy=float(excess_entropy),
        E_B=float(b_conv.intercept_scalar),
        E_R=float(r_conv.intercept_scalar),
        E_Q=float(q_conv.intercept_scalar),
        E_W=float(w_conv.intercept_scalar),
        caekl_rate=float(j_conv.rate),
        caekl_intercept_scalar=float(j_conv.intercept_scalar),
        caekl_rate_converged=bool(j_conv.converged),
        statistical_complexity=float(statistical_complexity),
        crypticity=float(crypticity),
        markov_order=machine.markov_order(),
        cryptic_order=machine.cryptic_order(),
        excess_entropy_lower=excess_entropy_lower,
        excess_entropy_upper=excess_entropy_upper,
        excess_entropy_estimate=excess_entropy_estimate,
        synchronization_estimate=synchronization,
        reverse_synchronization_estimate=reverse_synchronization,
        transient_information_estimate=transient_information,
        predictability_gain_estimate=predictability_gain,
        oracular_information_estimate=oracular_information,
        gauge_information_estimate=gauge_information,
    )


def plot_block_convergence_diagram(
    machine: EpsilonMachine,
    max_length: int,
    ax: Any | None = None,
    **kwargs: Any,
) -> Any:
    return block_convergence_diagram(machine, max_length).plot(ax=ax, **kwargs)


def residual_entropy_from_distribution(distribution: dict[tuple[Any, ...], float]) -> float:
    """Dit-backed residual entropy for a word distribution."""
    if not distribution:
        return 0.0
    length = len(next(iter(distribution)))
    if length == 0:
        return 0.0
    dist = _word_distribution(distribution)
    return _block_residual_entropy(dist, length)
