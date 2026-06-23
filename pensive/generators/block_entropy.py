"""Block entropy convergence diagrams for epsilon-machines."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pensive.generators.epsilon_machine import EpsilonMachine

_TOL = 1e-15


@dataclass(frozen=True)
class BlockEntropyDiagram:
    """Finite-block entropy curves and computational-mechanics annotations.

    The curves follow the entropy-convergence diagrams used for finite-state
    epsilon-machines: block entropy ``H[X_0:L]``, state-block entropy
    ``H[S_0, X_0:L]``, block-state entropy ``H[X_0:L, S_L]``, the linear
    asymptote ``E + h_mu L``, and the finite-length crypticity estimate
    ``chi(L)``.
    """

    lengths: np.ndarray
    block_entropy: np.ndarray
    state_block_entropy: np.ndarray
    block_state_entropy: np.ndarray
    entropy_asymptote: np.ndarray
    crypticity_estimate: np.ndarray
    entropy_rate_estimate: np.ndarray
    entropy_rate: float
    excess_entropy: float
    statistical_complexity: float
    crypticity: float
    markov_order: int | float
    cryptic_order: int | float

    def plot(
        self,
        ax: Any | None = None,
        *,
        show_block_entropy: bool = True,
        show_state_block_entropy: bool = False,
        show_block_state_entropy: bool = True,
        show_asymptote: bool = True,
        show_markov_order: bool = True,
        show_cryptic_order: bool = True,
        show_crypticity: bool = False,
        show_excess_entropy: bool = False,
        show_statistical_complexity: bool = False,
        show_entropy_rate_estimate: bool = False,
        show_grid: bool = True,
        show_legend: bool = True,
        title: str | None = None,
        marker: str = "o",
    ) -> Any:
        """Plot selected block entropy diagram features on ``ax``.

        ``matplotlib`` is imported only when this method is called. The return
        value is the axes object used for plotting.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        if show_block_entropy:
            ax.plot(self.lengths, self.block_entropy, marker=marker, label=r"$H[X_{0:L}]$")
        if show_state_block_entropy:
            ax.plot(
                self.lengths,
                self.state_block_entropy,
                marker=marker,
                label=r"$H[S_0, X_{0:L}]$",
            )
        if show_block_state_entropy:
            ax.plot(
                self.lengths,
                self.block_state_entropy,
                marker=marker,
                label=r"$H[X_{0:L}, S_L]$",
            )
        if show_asymptote:
            ax.plot(
                self.lengths,
                self.entropy_asymptote,
                linestyle="--",
                color="black",
                label=r"$E + h_\mu L$",
            )
        if show_crypticity:
            ax.plot(
                self.lengths,
                self.crypticity_estimate,
                marker=marker,
                linestyle="-.",
                label=r"$\chi(L)$",
            )
            ax.axhline(
                self.crypticity,
                linestyle=":",
                color="tab:purple",
                label=r"$\chi$",
            )
        if show_excess_entropy:
            ax.axhline(self.excess_entropy, linestyle=":", color="tab:green", label=r"$E$")
        if show_statistical_complexity:
            ax.axhline(
                self.statistical_complexity,
                linestyle=":",
                color="tab:brown",
                label=r"$C_\mu$",
            )
        if show_entropy_rate_estimate:
            ax.plot(
                self.lengths,
                self.entropy_rate_estimate,
                marker=marker,
                linestyle=":",
                label=r"$h_\mu(L)$",
            )
            ax.axhline(self.entropy_rate, linestyle="--", color="tab:gray", label=r"$h_\mu$")

        if show_markov_order:
            self._plot_order_line(ax, self.markov_order, "tab:red", r"$R$")
        if show_cryptic_order:
            self._plot_order_line(ax, self.cryptic_order, "tab:orange", r"$k_\chi$")

        if show_grid:
            ax.set_axisbelow(True)
            ax.grid(True, color="0.85", linewidth=0.8)
        else:
            ax.grid(False)
        ax.set_xlabel("Block length L")
        ax.set_ylabel("Information (bits)")
        if title is not None:
            ax.set_title(title)
        if show_legend:
            ax.legend()
        return ax

    def _plot_order_line(self, ax: Any, order: int | float, color: str, symbol: str) -> None:
        if not _is_finite_order(order):
            return
        if float(order) < float(self.lengths[0]) or float(order) > float(self.lengths[-1]):
            return
        label = f"{symbol} = {int(order)}"
        ax.axvline(float(order), linestyle="--", color=color, alpha=0.75, label=label)


@dataclass(frozen=True)
class BlockEntropyEstimates:
    """Finite-block estimates for computational-mechanics quantities."""

    lengths: np.ndarray
    block_entropy: np.ndarray
    state_block_entropy: np.ndarray
    block_state_entropy: np.ndarray
    entropy_asymptote: np.ndarray
    entropy_rate_estimate: np.ndarray
    residual_entropy: np.ndarray
    residual_entropy_rate_estimate: np.ndarray
    excess_entropy_lower: np.ndarray
    excess_entropy_upper: np.ndarray
    excess_entropy_estimate: np.ndarray
    crypticity_estimate: np.ndarray
    synchronization_estimate: np.ndarray
    reverse_synchronization_estimate: np.ndarray
    transient_information_estimate: np.ndarray
    predictability_gain_estimate: np.ndarray
    oracular_information_estimate: np.ndarray
    gauge_information_estimate: np.ndarray
    entropy_rate: float
    excess_entropy: float
    statistical_complexity: float
    crypticity: float
    predicted_information: float
    bound_information: float
    ephemeral_information: float

    @property
    def h_mu(self) -> float:
        return self.entropy_rate

    @property
    def E(self) -> float:
        return self.excess_entropy

    @property
    def r_mu(self) -> float:
        return self.ephemeral_information

    @property
    def b_mu(self) -> float:
        return self.bound_information

    def information_anatomy(self) -> dict[str, float]:
        """Return finite-block estimates using the exact anatomy key names."""
        return {
            "rho_mu": self.predicted_information,
            "bound_mu": self.bound_information,
            "ephemeral_mu": self.ephemeral_information,
            "entropy_rate": self.entropy_rate,
            "excess_entropy": self.excess_entropy,
            "crypticity": self.crypticity,
        }


def block_entropy_diagram(machine: EpsilonMachine, max_length: int) -> BlockEntropyDiagram:
    """Compute finite-block entropy convergence curves for an epsilon-machine."""
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")

    lengths, block_entropy, state_block_entropy, block_state_entropy, pi, symbol_matrices = _block_entropy_curves(
        machine,
        max_length,
    )

    entropy_rate = _entropy_rate(pi, symbol_matrices)
    statistical_complexity = _entropy(pi)
    excess_entropy = _excess_entropy(machine, entropy_rate=entropy_rate, block_entropy=block_entropy)
    crypticity = statistical_complexity - excess_entropy
    entropy_asymptote = excess_entropy + entropy_rate * lengths
    crypticity_estimate = state_block_entropy - block_state_entropy
    entropy_rate_estimate = np.empty(max_length + 1, dtype=float)
    entropy_rate_estimate[0] = math.nan
    if max_length > 0:
        entropy_rate_estimate[1:] = np.diff(block_entropy)

    return BlockEntropyDiagram(
        lengths=lengths,
        block_entropy=block_entropy,
        state_block_entropy=state_block_entropy,
        block_state_entropy=block_state_entropy,
        entropy_asymptote=entropy_asymptote,
        crypticity_estimate=crypticity_estimate,
        entropy_rate_estimate=entropy_rate_estimate,
        entropy_rate=entropy_rate,
        excess_entropy=excess_entropy,
        statistical_complexity=statistical_complexity,
        crypticity=crypticity,
        markov_order=machine.markov_order(),
        cryptic_order=machine.cryptic_order(),
    )


def block_entropy_estimates(
    machine: EpsilonMachine,
    max_length: int,
    *,
    entropy_rate: float | None = None,
    use_exact: bool = True,
) -> BlockEntropyEstimates:
    """Approximate information quantities from finite block entropies.

    Exact excess entropy is used when available; ``use_exact`` is retained for
    API compatibility.
    """
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")
    if entropy_rate is None and max_length < 1:
        raise ValueError("max_length must be positive when entropy_rate is not supplied")

    lengths, block_entropy, state_block_entropy, block_state_entropy, pi, _symbol_matrices = _block_entropy_curves(
        machine,
        max_length,
    )

    entropy_rate_estimate = _entropy_rate_estimates(block_entropy)
    h_mu = float(entropy_rate if entropy_rate is not None else entropy_rate_estimate[-1])
    statistical_complexity = _entropy(pi)

    h_mu_l = h_mu * lengths
    excess_entropy_lower = block_entropy - h_mu_l
    excess_entropy_upper = block_state_entropy - h_mu_l
    excess_entropy_estimate = 0.5 * (excess_entropy_lower + excess_entropy_upper)
    excess_entropy = _estimated_excess_entropy(machine, excess_entropy_estimate, use_exact=use_exact)

    entropy_asymptote = excess_entropy + h_mu_l
    crypticity_estimate = state_block_entropy - block_state_entropy
    crypticity = float(crypticity_estimate[-1]) if crypticity_estimate.size else 0.0
    synchronization = block_state_entropy - block_entropy
    reverse_synchronization = state_block_entropy - block_entropy
    transient_information = np.cumsum(entropy_asymptote - block_entropy)
    predictability_gain = _predictability_gain(block_entropy)
    oracular_information = statistical_complexity + h_mu_l - state_block_entropy
    gauge_information = statistical_complexity - excess_entropy - crypticity_estimate - oracular_information

    residual_entropy = _residual_entropy_curve(machine, max_length)
    residual_entropy_rate_estimate = _entropy_rate_estimates(residual_entropy)
    r_mu = float(residual_entropy_rate_estimate[-1]) if max_length > 0 else math.nan
    b_mu = h_mu - r_mu
    rho_mu = float(block_entropy[1] - h_mu) if max_length >= 1 else math.nan

    return BlockEntropyEstimates(
        lengths=lengths,
        block_entropy=block_entropy,
        state_block_entropy=state_block_entropy,
        block_state_entropy=block_state_entropy,
        entropy_asymptote=entropy_asymptote,
        entropy_rate_estimate=entropy_rate_estimate,
        residual_entropy=residual_entropy,
        residual_entropy_rate_estimate=residual_entropy_rate_estimate,
        excess_entropy_lower=excess_entropy_lower,
        excess_entropy_upper=excess_entropy_upper,
        excess_entropy_estimate=excess_entropy_estimate,
        crypticity_estimate=crypticity_estimate,
        synchronization_estimate=synchronization,
        reverse_synchronization_estimate=reverse_synchronization,
        transient_information_estimate=transient_information,
        predictability_gain_estimate=predictability_gain,
        oracular_information_estimate=oracular_information,
        gauge_information_estimate=gauge_information,
        entropy_rate=h_mu,
        excess_entropy=excess_entropy,
        statistical_complexity=statistical_complexity,
        crypticity=crypticity,
        predicted_information=rho_mu,
        bound_information=b_mu,
        ephemeral_information=r_mu,
    )


def plot_block_entropy_diagram(
    machine: EpsilonMachine,
    max_length: int,
    ax: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Compute and plot a block entropy diagram for ``machine``."""
    return block_entropy_diagram(machine, max_length).plot(ax=ax, **kwargs)


def _block_entropy_curves(
    machine: EpsilonMachine,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[Any, np.ndarray]]:
    pi, symbol_matrices = _stationary_symbol_matrices(machine)
    lengths = np.arange(max_length + 1, dtype=int)
    block_entropy = np.zeros(max_length + 1, dtype=float)
    state_block_entropy = np.zeros(max_length + 1, dtype=float)
    block_state_entropy = np.zeros(max_length + 1, dtype=float)

    word_matrices = [np.eye(len(pi), dtype=float)]
    terminal = np.ones(len(pi), dtype=float)
    alphabet = tuple(sorted(symbol_matrices, key=repr))

    for length in range(max_length + 1):
        block_probs: list[float] = []
        state_block_probs: list[float] = []
        block_state_probs: list[float] = []

        for matrix in word_matrices:
            end_mass = pi @ matrix
            start_conditioned = matrix @ terminal
            block_probs.append(float(end_mass.sum()))
            state_block_probs.extend(float(prob) for prob in pi * start_conditioned)
            block_state_probs.extend(float(prob) for prob in end_mass)

        block_entropy[length] = _entropy(block_probs)
        state_block_entropy[length] = _entropy(state_block_probs)
        block_state_entropy[length] = _entropy(block_state_probs)

        if length < max_length:
            word_matrices = [matrix @ symbol_matrices[symbol] for matrix in word_matrices for symbol in alphabet]

    return lengths, block_entropy, state_block_entropy, block_state_entropy, pi, symbol_matrices


def _stationary_symbol_matrices(machine: EpsilonMachine) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    from pensive.generators.hmm_inference import _emission_transition_tensors

    pi = machine.stationary_distribution()
    _, raw_matrices = _emission_transition_tensors(machine)
    n = len(pi)
    zero = np.zeros((n, n), dtype=float)
    matrices = {
        symbol: np.array(raw_matrices.get(symbol, zero), dtype=float) for symbol in machine.observation_alphabet
    }
    return pi, matrices


def _entropy_rate(pi: np.ndarray, symbol_matrices: dict[Any, np.ndarray]) -> float:
    rate = 0.0
    for state_index, state_probability in enumerate(pi):
        if state_probability <= _TOL:
            continue
        symbol_probs = [float(matrix[state_index].sum()) for matrix in symbol_matrices.values()]
        rate += float(state_probability) * _entropy(symbol_probs)
    return rate


def _excess_entropy(
    machine: EpsilonMachine,
    *,
    entropy_rate: float | None = None,
    block_entropy: np.ndarray | None = None,
) -> float:
    try:
        return float(machine.to_bidirectional().excess_entropy())
    except Exception:
        return _block_entropy_excess_entropy(machine, entropy_rate=entropy_rate, block_entropy=block_entropy)


def _block_entropy_excess_entropy(
    machine: EpsilonMachine,
    *,
    entropy_rate: float | None = None,
    block_entropy: np.ndarray | None = None,
) -> float:
    order = machine.markov_order()
    if _is_finite_order(order):
        markov_order = int(order)
        h_mu = entropy_rate
        if h_mu is None:
            curves = _block_entropy_curves(machine, markov_order)
            _block_entropy = curves[1]
            pi = curves[4]
            symbol_matrices = curves[5]
            h_mu = _entropy_rate(pi, symbol_matrices)
            block_entropy = _block_entropy
        if block_entropy is not None and markov_order < len(block_entropy):
            H_R = float(block_entropy[markov_order])
        else:
            curves = _block_entropy_curves(machine, markov_order)
            H_R = float(curves[1][markov_order])
        return float(H_R - markov_order * h_mu)

    if block_entropy is None or len(block_entropy) == 0:
        raise RuntimeError("cannot estimate excess entropy without finite block entropies")
    h_mu = entropy_rate
    if h_mu is None:
        _lengths, _block_entropy, _state_block_entropy, _block_state_entropy, pi, symbol_matrices = (
            _block_entropy_curves(machine, len(block_entropy) - 1)
        )
        h_mu = _entropy_rate(pi, symbol_matrices)
        block_entropy = _block_entropy
    length = len(block_entropy) - 1
    return float(block_entropy[-1] - length * h_mu)


def _estimated_excess_entropy(machine: EpsilonMachine, estimate: np.ndarray, *, use_exact: bool) -> float:
    try:
        return float(machine.excess_entropy())
    except Exception:
        pass
    return float(estimate[-1]) if estimate.size else 0.0


def _entropy_rate_estimates(entropies: np.ndarray) -> np.ndarray:
    estimates = np.empty(len(entropies), dtype=float)
    estimates[0] = math.nan
    if len(entropies) > 1:
        estimates[1:] = np.diff(entropies)
    return estimates


def _predictability_gain(block_entropy: np.ndarray) -> np.ndarray:
    gain = np.full(len(block_entropy), math.nan, dtype=float)
    if len(block_entropy) > 2:
        gain[2:] = np.diff(block_entropy, n=2)
    return gain


def _residual_entropy_curve(machine: EpsilonMachine, max_length: int) -> np.ndarray:
    residual = np.zeros(max_length + 1, dtype=float)
    for length in range(1, max_length + 1):
        residual[length] = _residual_entropy(machine.word_probabilities(length))
    return residual


def _residual_entropy(distribution: dict[tuple[Any, ...], float]) -> float:
    if not distribution:
        return 0.0
    first = next(iter(distribution))
    length = len(first)
    if length == 0:
        return 0.0

    total = sum(float(prob) for prob in distribution.values())
    if total <= _TOL:
        return 0.0
    normalized = {word: float(prob) / total for word, prob in distribution.items() if prob > _TOL}
    joint_entropy = _entropy(normalized.values())

    marginal_entropies = 0.0
    for index in range(length):
        marginal: dict[tuple[Any, ...], float] = {}
        for word, prob in normalized.items():
            reduced = word[:index] + word[index + 1 :]
            marginal[reduced] = marginal.get(reduced, 0.0) + prob
        marginal_entropies += _entropy(marginal.values())

    return max(0.0, float(length * joint_entropy - marginal_entropies))


def _entropy(probabilities: Iterable[float]) -> float:
    probs = np.asarray([prob for prob in probabilities if prob > _TOL], dtype=float)
    if probs.size == 0:
        return 0.0
    total = float(probs.sum())
    if total <= _TOL:
        return 0.0
    probs = probs / total
    return float(-np.sum(probs * np.log2(probs)))


def _is_finite_order(order: int | float) -> bool:
    return not isinstance(order, float) or math.isfinite(order)
