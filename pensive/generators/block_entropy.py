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
        show_state_block_entropy: bool = True,
        show_block_state_entropy: bool = True,
        show_asymptote: bool = True,
        show_markov_order: bool = True,
        show_cryptic_order: bool = True,
        show_crypticity: bool = True,
        show_excess_entropy: bool = False,
        show_statistical_complexity: bool = False,
        show_entropy_rate_estimate: bool = False,
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


def block_entropy_diagram(machine: EpsilonMachine, max_length: int) -> BlockEntropyDiagram:
    """Compute finite-block entropy convergence curves for an epsilon-machine."""
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")

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

    entropy_rate = _entropy_rate(pi, symbol_matrices)
    statistical_complexity = _entropy(pi)
    excess_entropy = _excess_entropy(machine)
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


def plot_block_entropy_diagram(
    machine: EpsilonMachine,
    max_length: int,
    ax: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Compute and plot a block entropy diagram for ``machine``."""
    return block_entropy_diagram(machine, max_length).plot(ax=ax, **kwargs)


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


def _excess_entropy(machine: EpsilonMachine) -> float:
    bidirectional = machine.bidirectional_epsilon_machine()
    joint = bidirectional.joint_distribution()
    if not joint:
        return 0.0

    plus: dict[Any, float] = {}
    minus: dict[Any, float] = {}
    for (plus_state, minus_state), mass in joint.items():
        plus[plus_state] = plus.get(plus_state, 0.0) + float(mass)
        minus[minus_state] = minus.get(minus_state, 0.0) + float(mass)

    return _entropy(plus.values()) + _entropy(minus.values()) - _entropy(joint.values())


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
