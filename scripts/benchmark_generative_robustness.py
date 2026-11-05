#!/usr/bin/env python3
"""Benchmark generative_model robustness on standard processes.

Run from the sofic repo root::

    python scripts/benchmark_generative_robustness.py
    python scripts/benchmark_generative_robustness.py --quick
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from sofic.examples import golden_mean, golden_mean_bidirectional
from sofic.examples.epsilon_machines import from_symbol_matrices
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.generative_search import generative_model, generative_model_from_bidirectional


def _h(p: float) -> float:
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _rll() -> BidirectionalEpsilonMachine:
    machine = from_symbol_matrices(
        ("A", "B", "C"),
        (0, 1),
        {0: np.array([[0.5, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]), 1: np.array([[0, 0.5, 0], [0, 0, 0], [0, 0, 0]])},
    )
    return BidirectionalEpsilonMachine.from_forward(machine)


def _interleaved(eps: float = 0.1) -> BidirectionalEpsilonMachine:
    T = np.array([[1 - eps, eps], [eps, 1 - eps]])
    names = ["00", "01", "10", "11"]
    mats = {x: np.zeros((4, 4)) for x in (0, 1)}
    for i, (a, b) in enumerate(names):
        for c in "01":
            mats[int(c)][i, names.index(b + c)] = T[int(a), int(c)]
    return BidirectionalEpsilonMachine.from_forward(from_symbol_matrices(names, (0, 1), mats))


@dataclass
class Case:
    name: str
    machine: object
    target_upper: float | None
    expect_certified: bool | None


def _run_case(case: Case, *, starts: int, exact_starts: int, sweep: int) -> dict:
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    if isinstance(case.machine, BidirectionalEpsilonMachine):
        result = generative_model_from_bidirectional(
            case.machine,
            starts=starts,
            exact_starts=exact_starts,
            entropy_sweep_starts=sweep,
            universal_coupled=True,
            early_theorem_g_exit=True,
            rng=rng,
        )
    else:
        result = generative_model(
            case.machine,
            starts=starts,
            exact_starts=exact_starts,
            entropy_sweep_starts=sweep,
            rng=rng,
        )
    elapsed = time.perf_counter() - t0
    summary = result.stages[-1] if result.stages else {}
    row = {
        "name": case.name,
        "seconds": elapsed,
        "upper": result.upper,
        "lower": result.lower,
        "gap": result.gap,
        "certified": result.certified,
        "lower_certified": result.lower_certified,
        "upper_source": result.upper_source,
        "n_g_generating": result.n_g_optima_generating,
    }
    if case.target_upper is not None:
        row["upper_err"] = abs(result.upper - case.target_upper)
    if case.expect_certified is not None and result.certified != case.expect_certified:
        row["certified_mismatch"] = case.expect_certified
    if summary.get("stage") == "reliability summary":
        row["summary_ok"] = True
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Fewer multistarts")
    args = parser.parse_args()
    starts = 6 if args.quick else 12
    exact = 12 if args.quick else 24
    sweep = 20 if args.quick else 40

    gm_target = _h(1 / 3)
    cases = [
        Case("golden_mean_forward", golden_mean(0.5), gm_target, True),
        Case("golden_mean_bidir", golden_mean_bidirectional(0.5), gm_target, True),
        Case("rll", _rll(), 1.5, False),
        Case("interleaved_0.1", _interleaved(0.1), 2 * _h(4 / 9), None),
    ]

    print(f"{'case':<22} {'time':>6} {'upper':>8} {'lower':>8} {'gap':>8} {'cert':>5} {'source':<40}")
    print("-" * 110)
    for case in cases:
        row = _run_case(case, starts=starts, exact_starts=exact, sweep=sweep)
        print(
            f"{row['name']:<22} {row['seconds']:6.2f} {row['upper']:8.5f} {row['lower']:8.5f} "
            f"{row['gap']:8.5f} {str(row['certified']):>5} {row['upper_source'][:40]:<40}"
        )
        if "upper_err" in row and row["upper_err"] > 1e-4:
            print(f"  WARNING upper_err={row['upper_err']:.2e}")
        if row.get("certified_mismatch") is not None:
            print(f"  WARNING expected certified={row['certified_mismatch']}")


if __name__ == "__main__":
    main()
