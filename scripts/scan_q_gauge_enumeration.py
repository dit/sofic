#!/usr/bin/env python3
"""Max I[S+0 : X0 : S-1 | S-0, S+1] over topological ε-machines.

Uniform-outdegree topological machines from Johnson et al. (2010).
Workers use SIGALRM + a low mixed-state cap so hard reverses fail fast.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sofic.generators.topological_epsilon_enumeration import (
    idfa_string_to_epsilon_machine,
    iter_topological_epsilon_strings,
)

# Cap reverse MSP size so pathological machines fail immediately.
_MAX_MIXED_STATES = 128
_WORKER_TIMEOUT_S = 2.0


def _q_gauge(em) -> float:
    d = em.to_bidirectional().step_distribution()
    pmf: dict[tuple, float] = defaultdict(float)
    for outcome, p in zip(d.outcomes, d.pmf, strict=True):
        pmf[tuple(outcome)] += float(p)

    def marg(idxs: tuple[int, ...]) -> dict[tuple, float]:
        out: dict[tuple, float] = defaultdict(float)
        for outcome, p in pmf.items():
            out[tuple(outcome[i] for i in idxs)] += p
        return out

    def H(idxs: tuple[int, ...]) -> float:
        return -sum(p * math.log(p, 2) for p in marg(idxs).values() if p > 0)

    def I2(a: tuple[int, ...], b: tuple[int, ...], g: tuple[int, ...]) -> float:
        abg = tuple(sorted(set(a) | set(b) | set(g)))
        return H(a + g) + H(b + g) - H(abg) - (H(g) if g else 0.0)

    return I2((0,), (2,), (1, 3)) - I2((0,), (2,), (1, 3, 4))


class _AlarmError(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _AlarmError("worker timeout")


def _init_worker(max_mixed: int, timeout_s: float) -> None:
    global _MAX_MIXED_STATES, _WORKER_TIMEOUT_S
    _MAX_MIXED_STATES = max_mixed
    _WORKER_TIMEOUT_S = timeout_s
    # Patch mixed-state construction default for this process.
    import sofic.generators.epsilon_construction as ec
    import sofic.generators.mixed_state_construction as msc

    _orig_build_msp = msc.build_mixed_state_presentation

    def _capped_msp(hmm, initial_mixed_state=None, max_states=_MAX_MIXED_STATES, **kwargs):
        return _orig_build_msp(
            hmm,
            initial_mixed_state=initial_mixed_state,
            max_states=max_states,
            **kwargs,
        )

    msc.build_mixed_state_presentation = _capped_msp  # type: ignore[assignment]

    # epsilon_construction.build_epsilon_machine -> MSP; ensure it picks up patched symbol
    if hasattr(ec, "build_mixed_state_presentation"):
        ec.build_mixed_state_presentation = _capped_msp  # type: ignore[attr-defined]


def eval_one(payload: tuple[tuple[int, ...], int, int]) -> tuple[str, float | None, object]:
    transitions, n, k = payload
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, _WORKER_TIMEOUT_S)
    try:
        em = idfa_string_to_epsilon_machine(transitions, n=n, k=k)
        value = float(_q_gauge(em))
        return ("ok", value, transitions)
    except _AlarmError:
        return ("timeout", None, transitions)
    except Exception as exc:  # noqa: BLE001
        return ("fail", None, f"{type(exc).__name__}: {exc}"[:100])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--max-mixed", type=int, default=128)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    n, k = args.n, args.k
    out = args.out or Path(f"/tmp/q_gauge_n{n}_k{k}.json")
    t0 = time.time()

    print(f"listing topological strings n={n} k={k} ...", flush=True)
    strings = list(iter_topological_epsilon_strings(k=k, n=n))
    total = len(strings)
    print(
        f"count={total} workers={args.workers} timeout={args.timeout}s max_mixed={args.max_mixed}",
        flush=True,
    )

    best_v = -np.inf
    best_str = None
    ok = fail = timeout = 0
    top: list[tuple[float, tuple[int, ...]]] = []

    def save(status: str) -> None:
        out.write_text(
            json.dumps(
                {
                    "status": status,
                    "n": n,
                    "k": k,
                    "total": total,
                    "ok": ok,
                    "fail": fail,
                    "timeout": timeout,
                    "processed": ok + fail + timeout,
                    "best": best_v if np.isfinite(best_v) else None,
                    "best_string": list(best_str) if best_str is not None else None,
                    "top": [{"value": v, "string": list(s)} for v, s in top[:25]],
                    "elapsed_s": time.time() - t0,
                    "max_mixed": args.max_mixed,
                    "timeout_s": args.timeout,
                },
                indent=2,
            )
        )

    payloads = ((s, n, k) for s in strings)
    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.max_mixed, args.timeout),
        maxtasksperchild=200,
    ) as pool:
        for i, (status, value, meta) in enumerate(
            pool.imap_unordered(eval_one, payloads, chunksize=args.chunksize),
            start=1,
        ):
            if status == "ok" and value is not None:
                ok += 1
                if value > best_v:
                    best_v = value
                    best_str = meta
                    print(f"  new best {best_v:.8f}  string={best_str}", flush=True)
                top.append((value, meta))  # type: ignore[arg-type]
                top.sort(key=lambda t: -t[0])
                del top[50:]
            elif status == "timeout":
                timeout += 1
            else:
                fail += 1

            if i % 500 == 0 or i == total:
                rate = i / max(time.time() - t0, 1e-9)
                eta_m = (total - i) / max(rate, 1e-9) / 60.0
                print(
                    f"  {i}/{total} ok={ok} fail={fail} timeout={timeout} "
                    f"best={best_v if np.isfinite(best_v) else float('nan'):.6f} "
                    f"{rate:.1f}/s ETA={eta_m:.1f}m",
                    flush=True,
                )
                save("running")

    save("done")
    print(
        f"DONE max={best_v} string={best_str} ok={ok} fail={fail} timeout={timeout} "
        f"elapsed={time.time() - t0:.1f}s -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
