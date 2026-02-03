#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperbolic Petri Embedding + Dual Invariant — Clean & Robust (2026)
===================================================================

Educational proof-of-concept:
- discrete invariant   : local token/step coherence
- continuous invariant : global hyperbolic geometry bound

Zero external dependencies. MIT License.
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class PetriNet:
    incidence: List[List[int]]
    initial_marking: List[int]

    @property
    def n_places(self) -> int:
        return len(self.initial_marking)

    @property
    def n_trans(self) -> int:
        return len(self.incidence[0]) if self.incidence and self.incidence[0] else 0

    def can_fire(self, t: int, marking: List[int]) -> bool:
        if t < 0 or t >= self.n_trans:
            return False
        return all(marking[p] + self.incidence[p][t] >= 0 for p in range(self.n_places))

    def fire(self, t: int, marking: List[int]) -> List[int]:
        if not self.can_fire(t, marking):
            raise ValueError(f"Transition t{t} cannot fire from marking {marking}")
        new_marking = marking[:]
        for p in range(self.n_places):
            new_marking[p] += self.incidence[p][t]
        return new_marking


def embed_marking(marking: List[int], radius_scale: float = 0.32) -> Tuple[float, float]:
    total = sum(m for m in marking if m > 0)
    if total <= 0:
        return 0.0, 0.0

    theta = 0.0
    wsum = 0.0
    n = len(marking)
    for i, cnt in enumerate(marking):
        if cnt <= 0:
            continue
        frac = cnt / total
        angle = 2.0 * math.pi * i / max(1, n - 1)
        theta += frac * angle
        wsum += frac

    if wsum <= 0:
        return 0.0, 0.0

    theta /= wsum
    r = math.tanh(radius_scale * math.log1p(total))
    return r * math.cos(theta), r * math.sin(theta)


def poincare_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    aa = ax*ax + ay*ay
    bb = bx*bx + by*by
    diff2 = (ax - bx)**2 + (ay - by)**2
    denom = (1 - aa) * (1 - bb)

    if denom < 1e-12:
        return 0.0 if aa < 1e-8 and bb < 1e-8 else 999.0

    arg = 1.0 + 2.0 * diff2 / denom
    arg = max(arg, 1.0 + 1e-10)  # safe for acosh
    try:
        return math.acosh(arg)
    except ValueError:
        return 999.0 if arg > 1.0001 else 0.0


def hyperbolic_admissible(
    m0: List[int], mk: List[int], steps: int, factor: float = 0.9
) -> Tuple[bool, float, float]:
    d = poincare_distance(embed_marking(m0), embed_marking(mk))
    bound = factor * 2.8 * math.log1p(max(0, steps))
    return d <= bound, d, bound


def discrete_coherence_admissible(
    prev: List[int], curr: List[int], transition: int
) -> Tuple[bool, str]:
    fails = []
    # Example domain-specific checks — tune to your net
    if len(prev) >= 2 and len(curr) >= 2:
        if prev[1] > 0 and curr[1] < prev[1] - 3:
            fails.append("excessive buffer drain")
        if prev[1] >= 4 and curr[1] == 0:
            fails.append("sudden buffer zero-out")
    if transition == 3 and len(prev) >= 3 and prev[2] < 2:
        fails.append("premature completion")

    return not fails, "; ".join(fails) if fails else "OK"


def dual_invariants_hold(
    m0: List[int],
    prev: List[int],
    curr: List[int],
    t: int,
    steps: int,
    hyp_factor: float = 0.9
) -> Tuple[bool, str]:
    h_ok, h_d, h_b = hyperbolic_admissible(m0, curr, steps, hyp_factor)
    d_ok, d_reason = discrete_coherence_admissible(prev, curr, t)

    if h_ok and d_ok:
        return True, "both ✓"

    fails = []
    if not h_ok:
        fails.append(f"hyp {h_d:.3f} > bound {h_b:.3f}")
    if not d_ok:
        fails.append(f"discrete: {d_reason}")

    return False, "; ".join(fails)


def ascii_bar(val: float, maxv: float, width: int = 24) -> str:
    if maxv <= 0:
        return "─" * width
    filled = int(width * min(val, maxv) / maxv)
    return "█" * filled + "─" * (width - filled)


def print_final_summary(
    radii: List[float],
    hyp_dists: List[float],
    trace: List[int],
    final_marking: List[int],
    terminated: bool,
    violation: bool
):
    if not radii:
        print("  No steps executed.")
        return

    max_r = max(radii) + 1e-8
    max_d = max(hyp_dists) + 1e-8
    max_token = max(final_marking) + 1 if final_marking else 1

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│                  FINAL HYPERBOLIC TRAJECTORY                │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"{'Step':>5} │ {'Radius':^22} │ {'Hyp Dist':^22} │ Action")
    print("├─────────────────────────────────────────────────────────────┤")

    for i, (r, d, t) in enumerate(zip(radii, hyp_dists, trace), 1):
        rbar = ascii_bar(r, max_r, 20)
        dbar = ascii_bar(d, max_d, 20)
        print(f"{i:5d} │ {rbar} {r:5.3f} │ {dbar} {d:5.3f} │ t{t}")

    print("└─────────────────────────────────────────────────────────────┘")

    print("\nFinal marking:")
    for i, cnt in enumerate(final_marking):
        bar = ascii_bar(cnt, max_token, 28)
        print(f"  p{i:2} : {bar} {cnt:3}")

    print(f"\nTrace: {' → '.join(f't{x}' for x in trace)}")
    print(f"Length: {len(trace)}   Terminated naturally: {terminated}")
    if violation:
        print("→ Stopped due to invariant violation")


def compute_fingerprint(net: PetriNet, trace: List[int], final: List[int]) -> str:
    h = hashlib.sha3_512()
    for row in net.incidence:
        for v in row:
            h.update(v.to_bytes(2, "little", signed=True))
    for t in trace:
        h.update(t.to_bytes(2, "little", signed=False))
    for m in final:
        h.update(m.to_bytes(4, "little", signed=False))
    return h.hexdigest()[:64]


def run_simulation(
    net: PetriNet,
    max_steps: int = 150,
    output_prefix: Optional[str] = None,
    allow_violation: bool = False,
    json_only: bool = False,
    verbose_progress: bool = False,
) -> int:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    exit_code = 0
    violation_occurred = False
    terminated = False

    txtfile = None
    json_path = None
    if output_prefix:
        try:
            txtfile = open(f"{output_prefix}.txt", "w", encoding="utf-8")
            json_path = f"{output_prefix}.json"
        except Exception as e:
            print(f"Error opening output files: {e}", file=sys.stderr)
            return 2

    def log(msg: str, force: bool = False) -> None:
        if json_only and not force:
            return
        print(msg)
        if txtfile:
            txtfile.write(msg + "\n")
            txtfile.flush()

    log(f"Hyperbolic Petri + Dual Invariant  •  {ts}", force=True)
    log(f"Places: {net.n_places}   Transitions: {net.n_trans}")
    log(f"Initial: {net.initial_marking}")

    marking = net.initial_marking[:]
    trace: List[int] = []
    radii: List[float] = []
    hyp_dists: List[float] = []

    origin_embed = embed_marking(net.initial_marking)

    try:
        for step in range(1, max_steps + 1):
            fired = False
            for t in range(net.n_trans):
                if not net.can_fire(t, marking):
                    continue

                prev_marking = marking[:]
                prev_embed = embed_marking(prev_marking)

                marking = net.fire(t, prev_marking)
                trace.append(t)
                fired = True

                curr_embed = embed_marking(marking)
                curr_r = math.hypot(*curr_embed)

                radii.append(curr_r)
                hyp_dists.append(poincare_distance(origin_embed, curr_embed))

                ok, reason = dual_invariants_hold(
                    net.initial_marking, prev_marking, marking, t, len(trace)
                )

                if not ok:
                    msg = f"Invariant violation at step {step} (t{t}): {reason}"
                    log(f"\n!!! {msg} !!!")
                    violation_occurred = True
                    if not allow_violation:
                        exit_code = 1
                        log("Stopping (use --allow-violation to continue)")
                        break

                break  # greedy: fire first enabled transition

            if violation_occurred and not allow_violation:
                break

            if not fired:
                terminated = True
                total_initial = sum(net.initial_marking)
                if len(marking) > 3 and marking[3] == total_initial:
                    log("→ All items processed (goal state reached)")
                else:
                    log("→ No enabled transitions remain — terminated naturally")
                break

            if verbose_progress and step % 25 == 0:
                log(f"[step {step:3}] r={curr_r:.3f} d={hyp_dists[-1]:.3f}")

    except Exception as e:
        log(f"\nRuntime error: {type(e).__name__}: {e}", force=True)
        exit_code = 2

    fp = compute_fingerprint(net, trace, marking)

    summary = {
        "timestamp_utc": ts,
        "places": net.n_places,
        "transitions": net.n_trans,
        "incidence": net.incidence,
        "initial_marking": net.initial_marking,
        "final_marking": marking,
        "trace": trace,
        "fingerprint": fp,
        "terminated_normally": terminated,
        "violation_occurred": violation_occurred,
        "exit_code": exit_code,
        "radii": [round(x, 4) for x in radii],
        "hyp_distances": [round(x, 4) for x in hyp_dists],
    }

    if json_path:
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            log(f"Saved: {json_path}")
        except Exception as e:
            log(f"Error saving JSON: {e}", force=True)

    if txtfile:
        txtfile.close()

    if not json_only:
        print("\n" + "═" * 65)
        print("FINAL RESULT")
        print_final_summary(radii, hyp_dists, trace, marking, terminated, violation_occurred)
        print(f"\nFingerprint (SHA3-512 truncated): {fp}")
        if exit_code == 0:
            print("→ Success")
        elif exit_code == 1:
            print("→ Stopped on invariant violation")
        else:
            print("→ Runtime error occurred")

    if json_only:
        print(json.dumps(summary, indent=2))

    return exit_code


# ─── Example net (producer → buffer → consumer → completed) ────────────────

EXAMPLE_NET = PetriNet(
    incidence=[
        [-1,  1,  0,  0],   # p0: producer
        [ 1, -1, -1,  0],   # p1: buffer
        [ 0,  1, -1, -1],   # p2: consumer
        [ 0,  0,  1,  0],   # p3: completed
    ],
    initial_marking=[12, 5, 0, 0],
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-Invariant Hyperbolic Petri Net simulation",
        epilog="Examples:\n"
               "  python dual_hpe.py --out run1\n"
               "  python dual_hpe.py --allow-violation --max-steps 300 --verbose-progress"
    )
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="output prefix (.txt + .json)")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="maximum number of simulation steps")
    parser.add_argument("--allow-violation", action="store_true",
                        help="continue even after invariant violation")
    parser.add_argument("--json-only", action="store_true",
                        help="output only JSON to stdout (no visual summary)")
    parser.add_argument("--verbose-progress", action="store_true",
                        help="print progress every ~25 steps")
    args = parser.parse_args()

    try:
        sys.exit(run_simulation(
            EXAMPLE_NET,
            max_steps=args.max_steps,
            output_prefix=args.out,
            allow_violation=args.allow_violation,
            json_only=args.json_only,
            verbose_progress=args.verbose_progress,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Fatal: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
