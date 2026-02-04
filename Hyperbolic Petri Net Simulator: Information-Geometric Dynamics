#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Canonical Hyperbolic Petri Net Simulator – Research Version

Fully self-contained simulation framework for studying
information-geometric dynamics of discrete state systems
with Pareto-Nash efficiency and admissibility invariants.

Features:
- Deterministic Petri Net simulation
- Hyperbolic embedding (Poincaré Disk)
- Admissibility / Rational Inattention check
- SHA3-256 fingerprint of trajectory
- Dynamic random seed for reproducibility per run
- Real-time ASCII summary and visualization

Author: Eric Ren
License: MIT
Date: 2026
"""

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ─── Petri Net Definition ────────────────────────────────────────────────

@dataclass
class PetriNet:
    incidence: List[List[int]]
    initial_marking: List[int]

    @property
    def n_places(self) -> int:
        return len(self.initial_marking)

    @property
    def n_trans(self) -> int:
        return len(self.incidence[0]) if self.incidence else 0

    def can_fire(self, t: int, marking: List[int]) -> bool:
        return all(marking[p] + self.incidence[p][t] >= 0 for p in range(self.n_places))

    def fire(self, t: int, marking: List[int]) -> List[int]:
        return [marking[p] + self.incidence[p][t] for p in range(self.n_places)]

# ─── Hyperbolic Embedding ───────────────────────────────────────────────

def embed_marking(marking: List[int], radius_scale: float = 0.38, jitter: float = 0.008
                  ) -> Tuple[float, float]:
    total = sum(marking)
    if total == 0: return 0.0, 0.0
    theta, weight_sum = 0.0, 0.0
    n = len(marking)
    for i, cnt in enumerate(marking):
        if cnt == 0: continue
        frac = cnt / total
        angle = 2.0 * math.pi * i / max(1, n - 1)
        theta += frac * angle
        weight_sum += frac
    if weight_sum > 0: theta /= weight_sum
    r = math.tanh(radius_scale * math.log1p(total))
    if jitter != 0.0: r += random.uniform(-jitter, jitter) * (1.0 - r)
    r = max(0.0, min(0.999, r))
    return r * math.cos(theta), r * math.sin(theta)

def poincare_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a; bx, by = b
    aa, bb = ax*ax + ay*ay, bx*bx + by*by
    diff2 = (ax - bx)**2 + (ay - by)**2
    denom = (1.0 - aa) * (1.0 - bb)
    if denom < 1e-10: return 999.0
    arg = 1.0 + 2.0 * diff2 / denom
    return math.acosh(max(arg, 1.0000001))

# ─── Admissibility / Rational Inattention ───────────────────────────────

def admissible(initial: List[int], current: List[int], steps: int,
               factor: float = 0.92) -> Tuple[bool, float, float]:
    d = poincare_distance(embed_marking(initial), embed_marking(current))
    bound = factor * 3.4 * math.log1p(steps + 1)
    return d <= bound, d, bound

# ─── ASCII Visualization ────────────────────────────────────────────────

def ascii_bar(value: int, max_val: int, width: int = 22, char: str = "█") -> str:
    if max_val <= 0: return " " * width
    filled = int(width * min(value, max_val) / max_val)
    return char * filled + " " * (width - filled)

def print_marking_bars(marking: List[int], labels: Optional[List[str]] = None):
    if not marking: return
    max_tokens = max(marking) + 1
    lbls = labels or [f"p{i}" for i in range(len(marking))]
    print("  Marking:")
    for name, cnt in zip(lbls, marking):
        bar = ascii_bar(cnt, max_tokens)
        print(f"   {name:>3} | {bar} {cnt:2d}")

def print_trajectory_summary(radii: List[float], total_distances: List[float]):
    if not radii:
        print("  (no steps performed)")
        return
    n_r = len(radii)
    show_count = min(12, n_r)
    last_d = total_distances[-1] if total_distances else None
    for k in range(show_count):
        idx = n_r - show_count + k
        r = radii[idx]
        d_str = f"{last_d:6.3f}" if last_d is not None else "  —  "
        print(f" {idx+1:3d} | r = {r:.5f}    total d ≈ {d_str}")

# ─── Plot ───────────────────────────────────────────────────────────────

def optional_plot(steps: List[int], radii: List[float], distances: List[float], prefix: Optional[str] = None):
    if not MATPLOTLIB_AVAILABLE or not steps: return
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    ax1.plot(steps, radii, "o-", lw=1.2, ms=4, label="radius", color="#1f77b4")
    ax1.set_ylabel("Disk radius")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    if distances:
        dist_steps = steps[:len(distances)]
        ax2.plot(dist_steps, distances, "s--", lw=1.3, ms=5,
                 label="total hyp. dist", color="#d62728")
    ax2.set_ylabel("Hyperbolic distance")
    fig.suptitle("Hyperbolic Petri Net Trajectory", fontsize=13)
    fig.legend(loc="upper center", ncol=2, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0,0.04,1,0.94])
    if prefix:
        path = f"{prefix}_trajectory.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")
    else:
        plt.show()

# ─── SHA3-256 Fingerprint ──────────────────────────────────────────────

def fingerprint(net: PetriNet, trace: List[int], final: List[int]) -> str:
    h = hashlib.sha3_256()
    for row in net.incidence:
        for val in row:
            h.update((val & 0xFF).to_bytes(1, byteorder='little'))
    for t in trace:
        h.update(t.to_bytes(2, byteorder='little', signed=False))
    for cnt in final:
        h.update(cnt.to_bytes(4, byteorder='little', signed=False))
    return h.hexdigest()[:32]

# ─── Simulation Core ────────────────────────────────────────────────────

def run_simulation(net: PetriNet, max_steps: int = 800, output_prefix: Optional[str] = None, verbose: bool = True, seed: Optional[int] = None):
    if seed is None:
        seed = random.randint(0, 2**31-1)
    random.seed(seed)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_file, json_path = None, None
    if output_prefix:
        out_file = open(f"{output_prefix}.txt","w",encoding="utf-8")
        json_path = f"{output_prefix}.json"

    def log(msg: str = "", end="\n"):
        if verbose: print(msg, end=end)
        if out_file: out_file.write(msg+end); out_file.flush()

    log("Canonical Hyperbolic Petri Net Simulator")
    log(f"  started:      {ts}  |  seed: {seed}")
    log(f"  places:       {net.n_places}    transitions: {net.n_trans}")
    log(f"  initial:      {net.initial_marking}")
    log("─"*66)

    marking = net.initial_marking[:]
    trace: List[int] = []
    radii: List[float] = []
    total_distances: List[float] = []
    step_numbers: List[int] = []

    prev_total_tokens = sum(marking)
    last_full_check = 0

    for step in range(1, max_steps+1):
        fired = False
        for t in range(net.n_trans):
            if net.can_fire(t, marking):
                marking = net.fire(t, marking)
                trace.append(t)
                fired = True

                curr_embed = embed_marking(marking)
                r = math.hypot(*curr_embed)
                radii.append(r)
                step_numbers.append(len(trace))

                curr_total = sum(marking)
                if (step - last_full_check >= 5 or curr_total != prev_total_tokens or step == max_steps):
                    d = poincare_distance(embed_marking(net.initial_marking), curr_embed)
                    total_distances.append(d)
                    ok, _, bound = admissible(net.initial_marking, marking, len(trace))
                    if not ok:
                        log(f"\n!!! INVARIANT VIOLATION at step {step} !!!")
                        log(f"  total distance = {d:.5f} > bound {bound:.5f}")
                        raise RuntimeError("Admissibility invariant broken")
                    last_full_check = step
                    prev_total_tokens = curr_total

                if verbose and step % 50 == 0:
                    mstr = " ".join(f"{x:2d}" for x in marking)
                    log(f" {step:4d} | t{t} → [{mstr}]   r ≈ {r:.5f}")
                break
        if not fired:
            log(f"\n→ Terminated naturally after {len(trace)} steps (no enabled transitions)")
            break

    # ── Final summary ─────────────────────────
    final_r = math.hypot(*embed_marking(marking))
    final_d = poincare_distance(embed_marking(net.initial_marking), embed_marking(marking))
    ok, _, bound = admissible(net.initial_marking, marking, len(trace))
    log("─"*66)
    log("FINAL RESULT")
    print_marking_bars(marking)
    log(f"\n  steps performed:           {len(trace)}")
    log(f"  final radius:              {final_r:.5f}")
    log(f"  total hyperbolic distance: {final_d:.5f}  ≤  {bound:.5f}   {'OK' if ok else 'VIOLATED'}")
    log(f"  fingerprint (SHA3-256 / 128-bit prefix): {fingerprint(net, trace, marking)}")
    print_trajectory_summary(radii, total_distances)

    if output_prefix:
        summary = {
            "timestamp_utc": ts,
            "seed": seed,
            "places": net.n_places,
            "transitions": net.n_trans,
            "initial_marking": net.initial_marking,
            "final_marking": marking,
            "trace": trace,
            "fingerprint": fingerprint(net, trace, marking),
            "radii": [round(r,5) for r in radii],
            "total_distances": [round(d,5) for d in total_distances],
            "terminated_naturally": len(trace) < max_steps
        }
        with open(json_path,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
        log(f"\nResults saved → {output_prefix}.txt  +  {json_path}")

    if out_file: out_file.close()
    optional_plot(step_numbers, radii, total_distances, output_prefix)

# ─── Example Petri Net ────────────────────────────────────────────────
EXAMPLE_NET = PetriNet(
    incidence=[
        [-1,  1,  0,  0],
        [ 1, -1, -1,  0],
        [ 0,  1, -1, -1],
        [ 0,  0,  1,  0]
    ],
    initial_marking=[20, 10, 0, 0]
)

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Canonical Hyperbolic Petri Net Simulator")
    parser.add_argument("--out","-o",type=str,default=None,help="output prefix")
    parser.add_argument("--max-steps",type=int,default=1000,help="maximum simulation steps")
    parser.add_argument("--quiet",action="store_true",help="suppress console output")
    args = parser.parse_args()

    run_simulation(EXAMPLE_NET, max_steps=args.max_steps, output_prefix=args.out, verbose=not args.quiet)

if __name__ == "__main__":
    main()
