

#!/usr/bin/env python3
# tsp_data_gen.py
# Simple, configurable TSP data generator for TRM-style mid-trajectory supervision.
# Teachers:
#   - Cheapest Insertion (constructive steps).
#   - 2-Opt Best-Improvement (local improvement steps).
# Outputs JSONL with one supervised step per line.

from __future__ import annotations
import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


# ---------------------------
# Core geometry and utilities
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            dist[i, j] = dist[j, i] = d
    return dist

def tour_length(dist: np.ndarray, tour: List[int]) -> float:
    total = 0.0
    m = len(tour)
    for i in range(m):
        a = tour[i]
        b = tour[(i + 1) % m]
        total += float(dist[a, b])
    return total

def rotate_tour_to_start_at_zero(tour: List[int]) -> List[int]:
    """Canonicalize: rotate so 0 is first; choose orientation with smaller second element."""
    if 0 not in tour:
        return tour
    n = len(tour)
    idx0 = tour.index(0)
    rot = tour[idx0:] + tour[:idx0]
    rev = [rot[0]] + list(reversed(rot[1:]))
    if len(rot) == 1:
        return rot
    return rot if rot[1] < rev[1] else rev

def two_opt_swap(tour: List[int], i: int, j: int) -> List[int]:
    assert 0 <= i < j < len(tour)
    return tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]

def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> List[int]:
    n = dist.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda k: dist[cur, k])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return rotate_tour_to_start_at_zero(tour)

def farthest_pair(dist: np.ndarray) -> Tuple[int, int]:
    n = dist.shape[0]
    best = (0, 1)
    bestd = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            if d > bestd:
                bestd = d
                best = (i, j)
    return best

def initial_triangle(dist: np.ndarray) -> List[int]:
    a, b = farthest_pair(dist)
    n = dist.shape[0]
    best_c, best_perim = None, -1.0
    for c in range(n):
        if c in (a, b):
            continue
        perim = float(dist[a, b] + dist[b, c] + dist[c, a])
        if perim > best_perim:
            best_perim = perim
            best_c = c
    tri = [a, b, best_c]
    return rotate_tour_to_start_at_zero(tri)


# ---------------------------
# Teachers and trajectories
# ---------------------------

@dataclass
class InsertStep:
    tour_before: List[int]
    city: int
    position: int
    cost_before: float
    cost_after: float

def cheapest_insertion_trajectory(dist: np.ndarray) -> List[InsertStep]:
    n = dist.shape[0]
    tour = initial_triangle(dist)
    visited = set(tour)
    steps: List[InsertStep] = []
    while len(visited) < n:
        best_city = None
        best_pos = None
        best_delta = float("inf")
        cur_cost = tour_length(dist, tour)
        m = len(tour)
        for c in range(n):
            if c in visited:
                continue
            for pos in range(m):
                u = tour[pos]
                v = tour[(pos + 1) % m]
                delta = float(dist[u, c] + dist[c, v] - dist[u, v])
                if delta < best_delta:
                    best_delta = delta
                    best_city = c
                    best_pos = pos + 1
        assert best_city is not None and best_pos is not None
        new_tour = tour[:best_pos] + [best_city] + tour[best_pos:]
        new_cost = tour_length(dist, new_tour)
        steps.append(InsertStep(
            tour_before=tour.copy(),
            city=best_city,
            position=best_pos,
            cost_before=cur_cost,
            cost_after=new_cost
        ))
        tour = new_tour
        visited.add(best_city)
    return steps

@dataclass
class TwoOptStep:
    tour_before: List[int]
    i: Optional[int]
    j: Optional[int]
    cost_before: float
    cost_after: float
    stop: bool

def two_opt_best_improvement_trajectory(dist: np.ndarray, start_tour: List[int]) -> List[TwoOptStep]:
    n = dist.shape[0]
    tour = start_tour.copy()
    steps: List[TwoOptStep] = []
    while True:
        best_delta = 0.0
        best_i = None
        best_j = None
        cur_cost = tour_length(dist, tour)
        # Search all valid pairs (avoid adjacent edges)
        for i in range(1, n - 2):
            a = tour[i - 1]; b = tour[i]
            for j in range(i + 1, n - 1):
                c = tour[j]; d = tour[(j + 1) % n]
                if b == c or a == d:
                    continue
                delta = (float(dist[a, c]) + float(dist[b, d])
                         - float(dist[a, b]) - float(dist[c, d]))
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_i, best_j = i, j
        if best_i is None:
            steps.append(TwoOptStep(
                tour_before=tour.copy(),
                i=None, j=None,
                cost_before=cur_cost, cost_after=cur_cost, stop=True
            ))
            break
        new_tour = two_opt_swap(tour, best_i, best_j)
        new_cost = tour_length(dist, new_tour)
        steps.append(TwoOptStep(
            tour_before=tour.copy(),
            i=best_i, j=best_j,
            cost_before=cur_cost, cost_after=new_cost, stop=False
        ))
        tour = new_tour
    return steps

def random_2opt_noise(tour: List[int], k: int, rng: random.Random) -> List[int]:
    n = len(tour)
    cur = tour.copy()
    for _ in range(k):
        i = rng.randint(1, n - 3)
        j = rng.randint(i + 1, n - 2)
        cur = two_opt_swap(cur, i, j)
    return cur


# ---------------------------
# Sampling strategies
# ---------------------------

def choose_indices(num_steps: int, k: int, strategy: str, rng: random.Random) -> List[int]:
    """Pick k indices from [0, num_steps-1] according to a simple strategy."""
    if k <= 0 or num_steps == 0:
        return []
    k = min(k, num_steps)
    thirds = max(1, num_steps // 3)
    if strategy == "all":
        return list(range(num_steps))
    if strategy == "early":
        pool = list(range(min(thirds, num_steps)))
    elif strategy == "late":
        start = max(0, num_steps - thirds)
        pool = list(range(start, num_steps))
    elif strategy == "mid":
        start = thirds
        end = min(num_steps, 2 * thirds)
        if start >= end:
            pool = list(range(num_steps))
        else:
            pool = list(range(start, end))
    else:
        pool = list(range(num_steps))  # uniform
    if len(pool) <= k:
        return sorted(pool)
    return sorted(rng.sample(pool, k=k))


# ---------------------------
# JSONL example builders
# ---------------------------

def build_insert_example(pid: str, coords: np.ndarray, step: InsertStep, idx: int, total: int) -> Dict[str, Any]:
    return {
        "mode": "insert",
        "problem_id": pid,
        "n": int(coords.shape[0]),
        "coords": coords.tolist(),
        "tour_partial": step.tour_before,
        "action": {"city": step.city, "position": step.position},
        "cost_before": float(step.cost_before),
        "cost_after": float(step.cost_after),
        "teacher": "cheapest_insertion",
        "step_index": int(idx),
        "total_steps": int(total)
    }

def build_two_opt_example(pid: str, coords: np.ndarray, step: TwoOptStep, idx: int, total: int) -> Dict[str, Any]:
    action = {"stop": True} if step.stop else {"i": step.i, "j": step.j}
    return {
        "mode": "two_opt",
        "problem_id": pid,
        "n": int(coords.shape[0]),
        "coords": coords.tolist(),
        "tour_full": step.tour_before,
        "action": action,
        "cost_before": float(step.cost_before),
        "cost_after": float(step.cost_after),
        "teacher": "two_opt_best",
        "step_index": int(idx),
        "total_steps": int(total)
    }


# ---------------------------
# Dataset writer
# ---------------------------

def write_dataset(
    out_path: str,
    num_problems: int,
    n_min: int,
    n_max: int,
    seed: int,
    p_constructive: float,
    ins_per: int,
    opt_per: int,
    step_sample: str,
    noise_min: int,
    noise_max: int,
    include_stop_prob: float
) -> None:
    set_seed(seed)
    rng = random.Random(seed)

    with open(out_path, "w") as f:
        for pid in range(num_problems):
            n = rng.randint(n_min, n_max)
            coords = np.random.rand(n, 2).astype(np.float32)
            dist = pairwise_euclidean(coords)
            name = f"p{pid}_n{n}_s{rng.randint(1, 10**9)}"

            # Constructive part
            if rng.random() < p_constructive:
                steps = cheapest_insertion_trajectory(dist)
                sel = choose_indices(num_steps=len(steps), k=ins_per, strategy=step_sample, rng=rng)
                for si in sel:
                    ex = build_insert_example(name, coords, steps[si], idx=si, total=len(steps))
                    f.write(json.dumps(ex) + "\n")

            # Improvement part
            start_tour = nearest_neighbor_tour(dist, start=0)
            k_noise = rng.randint(noise_min, noise_max)
            noisy = random_2opt_noise(start_tour, k=k_noise, rng=rng)
            imp_steps = two_opt_best_improvement_trajectory(dist, start_tour=noisy)

            # indices to sample (exclude STOP unless we decide to include it)
            indices = [i for i, s in enumerate(imp_steps) if not s.stop]
            if not indices:
                indices = [len(imp_steps) - 1]  # if only stop exists
            sel2 = choose_indices(num_steps=len(indices), k=opt_per, strategy=step_sample, rng=rng)
            for j, si in enumerate(sel2):
                real_idx = indices[si]
                ex2 = build_two_opt_example(name, coords, imp_steps[real_idx], idx=real_idx, total=len(imp_steps))
                f.write(json.dumps(ex2) + "\n")
            # Add a STOP example sometimes
            if imp_steps and imp_steps[-1].stop and rng.random() < include_stop_prob:
                stop_ex = build_two_opt_example(name, coords, imp_steps[-1], idx=len(imp_steps) - 1, total=len(imp_steps))
                f.write(json.dumps(stop_ex) + "\n")


# ---------------------------
# CLI helpers: demo, stats, show
# ---------------------------

def demo_once(n: int, seed: int, noise_k: int) -> None:
    set_seed(seed)
    coords = np.random.rand(n, 2).astype(np.float32)
    dist = pairwise_euclidean(coords)
    print(f"Demo TSP with n={n}, seed={seed}")
    print("City coordinates (index: x y):")
    for i, (x, y) in enumerate(coords):
        print(f"  {i:2d}: {x:.4f} {y:.4f}")

    # Cheapest insertion full trajectory
    print("\nCheapest Insertion trajectory:")
    ins_steps = cheapest_insertion_trajectory(dist)
    for t, s in enumerate(ins_steps):
        delta = s.cost_after - s.cost_before
        print(f"  step {t:2d}: insert city {s.city} at position {s.position}; "
              f"len {s.cost_before:.4f} -> {s.cost_after:.4f} (delta {delta:+.4f})")
        print(f"    partial tour: {s.tour_before}")

    # 2-opt improvement from a degraded NN tour
    print("\n2-Opt Best-Improvement trajectory:")
    start = nearest_neighbor_tour(dist, start=0)
    degraded = random_2opt_noise(start, k=noise_k, rng=random.Random(seed + 7))
    print(f"  start (degraded) length {tour_length(dist, degraded):.4f}")
    imp_steps = two_opt_best_improvement_trajectory(dist, start_tour=degraded)
    for t, s in enumerate(imp_steps):
        if s.stop:
            print(f"  step {t:2d}: STOP at length {s.cost_before:.4f}")
            break
        print(f"  step {t:2d}: 2-opt swap on indices (i={s.i}, j={s.j}); "
              f"len {s.cost_before:.4f} -> {s.cost_after:.4f} (delta {s.cost_after - s.cost_before:+.4f})")
        print(f"    tour before: {s.tour_before}")

def dataset_stats(path: str, max_lines: int = 10_000_000) -> None:
    counts = {"insert": 0, "two_opt": 0}
    n_vals: List[int] = []
    step_pos: List[float] = []  # relative step index
    deltas: List[float] = []

    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= max_lines:
                break
            ex = json.loads(line)
            mode = ex["mode"]
            counts[mode] = counts.get(mode, 0) + 1
            n_vals.append(ex["n"])
            if ex["total_steps"] > 0:
                step_pos.append(ex["step_index"] / ex["total_steps"])
            d = ex["cost_after"] - ex["cost_before"]
            deltas.append(d)

    if not n_vals:
        print("Empty dataset.")
        return

    def mean(x): return sum(x) / max(1, len(x))

    print(f"Lines read: {len(n_vals)}")
    print(f"Mode counts: {counts}")
    print(f"n: min={min(n_vals)} max={max(n_vals)} mean={mean(n_vals):.2f}")
    print(f"step_index/total_steps: mean={mean(step_pos):.3f} "
          f"(0=early, ~0.5=mid, 1=late)")
    print(f"mean delta (cost_after - cost_before): {mean(deltas):+.4f}")

def show_examples(path: str, k: int = 5) -> None:
    lines: List[str] = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            lines.append(line.strip())
    if not lines:
        print("Empty dataset.")
        return
    k = min(k, len(lines))
    print(f"Showing {k} examples from {len(lines)} lines:")
    rng = random.Random(123)
    idxs = sorted(rng.sample(range(len(lines)), k=k))
    for idx in idxs:
        ex = json.loads(lines[idx])
        print("\n----------------------------")
        print(f"line {idx}: mode={ex['mode']} n={ex['n']} step={ex['step_index']}/{ex['total_steps']} teacher={ex['teacher']}")
        if ex["mode"] == "insert":
            print(f"tour_partial: {ex['tour_partial']}")
            print(f"action: insert city {ex['action']['city']} at position {ex['action']['position']}")
        else:
            if ex["action"].get("stop", False):
                print("action: STOP")
            else:
                print(f"action: 2-opt (i={ex['action']['i']}, j={ex['action']['j']})")
            print(f"tour_full: {ex['tour_full']}")
        print(f"cost_before -> cost_after: {ex['cost_before']:.4f} -> {ex['cost_after']:.4f}")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="TSP data generator for TRM-style mid-trajectory supervision.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="Generate a JSONL dataset")
    g.add_argument("--out", type=str, required=True)
    g.add_argument("--num_problems", type=int, default=5000)
    g.add_argument("--n_min", type=int, default=6)
    g.add_argument("--n_max", type=int, default=12)
    g.add_argument("--seed", type=int, default=1234)

    g.add_argument("--p_constructive", type=float, default=0.6, help="Probability to include cheapest-insertion samples for a problem")
    g.add_argument("--ins_per", type=int, default=6, help="Number of insertion snapshots per problem")
    g.add_argument("--opt_per", type=int, default=6, help="Number of 2-opt snapshots per problem")
    g.add_argument("--step_sample", type=str, default="uniform", choices=["uniform", "early", "mid", "late", "all"])

    g.add_argument("--noise_min", type=int, default=2, help="Minimum 2-opt noise swaps applied to NN tour before improving")
    g.add_argument("--noise_max", type=int, default=6, help="Maximum 2-opt noise swaps")
    g.add_argument("--include_stop_prob", type=float, default=0.25, help="Probability to include the final STOP example")

    d = sub.add_parser("demo", help="Print one problem with full trajectories")
    d.add_argument("--n", type=int, default=10)
    d.add_argument("--seed", type=int, default=2024)
    d.add_argument("--noise_k", type=int, default=4)

    s = sub.add_parser("stats", help="Quick stats over a JSONL dataset")
    s.add_argument("--path", type=str, required=True)

    sh = sub.add_parser("show", help="Pretty print a few random examples from a JSONL dataset")
    sh.add_argument("--path", type=str, required=True)
    sh.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "gen":
        write_dataset(
            out_path=args.out,
            num_problems=args.num_problems,
            n_min=args.n_min,
            n_max=args.n_max,
            seed=args.seed,
            p_constructive=args.p_constructive,
            ins_per=args.ins_per,
            opt_per=args.opt_per,
            step_sample=args.step_sample,
            noise_min=args.noise_min,
            noise_max=args.noise_max,
            include_stop_prob=args.include_stop_prob
        )
        print(f"Wrote dataset to {args.out}")
    elif args.cmd == "demo":
        demo_once(n=args.n, seed=args.seed, noise_k=args.noise_k)
    elif args.cmd == "stats":
        dataset_stats(args.path)
    elif args.cmd == "show":
        show_examples(args.path, k=args.k)


if __name__ == "__main__":
    main()
